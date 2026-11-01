"""
Noise generation utilities for nanoBragg PyTorch implementation.

This module implements Poisson noise generation following the spec requirements:
- Exact Poisson for means < 12
- Rejection sampling for means between 12 and 1e6
- Gaussian approximation for means > 1e6
"""

import math
import struct

import numpy as np
import torch
from typing import Optional, Tuple, Union


def _poisson_photons(mean: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Poisson draw per pixel, in poidev's three regimes. See generate_poisson_noise."""
    noisy = torch.zeros_like(mean)

    # Separate pixels into three regimes
    small_mask = mean < 12
    medium_mask = (mean >= 12) & (mean <= 1e6)
    large_mask = mean > 1e6

    # Regime 1: Exact Poisson for small means (<12)
    if small_mask.any():
        # A negative pixel draws 0, as in poidev
        small_means = mean[small_mask].clamp(min=0)
        # For small counts, use torch.poisson which is exact for small lambda
        small_noise = torch.poisson(small_means, generator=generator)
        noisy[small_mask] = small_noise

    # Regime 2: Rejection sampling for medium means (12 to 1e6)
    if medium_mask.any():
        medium_means = mean[medium_mask]
        # PyTorch's poisson uses rejection sampling internally for this range
        # We use it directly rather than reimplementing the rejection algorithm
        medium_noise = torch.poisson(medium_means, generator=generator)
        noisy[medium_mask] = medium_noise

    # Regime 3: Gaussian approximation for large means (>1e6)
    if large_mask.any():
        large_means = mean[large_mask]
        # For very large means, Poisson approaches N(mean, variance=mean)
        std = torch.sqrt(large_means)
        gaussian_noise = torch.normal(mean=large_means, std=std, generator=generator)
        # Round to nearest integer since we need photon counts
        noisy[large_mask] = torch.round(gaussian_noise)

    noisy[torch.isnan(mean)] = float('nan')
    return noisy


def generate_poisson_noise(
    mean: torch.Tensor,
    seed: Optional[int] = None,
    adc_offset: float = 40.0,
    readout_noise: float = 3.0,
    overload_value: float = 65535.0
) -> tuple[torch.Tensor, int]:
    """
    Generate Poisson noise following nanoBragg.c's three-regime approach.

    Per spec AT-NOISE-001:
    - For mean < 12: Use exact Poisson algorithm
    - For 12 <= mean <= 1e6: Use rejection sampling
    - For mean > 1e6: Use Gaussian approximation N(mean, variance=mean)

    C-Code Implementation Reference (from nanoBragg.c, lines 43726-43852):
    ```c
    // Three regimes for Poisson noise generation
    if(intfile_scale*floatimage[j] < 12) {
        // Exact Poisson for small means
        L = exp(-(intfile_scale*floatimage[j]));
        k=0; p=1.0;
        do {
            k++;
            p *= ((double) nrand(seed)/(double) RAND_MAX);
        } while (p > L);
        intimage[j] = k-1;
    } else if(intfile_scale*floatimage[j] < 1e6) {
        // Rejection sampling for medium means
        c = 0.767 - 3.36/(intfile_scale*floatimage[j]);
        // ... rejection algorithm ...
    } else {
        // Gaussian approximation for large means
        intimage[j] = poidev((intfile_scale*floatimage[j]),&seed);
    }
    ```

    Args:
        mean: Input intensities (photon counts) as float tensor
        seed: Random seed for reproducibility (None = use PyTorch default)
        adc_offset: ADC offset to add to all pixels
        readout_noise: Gaussian readout noise sigma
        overload_value: Maximum value before saturation

    Returns:
        Tuple of (noisy image, overload count)
        - noisy image: Integer tensor with Poisson noise, ADC offset, and clipping
        - overload_count: Number of pixels that exceeded overload_value
    """
    # Set up random generator with seed if provided
    if seed is not None:
        generator = torch.Generator(device=mean.device)
        generator.manual_seed(seed)
    else:
        generator = None

    noisy = _poisson_photons(mean, generator)

    # Add readout noise (Gaussian)
    if readout_noise > 0:
        readout = torch.normal(
            mean=torch.zeros_like(noisy),
            std=readout_noise,
            generator=generator
        )
        noisy = noisy + readout

    # Add ADC offset
    noisy = noisy + adc_offset

    # Ensure non-negative (clip at 0)
    noisy = torch.clamp(noisy, min=0)

    # Count overloads before clipping
    overload_count = (noisy > overload_value).sum().item()

    # Clip at overload value
    noisy = torch.clamp(noisy, max=overload_value)

    # Convert to integer. A NaN is pixel 0, as C's cast leaves it
    noisy = torch.nan_to_num(noisy, nan=0.0)
    noisy = torch.round(noisy).to(torch.int32)

    return noisy, overload_count


def lcg_random(
    seed: int,
    n: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Linear Congruential Generator (LCG) for C-compatible random numbers.

    Implements the same LCG as used in nanoBragg.c for exact reproducibility.

    C-Code Implementation Reference (from nanoBragg.c):
    ```c
    #define RAND_MAX 2147483647
    #define RAND_MULT 1103515245
    #define RAND_ADD 12345
    #define RAND_SEED (seed)

    static long int idum;
    double nrand(long *idum) {
        *idum = (RAND_MULT*(*idum)+RAND_ADD) & RAND_MAX;
        return (double) (*idum)/(double) RAND_MAX;
    }
    ```

    Args:
        seed: Initial seed value
        n: Number of random values to generate
        dtype: Data type for output tensor (default: torch.float32)
        device: Device for output tensor (default: CPU)

    Returns:
        Tensor of n random values in [0, 1)
    """
    # LCG parameters from nanoBragg.c
    RAND_MAX = 2147483647
    RAND_MULT = 1103515245
    RAND_ADD = 12345

    values = torch.zeros(n, dtype=dtype, device=device)
    current = seed

    for i in range(n):
        current = (RAND_MULT * current + RAND_ADD) & RAND_MAX
        values[i] = float(current) / float(RAND_MAX)

    return values

def roi_pixels(shape: Tuple[int, int], roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """True inside the inclusive (xmin, xmax, ymin, ymax), the test at nanoBragg.c:3251."""
    if roi is None:
        return np.ones(shape, dtype=bool)
    xmin, xmax, ymin, ymax = roi
    slow, fast = np.arange(shape[0]), np.arange(shape[1])
    return np.outer((slow >= ymin) & (slow <= ymax), (fast >= xmin) & (fast <= xmax))


def _noise_image_pixels(photons: np.ndarray, adc_offset: float, in_roi: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    nanoBragg.c:3257-3264 for the ROI's draws, as doubles: uint16 image, overloads.

    The clip is one-sided, so ``(unsigned short) test`` also meets negative numbers and NaN.
    That cast is formally undefined. The compiled form is a saturating conversion to int
    followed by the low 16 bits (fcvtzs on arm64, and cvttsd2si gives the same pixels on x86):
    -5 wraps to 65531, -1e12 and NaN become 0.
    """
    test = photons + adc_offset
    overloaded = test > 65535.0
    test = np.where(overloaded, 65535.0, test)
    in_int_range = test > -2147483649.0  # False for NaN
    counts = np.zeros(test.shape, dtype=np.uint16)
    counts[in_int_range] = (test[in_int_range].astype(np.int64) & 0xFFFF).astype(np.uint16)
    output = np.zeros(in_roi.shape, dtype=np.uint16)
    output[in_roi] = counts
    return output, int(overloaded.sum())


def fast_noise_image(
    image: torch.Tensor,
    seed: int,
    adc_offset: float = 40.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    The -fast_noise image: torch's sampler, not C's stream, with C's handling of the result.

    Pixels and overloads follow nanoBragg.c:3257-3264 as in c_noise_image, so the two differ
    only in which Poisson draw a pixel gets. An infinite pixel gets inf + inf*N(0, 1), as in
    C: an overload, or NaN, which is pixel 0. A NaN pixel stays NaN and a negative one draws 0.

    Args, returns: as c_noise_image.
    """
    generator = torch.Generator(device=image.device)
    generator.manual_seed(seed)
    in_roi = roi_pixels(tuple(image.shape), roi)
    photons = _poisson_photons(image.detach(), generator).double().cpu().numpy()[in_roi]
    output, overloads = _noise_image_pixels(photons, adc_offset, in_roi)
    return output, float(photons.sum()), overloads


def c_noise_image(
    image,
    seed: int,
    adc_offset: float = 40.0,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    Noise image as nanoBragg.c:3245-3265 makes it, bit for bit given its float image.

    Walks the pixels in file order drawing poidev from ran1, in the single precision the C
    code uses: the number of draws a pixel takes depends on the rounding, so anything looser
    loses the sequence for every pixel after it. This is a Python loop, about a second per
    million pixels. fast_noise_image is the fast alternative.

    As in C, variance is up to 16% high for means from about 6e5 to 1e6, where float32 rounds
    the acceptance exponent; those pixels clip at 65535, so only the photon count sees it.

    Args:
        image: Photons per pixel, (slow, fast). Read as float32, like C's floatimage.
        seed: The value given to -seed. C negates it, and ran1 starts from its magnitude,
            or from 1 for seed 0.
        adc_offset: Added to each pixel after the draw.
        roi: Inclusive (xmin, xmax, ymin, ymax). Pixels outside stay 0 and draw nothing.

    Returns:
        uint16 image, photons drawn, overloaded pixels.
    """
    pixels = np.asarray(image, dtype=np.float32)
    in_roi = roi_pixels(pixels.shape, roi)

    packer = struct.Struct("f")
    pack, unpack = packer.pack, packer.unpack

    def f32(x):
        try:
            return unpack(pack(x))[0]
        except OverflowError:  # a C cast gives inf, struct raises
            return math.copysign(math.inf, x)

    IA, IM, IQ, IR, NTAB = 16807, 2147483647, 127773, 2836, 32
    NDIV = 1 + (IM - 1) // NTAB
    AM = 1.0 / IM
    RNMX = 1.0 - 1.2e-7

    idum = abs(seed) or 1
    iv = [0] * NTAB
    for j in range(NTAB + 7, -1, -1):
        k = idum // IQ
        idum = IA * (idum - k * IQ) - IR * k
        if idum < 0:
            idum += IM
        if j < NTAB:
            iv[j] = idum
    iy = iv[0]

    def ran1():
        nonlocal idum, iy
        k = idum // IQ
        idum = IA * (idum - k * IQ) - IR * k
        if idum < 0:
            idum += IM
        j = iy // NDIV
        iy = iv[j]
        iv[j] = idum
        temp = f32(AM * iy)
        return f32(RNMX) if temp > RNMX else temp

    def gammln(xx):
        x = y = xx
        tmp = x + 5.5
        tmp -= (x + 0.5) * math.log(tmp)
        ser = 1.000000000190015
        for c in (76.18009172947146, -86.50532032941677, 24.01409824083091,
                  -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5):
            y += 1.0
            ser += c / y
        return f32(-tmp + math.log(2.5066282746310005 * ser / x))

    spare = None

    def gaussdev():
        nonlocal spare
        if spare is not None:
            value, spare = spare, None
            return value
        while True:
            v1 = f32(2.0 * ran1() - 1.0)
            v2 = f32(2.0 * ran1() - 1.0)
            rsq = f32(f32(v1 * v1) + f32(v2 * v2))
            if rsq < 1.0 and rsq != 0:
                break
        fac = f32(math.sqrt(-2.0 * math.log(rsq) / rsq))
        spare = f32(v1 * fac)
        return f32(v2 * fac)

    def poidev(xm):
        if xm > 1.0e6:
            return f32(xm + math.sqrt(xm) * gaussdev())
        if xm < 12.0:
            try:
                g = f32(math.exp(-xm))
            except OverflowError:  # a large negative pixel
                g = math.inf
            em = -1.0
            t = 1.0
            while True:
                em += 1.0
                t = f32(t * ran1())
                if not t > g:
                    return em
        sq = f32(math.sqrt(2.0 * xm))
        alxm = f32(math.log(xm))
        g = f32(f32(xm * alxm) - gammln(f32(xm + 1.0)))
        while True:
            while True:
                y = f32(math.tan(math.pi * ran1()))
                em = f32(f32(sq * y) + xm)
                if not em < 0.0:
                    break
            if em == em:  # NaN pixel: floor passes it on, and C returns it after these two draws
                em = float(math.floor(em))
            t = f32(0.9 * (1.0 + f32(y * y))
                    * math.exp(f32(f32(f32(em * alxm) - gammln(f32(em + 1.0))) - g)))
            if not ran1() > t:
                return em

    photons = 0.0
    draws = []
    for mean in pixels[in_roi].tolist():
        test = poidev(mean)
        photons += test
        draws.append(test)
    output, overloads = _noise_image_pixels(np.array(draws, dtype=np.float64), adc_offset, in_roi)
    return output, photons, overloads
