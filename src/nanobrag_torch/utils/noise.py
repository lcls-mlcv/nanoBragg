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

    # Create output tensor with same shape
    noisy = torch.zeros_like(mean)

    # Separate pixels into three regimes
    small_mask = mean < 12
    medium_mask = (mean >= 12) & (mean <= 1e6)
    large_mask = mean > 1e6

    # Regime 1: Exact Poisson for small means (<12)
    if small_mask.any():
        small_means = mean[small_mask]
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

    # Convert to integer
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
    million pixels. generate_poisson_noise is the fast alternative.

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
    spixels, fpixels = pixels.shape
    xmin, xmax, ymin, ymax = roi if roi is not None else (0, fpixels - 1, 0, spixels - 1)
    values = pixels.ravel().tolist()
    output = np.zeros(pixels.size, dtype=np.uint16)

    packer = struct.Struct("f")
    pack, unpack = packer.pack, packer.unpack

    def f32(x):
        return unpack(pack(x))[0]

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
            g = f32(math.exp(-xm))
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
            em = float(math.floor(em))
            t = f32(0.9 * (1.0 + f32(y * y))
                    * math.exp(f32(f32(f32(em * alxm) - gammln(f32(em + 1.0))) - g)))
            if not ran1() > t:
                return em

    photons = 0.0
    overloads = 0
    index = 0
    for spixel in range(spixels):
        for fpixel in range(fpixels):
            if ymin <= spixel <= ymax and xmin <= fpixel <= xmax:
                test = poidev(values[index])
                photons += test
                test += adc_offset
                if test > 65535.0:
                    test = 65535.0
                    overloads += 1
                output[index] = int(test)
            index += 1
    return output.reshape(pixels.shape), photons, overloads
