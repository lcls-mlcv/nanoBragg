"""Negative log likelihoods of a simulated model against measured pixels.

``model`` is in photons and includes the background, as ExperimentModel returns it, and
pixels overlap nowhere: model(roi_only=True) evaluates each pixel once, which is what
diffBragg's division by all_freq amounts to for overlapping shoeboxes.
"""
import math
from typing import Optional

import torch


def _select(model, data, trusted):
    if trusted is None:
        return model, data
    return model[trusted], data[trusted]


def gaussian_nll(
    model: torch.Tensor,
    data: torch.Tensor,
    sigma_rdout=3.0,
    trusted: Optional[torch.Tensor] = None,
    adu_per_photon: float = 1.0,
    variance_floor: float = 1e-6,
) -> torch.Tensor:
    """diffBragg hopper's target: Σ ½·(log 2πV + (data - model)²/V), V = model + sigma_rdout².

    Args:
        model: Modelled photons per pixel.
        data: Measured pixels, in ADU when ``adu_per_photon`` is given.
        sigma_rdout: Readout noise of the data, a number or per pixel, in the data's units.
        trusted: Boolean mask; only these pixels count.
        adu_per_photon: Detector gain; data and sigma_rdout are divided by it.
        variance_floor: Lower bound on V, which a negative background can push through
            zero. hopper has none and returns NaN there.
    """
    sigma = torch.as_tensor(sigma_rdout, device=model.device, dtype=model.dtype) / adu_per_photon
    if sigma.dim() and trusted is not None:
        sigma = sigma[trusted]
    model, data = _select(model, data.to(model.dtype) / adu_per_photon, trusted)
    variance = (model + sigma * sigma).clamp_min(variance_floor)
    return 0.5 * (torch.log(2 * math.pi * variance) + (data - model) ** 2 / variance).sum()


def poisson_nll(
    model: torch.Tensor,
    data: torch.Tensor,
    trusted: Optional[torch.Tensor] = None,
    adu_per_photon: float = 1.0,
    model_floor: float = 1e-6,
) -> torch.Tensor:
    """Poisson likelihood of photon counts: Σ model - data·log(model) + log(data!).

    Args:
        model: Modelled photons per pixel.
        data: Measured pixels, in ADU when ``adu_per_photon`` is given. Must not be
            negative where trusted: use gaussian_nll for pedestal-subtracted data.
        trusted: Boolean mask; only these pixels count.
        adu_per_photon: Detector gain; data is divided by it.
        model_floor: Lower bound on the model inside the logarithm.
    """
    model, data = _select(model, data.to(model.dtype) / adu_per_photon, trusted)
    if (data < 0).any():
        raise ValueError("poisson_nll needs counts >= 0: mask negative pixels with trusted, or use gaussian_nll")
    return (model - data * torch.log(model.clamp_min(model_floor)) + torch.lgamma(data + 1)).sum()
