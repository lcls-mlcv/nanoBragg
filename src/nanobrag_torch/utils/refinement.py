"""L-BFGS-B refinement of an ExperimentModel or SerialExperiment.

The parameters differ by orders of magnitude in how far one unit moves the image: a
0.15 degree misorientation is 0.015 of tanh(misset/10 deg) and ruins a shot, while log G
can start wrong by a factor of 10. One step size cannot suit both, so a plain torch.optim
loop stalls: on a shot hopper refines to 0.0013 degrees in 112 iterations, Adam and
torch.optim.LBFGS on the raw parameters stopped with G 34% off. Rescaling to comparable
sensitivity, as hopper does, and using SciPy's L-BFGS-B gives hopper's answer.
"""
from typing import Callable, Optional

import numpy as np
import torch

# One unit of the optimizer's variable, in each group's raw parameter.
DEFAULT_SCALES = {
    "scale": 1.0,
    "fluence": 1.0,
    "Ncells": 0.1,
    "eta": 0.1,
    "misset": 0.002,  # tanh(misset / 10 deg): 0.02 degrees
    "cell": 0.002,  # log a, b, c; raw angles
    "distance": 0.002,  # log distance
    "detz_shift": 0.1,  # mm
    "detector_rotation": 0.002,
    "twotheta": 0.002,
    "beam_center": 0.01,
    "Fhkl": 0.1,  # log F
    "background": 1.0,  # photons across the region
    "wavelength": 0.001,  # Angstrom, log scale
    "source_weights": 0.1,  # log weight
}


def scaled_parameters(model, scales: Optional[dict] = None) -> list:
    """(parameter, scale) for each trainable parameter, each once, in a stable order."""
    scales = {**DEFAULT_SCALES, **(scales or {})}
    shots = getattr(model, "shots", None)
    groups = (shots[0] if shots is not None else model).PARAMETER_GROUPS
    by_name = {name: group for group, names in groups.items() for name in names}
    chosen, seen = [], set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not parameter.numel() or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if shots is not None:
            name = name.split(".", 2)[2]  # shots.<i>.<block>.<parameter>
        chosen.append((parameter, scales[by_name[name]]))
    return chosen


def refine_lbfgsb(
    model,
    objective: Callable[[], torch.Tensor],
    scales: Optional[dict] = None,
    ftol: float = 1e-10,
    gtol: float = 1e-8,
    maxiter: int = 1000,
):
    """Minimise ``objective()``, a closure over the model, in the trainable parameters.

    Choose what refines with model.refine()/freeze() first. ``scales`` overrides entries
    of DEFAULT_SCALES by group name. The model is left at the last point evaluated, which
    is scipy's solution. Returns scipy's OptimizeResult, in the scaled variables.

        model.freeze().refine("scale", "misset", "Ncells", "cell")
        data = model.roi_values(image)
        refine_lbfgsb(model, lambda: gaussian_nll(model(roi_only=True), data, sigma))
    """
    from scipy.optimize import minimize

    chosen = scaled_parameters(model, scales)
    if not chosen:
        raise ValueError("nothing to refine: every parameter is frozen")
    starts = [parameter.detach().clone() for parameter, _ in chosen]

    def load(x):
        at = 0
        with torch.no_grad():
            for (parameter, scale), start in zip(chosen, starts):
                step = torch.as_tensor(x[at:at + parameter.numel()], device=parameter.device, dtype=parameter.dtype)
                parameter.copy_(start + scale * step.reshape(parameter.shape))
                at += parameter.numel()

    def value_and_gradient(x):
        load(x)
        for parameter, _ in chosen:
            parameter.grad = None
        loss = objective()
        loss.backward()
        gradient = [
            (torch.zeros_like(p) if p.grad is None else p.grad * scale).reshape(-1).detach().cpu().numpy()
            for p, scale in chosen
        ]
        return loss.item(), np.concatenate(gradient).astype(np.float64)

    result = minimize(value_and_gradient, np.zeros(sum(p.numel() for p, _ in chosen)), jac=True, method="L-BFGS-B",
                      options=dict(ftol=ftol, gtol=gtol, maxiter=maxiter, maxfun=3 * maxiter))
    load(result.x)
    return result
