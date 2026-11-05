"""L-BFGS-B refinement of an ExperimentModel or SerialExperiment.

The parameters differ by orders of magnitude in how far one unit moves the image: a
0.15 degree misorientation is 0.015 of tanh(misset/10 deg) and ruins a shot, while log G
can start wrong by a factor of 10. One step size cannot suit both, so a plain torch.optim
loop stalls: on a shot hopper refines to 0.0013 degrees in 112 iterations, Adam and
torch.optim.LBFGS on the raw parameters stopped with G 34% off. Rescaling to comparable
sensitivity, as hopper does, and using SciPy's L-BFGS-B gives hopper's answer.
"""
import os
import subprocess
import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch

# One unit of the optimizer's variable, in each group's raw parameter.
DEFAULT_SCALES = {
    "scale": 1.0,
    "fluence": 1.0,
    "Ncells": 0.1,
    "Ndef": 0.5,  # cells, additive off-diagonals
    "eta": 0.1,
    "misset": 0.002,  # tanh(misset / 10 deg): 0.02 degrees
    "cell": 0.002,  # log a, b, c; raw angles
    "distance": 0.002,  # log distance
    "detz_shift": 0.1,  # mm
    "panel_rotation": 0.001,  # radians, about each panel's own axes
    "panel_shift": 0.05,  # mm, lab x, y, z
    "detector_rotation": 0.002,
    "twotheta": 0.002,
    "beam_center": 0.01,
    # log F; scripts/benchmarks/serial_scaling.py converges sooner at 1.0 than at 0.1 or 0.3
    "Fhkl": 1.0,
    "background": 1.0,  # photons across the region
    "wavelength": 0.001,  # Angstrom, log scale
    "source_weights": 0.1,  # log weight
    # Optional groups (ExperimentModel.OPTIONAL_GROUPS)
    "diffuse": 0.1,  # the diffuse model's own parameters, logs in the reference models
    "diffuse_scale": 1.0,  # log k_d, as log G
}


def git_commit() -> str:
    """The commit this package runs from, "+dirty" if tracked files differ, "unknown" outside git.

    For result files, so that a number can be traced to the code that made it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + ("+dirty" if status.stdout.strip() else "")


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
    objective: Union[Callable[[], torch.Tensor], Sequence[Callable[[], torch.Tensor]]],
    scales: Optional[dict] = None,
    ftol: float = 1e-10,
    gtol: float = 1e-8,
    maxiter: int = 1000,
):
    """Minimise ``objective()``, a closure over the model, in the trainable parameters.

    Choose what refines with model.refine()/freeze() first. ``scales`` overrides entries
    of DEFAULT_SCALES by group name. The model is left at the last point evaluated, which
    is scipy's solution. Returns scipy's OptimizeResult, in the scaled variables.

    ``objective`` can be a list of closures whose sum is minimised. Each backward runs and
    frees its graph before the next closure starts, so graphs don't pile up across terms.
    Each closure must build everything it differentiates through itself, as
    ``serial([i], roi_only=True)`` does, which costs memory of its own: trainable Fhkl or a
    trainable shared detector are rebuilt per term and kept alive by each model until its
    next run, about 435 MB a shot for pixel coordinates on a 16M detector.

        model.freeze().refine("scale", "misset", "Ncells", "cell")
        data = model.roi_values(image)
        refine_lbfgsb(model, lambda: gaussian_nll(model(roi_only=True), data, sigma))
    """
    from scipy.optimize import minimize

    chosen = scaled_parameters(model, scales)
    if not chosen:
        raise ValueError("nothing to refine: every parameter is frozen")
    starts = [parameter.detach().clone() for parameter, _ in chosen]
    terms = [objective] if callable(objective) else list(objective)
    if not terms:
        raise ValueError("nothing to minimise: the list of objective terms is empty")

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
        loss = 0.0
        for term in terms:
            value = term()
            if value.requires_grad:  # a term can miss every trainable parameter, as in a sum
                value.backward()
            loss += value.item()
        gradient = [
            (torch.zeros_like(p) if p.grad is None else p.grad * scale).reshape(-1).detach().cpu().numpy()
            for p, scale in chosen
        ]
        return loss, np.concatenate(gradient).astype(np.float64)

    if any(p.dtype != torch.float64 for p, _ in chosen):
        warnings.warn("refine_lbfgsb needs float64: float32 losses can tie under ftol and stop it early")
    result = minimize(value_and_gradient, np.zeros(sum(p.numel() for p, _ in chosen)), jac=True, method="L-BFGS-B",
                      options=dict(ftol=ftol, gtol=gtol, maxiter=maxiter, maxfun=3 * maxiter))
    load(result.x)
    return result
