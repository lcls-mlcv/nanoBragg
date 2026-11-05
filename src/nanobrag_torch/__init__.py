"""
nanoBragg PyTorch Implementation

A PyTorch-based diffraction simulator for nanocrystals, providing GPU acceleration
and automatic differentiation capabilities for the original nanoBragg C code.

    from nanobrag_torch import BeamConfig, CrystalConfig, DetectorConfig, simulate

The library logs to the ``nanobrag_torch`` logger and prints nothing.
"""

import importlib

__version__ = "0.1.0"

_EXPORTS = {
    ".api": ["simulate"],
    ".config": [
        "BeamConfig",
        "CrystalConfig",
        "CrystalShape",
        "DetectorConfig",
        "DetectorConvention",
        "DetectorPivot",
        "NoiseConfig",
        "SpotMetric",
    ],
    ".diffuse": ["ConstantDiffuse", "DiffuseModel", "GaussianHalo"],
    ".models.crystal": ["Crystal"],
    ".models.detector": ["Detector"],
    ".models.experiment": ["ExperimentModel", "SerialExperiment"],
    ".simulator": ["Simulator"],
    ".utils.losses": ["gaussian_nll", "poisson_nll"],
    ".utils.refinement": ["refine_lbfgsb"],
    ".compat.cctbx": [
        "experiment_from_model",
        "experiment_model_from_dxtbx",
        "reflections_from_model",
        "set_structure_factors",
        "simulator_from_dxtbx",
    ],
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = sorted(_MODULE_OF)


def __getattr__(name):
    """Import on first use, so ``import nanobrag_torch`` stays cheap."""
    if name not in _MODULE_OF:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_MODULE_OF[name], __name__), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
