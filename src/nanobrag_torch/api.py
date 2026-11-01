"""One call from configs to an image."""

from typing import Optional

import torch

from .config import BeamConfig, CrystalConfig, DetectorConfig
from .models.crystal import Crystal
from .models.detector import Detector
from .simulator import Simulator


def simulate(
    crystal: CrystalConfig,
    detector: DetectorConfig,
    beam: Optional[BeamConfig] = None,
    *,
    F=None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Simulate one diffraction image in photons per pixel, shape (spixels, fpixels).

    Args:
        crystal, detector, beam: The experiment. ``beam`` defaults to ``BeamConfig()``.
        F: Structure factors. A path to an h k l F text file, a P1
            ``(indices, amplitudes)`` pair, or None for ``crystal.default_F`` everywhere.
        device: Where every tensor lives, CPU if None.
        dtype: Floating point type of every tensor.
    """
    from .compat.cctbx import set_structure_factors

    model = Crystal(crystal, device=device, dtype=dtype)
    if isinstance(F, tuple):
        set_structure_factors(model, *F)
    elif F is not None:
        model.load_hkl(F)
    simulator = Simulator(
        model,
        Detector(detector, device=device, dtype=dtype),
        crystal_config=crystal,
        beam_config=beam or BeamConfig(),
        device=device,
        dtype=dtype,
    )
    return simulator.run()
