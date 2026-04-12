"""
Run the C nanoBragg binary using configs and return float images as numpy arrays.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from nanobrag_torch.config import BeamConfig, CrystalConfig, DetectorConfig

from scripts.c_reference_utils import build_nanobragg_command
from scripts.nb_compare import find_c_binary, load_float_image


class CReferenceRunner:
    """Thin wrapper around the C binary for acceptance tests."""

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = work_dir

    def run_simulation(
        self,
        detector_config: DetectorConfig,
        crystal_config: CrystalConfig,
        beam_config: BeamConfig,
        label: str = "",
    ) -> Optional[np.ndarray]:
        """
        Run C nanoBragg and return a 2D float image, or None if the binary is missing.
        """
        try:
            c_bin = find_c_binary()
        except FileNotFoundError:
            return None

        argv = build_nanobragg_command(
            detector_config, crystal_config, beam_config
        )
        argv[0] = str(c_bin)

        if self.work_dir:
            wd = Path(self.work_dir)
            wd.mkdir(parents=True, exist_ok=True)
        else:
            wd = Path(tempfile.mkdtemp(prefix="nb_c_run_"))

        float_path = wd / "c_floatimage.bin"
        cmd = list(argv)
        if "-floatfile" in cmd:
            i = cmd.index("-floatfile")
            cmd[i + 1] = str(float_path)
        else:
            cmd += ["-floatfile", str(float_path)]

        env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        try:
            subprocess.run(
                cmd,
                cwd=str(wd),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        if not float_path.exists():
            return None

        return load_float_image(str(float_path), cmd)
