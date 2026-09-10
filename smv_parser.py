"""
Minimal SMV helper for tests (float image arrays via fabio).
"""

from __future__ import annotations

import numpy as np


def parse_smv_image(filepath: str) -> np.ndarray:
    """Load an SMV image as a numpy array (float64)."""
    import fabio

    img = fabio.open(filepath)
    return np.asarray(img.data, dtype=np.float64)
