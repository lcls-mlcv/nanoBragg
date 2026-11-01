"""Source file I/O for nanoBragg PyTorch (AT-SRC-001).

This module implements source file reading for beam divergence and spectral dispersion.
Source files contain X,Y,Z position, weight, and wavelength per source.

Each source keeps its wavelength from the file, and -lambda fills in a missing column,
as in nanoBragg.c:2341-2343 and 2748.
"""

import math
import re
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np


# nanoBragg.c:3884-3885
_DELIMITERS = " \t,;:!"
_NUMBER_CHARACTERS = "0123456789-+.EGeg"
_N_COLUMNS = 5
_STRTOD = re.compile(
    r"\s*([-+]?(?:nan(?:\(\w*\))?|inf(?:inity)?|0x[0-9a-f]*\.?[0-9a-f]+(?:p[-+]?\d+)?|(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?))",
    re.IGNORECASE,
)


def _atof(text: str) -> float:
    """C's atof(): the leading number of the string, 0.0 when there is none."""
    match = _STRTOD.match(text)
    if match is None:
        return 0.0
    number = match.group(1)
    if "0x" in number.lower():
        return float.fromhex(number)
    return float(number.split("(")[0])


def _read_columns(line: str) -> Tuple[Optional[List[float]], bool]:
    """
    One line of a source file, as read_text_file reads it (nanoBragg.c:3931-3962).

    Returns the values, NaN where the line has none, and whether the line ended without the
    newline C's loop waits for. Each value is the leading number of its token, and a token
    that isn't one reads as 0. Returns None for a blank line. C stops reading a line at
    "\\n" alone, so it goes on past the end of an unterminated last line, a CRLF line, or a last
    token that isn't a number, and stores 0 in every column left. That wavelength of 0 makes its
    image NaN, so here those columns stay missing, and the caller says what becomes of them.
    """
    token = line.lstrip(_DELIMITERS)
    if not token.strip():
        return None, False
    values = [math.nan] * _N_COLUMNS
    column = 0
    while True:
        if not token:
            return values, True
        values[column] = _atof(token)
        token = token.lstrip(_NUMBER_CHARACTERS)
        if token == "\n":
            return values, False
        end = next((i for i, character in enumerate(token) if character in _DELIMITERS), len(token))
        token = token[end:].lstrip(_DELIMITERS)
        if token == "\n":
            return values, False
        column += 1
        if column >= _N_COLUMNS:
            return values, False


def read_sourcefile(
    filepath: Path,
    default_wavelength_m: float,
    default_source_distance_m: float = 10.0,
    beam_direction: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read source file containing X,Y,Z position, weight, and wavelength.

    Per spec section "Sources, Divergence & Dispersion" (spec-a-core.md:150-151):
    Each line contains up to 5 columns: X, Y, Z (position in meters),
    weight (dimensionless), λ (wavelength in meters), split on any of " \\t,;:!" as C does.

    Missing fields default to:
    - Each position component from -source_distance·b (source_distance default 10 m)
    - Weight = 1.0
    - λ = λ0 (default wavelength)

    Positions are normalized to unit direction vectors.

    Each source keeps its own wavelength (nanoBragg.c:2748). The weight column is
    returned as read, and like C the Simulator does not use it unless asked.

    Args:
        filepath: Path to source file
        default_wavelength_m: Default wavelength in meters (λ0)
        default_source_distance_m: Default source distance in meters (10 m)
        beam_direction: Unit beam direction vector (default [0,0,1])
        dtype: Data type for output tensors (default: None, uses torch.get_default_dtype())
        device: Device for output tensors (default: CPU)

    Returns:
        Tuple of:
        - directions: (N, 3) tensor of unit direction vectors
        - weights: (N,) tensor of weights as read
        - wavelengths: (N,) tensor of wavelengths in meters
    """
    # Resolve dtype: use caller's dtype if provided, else torch default
    if dtype is None:
        dtype = torch.get_default_dtype()

    if beam_direction is None:
        beam_direction = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    directions = []
    weights = []
    wavelengths = []

    # Default position is along -source_distance·b
    default_position = -default_source_distance_m * beam_direction.to(dtype=dtype, device=device)

    defaults = default_position.tolist() + [1.0, default_wavelength_m]

    with open(filepath, 'r', newline='') as f:
        for line in f:
            values, ran_off_the_end = _read_columns(line)
            if values is None:
                continue
            # C reads a # line like any other. Where that gives it a wavelength of 0 and a NaN
            # image, skip the line, as a comment.
            if line.lstrip(_DELIMITERS).startswith('#') and (ran_off_the_end or values[4] == 0.0):
                continue

            # NaN is C's mark for a missing value, in the file or not (nanoBragg.c:2329-2343).
            x, y, z, weight, wavelength = (
                default if math.isnan(value) else value for value, default in zip(values, defaults)
            )

            # Unit vector, and a zero vector stays zero, as in unitize() (nanoBragg.c:3993).
            position = torch.tensor([x, y, z], dtype=torch.float64, device=device)
            norm = torch.linalg.norm(position)
            direction = position / norm if norm != 0.0 else position

            directions.append(direction.to(dtype))
            weights.append(weight)
            wavelengths.append(wavelength)

    if len(directions) == 0:
        raise ValueError(f"No valid source lines found in {filepath}")

    # Convert to tensors
    directions = torch.stack(directions)
    weights = torch.tensor(weights, dtype=dtype, device=device)
    wavelengths = torch.tensor(wavelengths, dtype=dtype, device=device)

    return directions, weights, wavelengths