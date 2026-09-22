"""SMV file I/O for nanoBragg PyTorch implementation.

Handles reading SMV format files including images and masks.
"""

import math
import re
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


def parse_smv_header(filename: str) -> Dict[str, str]:
    """Parse SMV header from a file.

    Per spec AT-CLI-004 and AT-IO-001:
    - SMV files have ASCII header followed by binary data
    - Header is exactly 512 bytes
    - Contains key=value pairs separated by semicolons
    - Can be used for both -img and -mask files

    Args:
        filename: Path to SMV file

    Returns:
        Dictionary of header key-value pairs

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If header format is invalid
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"SMV file not found: {filename}")

    with open(path, "rb") as f:
        # Read header (exactly 512 bytes per SMV spec)
        header_bytes = f.read(512)
        header_str = header_bytes.decode("ascii", errors="ignore")

        # Find header content between { and }
        start_idx = header_str.find("{")
        end_idx = header_str.find("}")

        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"Invalid SMV header format in {filename}")

        header_content = header_str[start_idx+1:end_idx]

        # Parse key=value pairs
        header_dict = {}
        for line in header_content.split(";"):
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                header_dict[key.strip()] = value.strip()

    return header_dict


def read_smv_header_text(filename: str) -> str:
    """Return the raw SMV header text, the equivalent of C's ``frame.header``.

    ``value_of`` scans this text the way nanoBragg.c's ``ValueOf`` scans the
    header buffer, so the raw string (not a parsed dict) is what we need.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"SMV file not found: {filename}")
    with open(path, "rb") as f:
        return f.read(512).decode("ascii", errors="ignore")


def _c_atof(text: str) -> float:
    """C's ``atof``: parse a leading number, return 0.0 if there isn't one."""
    match = re.match(r"\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text)
    if match is None:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


def value_of(keyword: str, header_text: str) -> float:
    """Faithful port of nanoBragg.c's ``ValueOf`` (nanoBragg.c:4248-4275).

    Two behaviours here are load-bearing and are *not* what a dict lookup does:

    1. The match is a **substring** search, not a key comparison, so
       ``value_of("ORGX", ...)`` matches ``XDS_ORGX=`` and
       ``value_of("DISTANCE", ...)`` matches ``CLOSE_DISTANCE=``.
    2. It keeps advancing while the keyword still occurs, so the **last**
       occurrence wins, then reads the first ``=`` after it.

    Verified against the binary: a header carrying ``ORGX=11`` followed by
    ``XDS_ORGX=50`` yields 50.

    Returns NaN when the keyword never appears (C returns NAN and the caller
    skips the assignment), and 0.0 when it appears with no following ``=``.
    """
    idx = 0
    found = False
    while True:
        hit = header_text.find(keyword, idx)
        if hit == -1:
            break
        found = True
        idx = hit + len(keyword)
    if not found:
        return float("nan")
    eq = header_text.find("=", idx)
    if eq == -1:
        return 0.0
    return _c_atof(header_text[eq + 1:])


def smv_header_defaults(header_text: str, is_mask: bool = False) -> Dict[str, float]:
    """Extract the geometry nanoBragg.c takes from a ``-img``/``-mask`` header.

    Mirrors the mask pre-pass at nanoBragg.c:419-459 and the img pre-pass at
    nanoBragg.c:462-502. Only keys actually present in the header are returned,
    so the caller can distinguish "header said nothing" from "header said 0".

    The one deliberate difference between the two blocks in C is BEAM_CENTER_Y:
    the mask block flips it (``detsize_s - value``), the img block does not.
    """
    out: Dict[str, float] = {}

    size1 = value_of("SIZE1", header_text)
    size2 = value_of("SIZE2", header_text)
    if not math.isnan(size1):
        out["fpixels"] = int(size1)
    if not math.isnan(size2):
        out["spixels"] = int(size2)

    pixel_size_mm = value_of("PIXEL_SIZE", header_text)
    if not math.isnan(pixel_size_mm):
        out["pixel_size_mm"] = pixel_size_mm

    # C recomputes detsize from the (possibly just-updated) pixel size, so the
    # flip below must use the same value C would have had.
    detsize_s_mm = None
    if "spixels" in out:
        detsize_s_mm = out["spixels"] * out.get("pixel_size_mm", 0.1)

    distance_mm = value_of("DISTANCE", header_text)
    if not math.isnan(distance_mm):
        out["distance_mm"] = distance_mm
    close_distance_mm = value_of("CLOSE_DISTANCE", header_text)
    if not math.isnan(close_distance_mm):
        out["close_distance_mm"] = close_distance_mm

    wavelength_A = value_of("WAVELENGTH", header_text)
    if not math.isnan(wavelength_A):
        out["wavelength_A"] = wavelength_A

    beam_x_mm = value_of("BEAM_CENTER_X", header_text)
    if not math.isnan(beam_x_mm):
        out["beam_center_x_mm"] = beam_x_mm
    beam_y_mm = value_of("BEAM_CENTER_Y", header_text)
    if not math.isnan(beam_y_mm):
        if is_mask and detsize_s_mm is not None:
            out["beam_center_y_mm"] = detsize_s_mm - beam_y_mm
        else:
            out["beam_center_y_mm"] = beam_y_mm

    orgx = value_of("ORGX", header_text)
    if not math.isnan(orgx):
        out["orgx"] = orgx
    orgy = value_of("ORGY", header_text)
    if not math.isnan(orgy):
        out["orgy"] = orgy

    phi_deg = value_of("PHI", header_text)
    if not math.isnan(phi_deg):
        out["phi_start_deg"] = phi_deg
    osc_deg = value_of("OSC_RANGE", header_text)
    if not math.isnan(osc_deg):
        out["osc_range_deg"] = osc_deg

    # TWOTHETA is deliberately NOT extracted. C reads it (nanoBragg.c:450, :493)
    # into `twotheta`, which is declared at :279 as the pixel-loop scratch
    # variable for the scattering angle and is overwritten on the first pixel.
    # The detector swing is the separate `detector_twotheta` (:254), which only
    # -twotheta writes (:766). So a header TWOTHETA has no effect in C -- it
    # writes TWOTHETA=0 back out for a header that said 15 -- and honouring it
    # here would be a divergence, not a fix. Pinned by PARITY-SMVHDR-001, whose
    # fixture header carries TWOTHETA=15.

    return out


def read_smv_mask(filename: str) -> Tuple[torch.Tensor, dict]:
    """Read a mask file in SMV format.

    Per spec AT-PRE-001 and AT-ROI-001:
    - Mask files use SMV format with binary data
    - Zero values indicate pixels to skip
    - Non-zero values indicate pixels to include
    - Header contains detector parameters that may override config

    Args:
        filename: Path to SMV mask file

    Returns:
        Tuple of:
        - mask_array: Binary mask tensor (spixels, fpixels) with 0=skip, 1=include
        - header_dict: Dictionary of header parameters from the mask file

    Raises:
        FileNotFoundError: If mask file doesn't exist
        ValueError: If mask file format is invalid
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {filename}")

    with open(path, "rb") as f:
        # Read header (up to 512 bytes per SMV spec)
        header_bytes = f.read(512)
        header_str = header_bytes.decode("ascii", errors="ignore")

        # Parse header into dict
        header_dict = {}

        # Find header content between { and }
        start_idx = header_str.find("{")
        end_idx = header_str.find("}")

        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"Invalid SMV header format in {filename}")

        header_content = header_str[start_idx+1:end_idx]

        # Parse key=value pairs
        for line in header_content.split(";"):
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                header_dict[key.strip()] = value.strip()

        # Extract dimensions
        if "SIZE1" not in header_dict or "SIZE2" not in header_dict:
            raise ValueError(f"Missing SIZE1/SIZE2 in mask header: {filename}")

        fpixels = int(header_dict["SIZE1"])
        spixels = int(header_dict["SIZE2"])

        # Determine byte order
        byte_order = header_dict.get("BYTE_ORDER", "little_endian")
        endian = "<" if byte_order == "little_endian" else ">"

        # Read data based on type
        data_type = header_dict.get("TYPE", "unsigned_short")

        if data_type == "unsigned_short":
            # Read unsigned short data (2 bytes per pixel)
            num_pixels = fpixels * spixels
            pixel_data = f.read(num_pixels * 2)

            if len(pixel_data) != num_pixels * 2:
                raise ValueError(f"Insufficient data in mask file: {filename}")

            # Unpack as unsigned shorts
            fmt = f"{endian}{num_pixels}H"
            values = struct.unpack(fmt, pixel_data)

            # Reshape to (spixels, fpixels) in row-major order
            # Per spec: pixel index = slow * fpixels + fast
            mask_array = torch.tensor(values, dtype=torch.float32).reshape(spixels, fpixels)

        else:
            raise ValueError(f"Unsupported mask data type: {data_type}")

    # Convert to binary mask (0 or 1)
    mask_array = (mask_array != 0).float()

    return mask_array, header_dict


def create_circular_mask(
    spixels: int,
    fpixels: int,
    center_s: float,
    center_f: float,
    radius: float
) -> torch.Tensor:
    """Create a circular mask centered at given position.

    Utility function for testing and simple mask generation.

    Args:
        spixels: Number of slow-axis pixels
        fpixels: Number of fast-axis pixels
        center_s: Center position on slow axis (pixels)
        center_f: Center position on fast axis (pixels)
        radius: Radius of circular mask (pixels)

    Returns:
        Binary mask tensor with 1 inside circle, 0 outside
    """
    s_coords = torch.arange(spixels).view(-1, 1).float()
    f_coords = torch.arange(fpixels).view(1, -1).float()

    # Calculate distance from center
    dist_s = s_coords - center_s
    dist_f = f_coords - center_f
    dist_squared = dist_s**2 + dist_f**2

    # Create binary mask
    mask = (dist_squared <= radius**2).float()

    return mask


def create_rectangle_mask(
    spixels: int,
    fpixels: int,
    roi_xmin: int,
    roi_xmax: int,
    roi_ymin: int,
    roi_ymax: int
) -> torch.Tensor:
    """Create a rectangular ROI mask.

    Utility function to create mask from ROI bounds.

    Args:
        spixels: Number of slow-axis pixels
        fpixels: Number of fast-axis pixels
        roi_xmin: Fast axis minimum (inclusive, 0-based)
        roi_xmax: Fast axis maximum (inclusive, 0-based)
        roi_ymin: Slow axis minimum (inclusive, 0-based)
        roi_ymax: Slow axis maximum (inclusive, 0-based)

    Returns:
        Binary mask tensor with 1 inside ROI, 0 outside
    """
    mask = torch.zeros(spixels, fpixels)

    # Set ROI region to 1
    # Note: slow axis is first dimension (rows), fast axis is second (columns)
    mask[roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1] = 1.0

    return mask