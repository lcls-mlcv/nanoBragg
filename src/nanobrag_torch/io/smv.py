"""SMV format writer for nanoBragg PyTorch.

This module implements the SMV (Simple Molecular Viewer) format writer
per spec-a.md AT-IO-001 requirements.
"""

import struct
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import numpy as np
import torch

from .mask import read_smv_header_block


def read_smv(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read an SMV format file.

    Args:
        filepath: Path to the SMV file

    Returns:
        Tuple of (image_data, header_dict) where:
        - image_data: numpy array with shape (slow, fast)
        - header_dict: dictionary of header key-value pairs
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        header_text = read_smv_header_block(f)

        # Parse header into dictionary
        header = {}

        # Find the start and end of header content (between { and })
        start = header_text.find('{')
        end = header_text.find('}')

        if start == -1 or end == -1:
            raise ValueError(f"Invalid SMV header in {filepath}")

        # Parse key=value pairs
        header_content = header_text[start+1:end]
        for line in header_content.split('\n'):
            line = line.strip()
            if '=' in line and line.endswith(';'):
                # Remove trailing semicolon
                line = line[:-1]
                key, value = line.split('=', 1)
                header[key.strip()] = value.strip()

        # Extract dimensions
        fpixels = int(header.get('SIZE1', 0))
        spixels = int(header.get('SIZE2', 0))

        if fpixels == 0 or spixels == 0:
            raise ValueError(f"Invalid dimensions in {filepath}: {fpixels}x{spixels}")

        # Determine byte order
        byte_order = header.get('BYTE_ORDER', 'little_endian')
        endian = '<' if 'little' in byte_order else '>'

        # Read image data
        data_type = header.get('TYPE', 'unsigned_short')
        if data_type == 'unsigned_short':
            dtype = np.uint16
            bytes_per_pixel = 2
        elif data_type == 'signed_short':
            dtype = np.int16
            bytes_per_pixel = 2
        elif data_type == 'unsigned_int':
            dtype = np.uint32
            bytes_per_pixel = 4
        elif data_type == 'signed_int':
            dtype = np.int32
            bytes_per_pixel = 4
        elif data_type == 'float':
            dtype = np.float32
            bytes_per_pixel = 4
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Read the binary data
        num_pixels = fpixels * spixels
        data_bytes = f.read(num_pixels * bytes_per_pixel)

        # Unpack based on data type and endianness
        if len(data_bytes) < num_pixels * bytes_per_pixel:
            raise ValueError(f"Insufficient data: expected {num_pixels * bytes_per_pixel} bytes, got {len(data_bytes)}")

        # Use numpy for efficient unpacking
        if endian == '<':
            data = np.frombuffer(data_bytes, dtype=np.dtype(dtype).newbyteorder('<'))
        else:
            data = np.frombuffer(data_bytes, dtype=np.dtype(dtype).newbyteorder('>'))

        # Reshape to (slow, fast) - data is stored fast-major
        image_data = np.array(data, dtype=dtype).reshape((spixels, fpixels))

    return image_data, header


def _smv_geometry_defaults(xbeam_mm, ybeam_mm, distance_mm, close_distance_mm, detsize_s_mm, pixel_size_mm) -> dict:
    """Header geometry of an unrotated MOSFLM detector: Fbeam = Ybeam and Sbeam = Xbeam."""
    return dict(
        xbeam_mm=xbeam_mm, ybeam_mm=ybeam_mm, fbeam_mm=ybeam_mm, sbeam_mm=xbeam_mm,
        distance_mm=distance_mm,
        close_distance_mm=distance_mm if close_distance_mm is None else close_distance_mm,
        detsize_s_mm=detsize_s_mm,
        orgx=ybeam_mm / pixel_size_mm + 0.5, orgy=xbeam_mm / pixel_size_mm + 0.5,
        dials_origin_mm=(-ybeam_mm, xbeam_mm, -distance_mm),
    )


def smv_geometry(detector, xbeam_mm: float, ybeam_mm: float, detsize_s_mm: float) -> dict:
    """
    Header values for write_smv, from the detector after its rotations (nanoBragg.c:1745-1770).

    Fbeam, Sbeam and ORGX are read back off the final geometry, so they differ from the
    beam centre that was asked for once the detector is rotated. Xbeam and Ybeam are as given.
    """
    pix0, fdet, sdet, odet = detector.pix0_vector, detector.fdet_vec, detector.sdet_vec, detector.odet_vec
    close = torch.dot(pix0, odet)
    distance = close / detector.r_factor
    spot = distance * detector.beam_vector - pix0
    pixel_m = detector.pixel_size
    # C's dot products with (0,0,1), (0,1,0) and (-1,0,0), whose sum sets the sign of a zero
    x, y, z = pix0.tolist()
    dials_origin_mm = (1000 * (0.0 * x + 0.0 * y + z), 1000 * (0.0 * x + y + 0.0 * z), 1000 * (-x + 0.0 * y + 0.0 * z))
    return dict(
        xbeam_mm=xbeam_mm, ybeam_mm=ybeam_mm,
        fbeam_mm=torch.dot(fdet, spot).item() * 1000, sbeam_mm=torch.dot(sdet, spot).item() * 1000,
        distance_mm=distance.item() * 1000, close_distance_mm=close.item() * 1000,
        detsize_s_mm=detsize_s_mm,
        orgx=(-torch.dot(pix0, fdet) / pixel_m + 0.5).item(), orgy=(-torch.dot(pix0, sdet) / pixel_m + 0.5).item(),
        dials_origin_mm=dials_origin_mm,
    )


def write_smv(
    filepath: Union[str, Path],
    image_data: Union[torch.Tensor, np.ndarray],
    pixel_size_mm: float,
    distance_mm: float,
    wavelength_angstrom: float,
    beam_center_x_mm: float,
    beam_center_y_mm: float,
    close_distance_mm: Optional[float] = None,
    phi_deg: Optional[float] = None,
    osc_start_deg: Optional[float] = None,
    osc_range_deg: Optional[float] = None,
    twotheta_deg: float = 0.0,
    detector_sn: str = "000",
    beamline: str = "fake",
    convention: str = "MOSFLM",
    byte_order: str = "little_endian",
    data_type: str = "unsigned_short",
    scale: float = 1.0,
    adc_offset: float = 40.0,
    geometry: Optional[Dict[str, Any]] = None,
) -> None:
    """Write image data in SMV format, with the header nanoBragg.c writes.

    The SMV format consists of:
    1. A text header with metadata, 512 bytes unless the text needs more (see below)
    2. Binary image data in fast-major (row-wise) order

    Args:
        filepath: Output file path
        image_data: Image data with shape (slow, fast) or (spixels, fpixels)
        pixel_size_mm: Pixel size in mm
        distance_mm: Sample-to-detector distance in mm
        wavelength_angstrom: X-ray wavelength in Angstroms
        beam_center_x_mm: Beam center X position in mm
        beam_center_y_mm: Beam center Y position in mm
        close_distance_mm: Minimum distance from sample to detector plane in mm
        phi_deg: Phi angle in degrees
        osc_start_deg: Oscillation start angle in degrees
        osc_range_deg: Oscillation range in degrees
        twotheta_deg: Two-theta angle in degrees
        detector_sn: Detector serial number
        beamline: Beamline name
        convention: Detector convention (affects origin calculations)
        byte_order: Byte order for binary data
        data_type: Data type for binary data
        scale: Scaling factor for float->int conversion
        adc_offset: ADC offset to add before conversion
        geometry: Header values from the rotated detector, see smv_geometry(). Without it
            they are worked out for an unrotated MOSFLM detector from the beam centre.

    Per spec AT-IO-001:
    - Header closed with "}\\f" and padded with spaces to 512 bytes, or to the next multiple
      of 512 given in HEADER_BYTES when the text is longer
    - Data is fast-major (row-wise) with pixel index = slow*fpixels + fast
    """
    filepath = Path(filepath)

    # Convert torch tensor to numpy if needed
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().cpu().numpy()

    # Ensure 2D array
    if image_data.ndim != 2:
        raise ValueError(f"Image data must be 2D, got shape {image_data.shape}")

    # Image dimensions (slow, fast)
    spixels, fpixels = image_data.shape

    g = _smv_geometry_defaults(beam_center_x_mm, beam_center_y_mm, distance_mm, close_distance_mm,
                               spixels * pixel_size_mm, pixel_size_mm)
    g.update(geometry or {})
    dials = g['dials_origin_mm']
    # The layout of nanoBragg.c:3178-3197. DIALS_ORIGIN has no semicolon there.
    header_body = (
        "DIM=2;\n"
        f"BYTE_ORDER={byte_order};\nTYPE={data_type};\n"
        f"SIZE1={fpixels};\nSIZE2={spixels};\n"
        f"PIXEL_SIZE={pixel_size_mm:g};\nDISTANCE={g['distance_mm']:g};\n"
        f"WAVELENGTH={wavelength_angstrom:g};\n"
        f"BEAM_CENTER_X={g['xbeam_mm']:g};\nBEAM_CENTER_Y={g['ybeam_mm']:g};\n"
        f"ADXV_CENTER_X={g['fbeam_mm']:g};\nADXV_CENTER_Y={g['detsize_s_mm'] - g['sbeam_mm']:g};\n"
        f"MOSFLM_CENTER_X={g['sbeam_mm'] - 0.5 * pixel_size_mm:g};\n"
        f"MOSFLM_CENTER_Y={g['fbeam_mm'] - 0.5 * pixel_size_mm:g};\n"
        f"DENZO_X_BEAM={g['sbeam_mm']:g};\nDENZO_Y_BEAM={g['fbeam_mm']:g};\n"
        f"DIALS_ORIGIN={dials[0]:g},{dials[1]:g},{dials[2]:g}\n"
        f"XDS_ORGX={g['orgx']:g};\nXDS_ORGY={g['orgy']:g};\n"
        f"CLOSE_DISTANCE={g['close_distance_mm']:g};\n"
        f"PHI={0.0 if phi_deg is None else phi_deg:g};\nOSC_START={0.0 if osc_start_deg is None else osc_start_deg:g};\n"
        f"OSC_RANGE={0.0 if osc_range_deg is None else osc_range_deg:g};\n"
        f"TWOTHETA={twotheta_deg:g};\n"
        f"DETECTOR_SN={detector_sn};\nBEAMLINE={beamline};\n"
        "}\f"
    )
    # %g can take the text past 512 bytes. C then writes it all and still declares 512, which
    # puts the pixels where no reader looks (docs/bugs/verified_c_bugs.md), so grow the block
    # to the next multiple of 512 and declare that. Up to 512 this is C's header, byte for byte.
    header_size = 512
    while True:
        header_bytes = f"{{\nHEADER_BYTES={header_size};\n{header_body}".encode("ascii")
        if len(header_bytes) <= header_size:
            break
        header_size = (len(header_bytes) + 511) // 512 * 512
    header_bytes = header_bytes.ljust(header_size, b' ')

    # Convert image data based on data type
    if data_type == "unsigned_short":
        # Apply scale and ADC offset, clip to uint16 range
        scaled_data = image_data * scale + adc_offset
        clipped_data = np.clip(scaled_data, 0, 65535)
        output_data = np.round(clipped_data).astype(np.uint16)
        dtype_str = "u2"
    elif data_type == "signed_short":
        scaled_data = image_data * scale + adc_offset
        clipped_data = np.clip(scaled_data, -32768, 32767)
        output_data = np.round(clipped_data).astype(np.int16)
        dtype_str = "i2"
    elif data_type == "unsigned_int":
        scaled_data = image_data * scale + adc_offset
        clipped_data = np.clip(scaled_data, 0, 2**32-1)
        output_data = np.round(clipped_data).astype(np.uint32)
        dtype_str = "u4"
    elif data_type == "signed_int":
        scaled_data = image_data * scale + adc_offset
        clipped_data = np.clip(scaled_data, -2**31, 2**31-1)
        output_data = np.round(clipped_data).astype(np.int32)
        dtype_str = "i4"
    elif data_type == "float":
        output_data = image_data.astype(np.float32)
        dtype_str = "f4"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    # Set endianness
    endian = "<" if byte_order == "little_endian" else ">"
    dtype_full = f"{endian}{dtype_str}"

    # Convert to fast-major (row-wise) order
    # Shape is (slow, fast), so this is already C-contiguous (row-major)
    # Ensure C-contiguous layout for correct binary output
    output_data = np.ascontiguousarray(output_data, dtype=dtype_full)

    # Write file
    with open(filepath, "wb") as f:
        f.write(header_bytes)
        output_data.tofile(f)