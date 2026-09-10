"""
Build C nanoBragg command-line argv from Detector/Crystal/Beam configs.

Mirrors conventions used for C↔PyTorch parity (pivot vs twotheta, MOSFLM flags).
"""

from __future__ import annotations

from typing import List

from nanobrag_torch.config import (
    BeamConfig,
    CrystalConfig,
    DetectorConfig,
    DetectorConvention,
    DetectorPivot,
)

# C code treats |twotheta| <= 1e-6 as "no twotheta" for CLI generation (pivot tests).
_TWOTHETA_EPS = 1e-6

# Matches BeamConfig.fluence default in nanobrag_torch.config — omit from CLI when unchanged.
_DEFAULT_FLUENCE = 125932015286227086360700780544.0


def _fmm(v) -> str:
    """Format a float for CLI (mm, degrees) matching test expectations like '15.0'."""
    x = float(v)
    return f"{x:.1f}"


def _fmt_twotheta(v) -> str:
    """Format twotheta for CLI (large angles use one decimal; tiny values use str())."""
    x = float(v)
    if abs(x) >= 1e-3:
        return f"{x:.1f}"
    return str(x)


def _conv_flag(conv: DetectorConvention) -> List[str]:
    if conv == DetectorConvention.MOSFLM:
        return ["-mosflm"]
    if conv == DetectorConvention.XDS:
        return ["-xds"]
    if conv == DetectorConvention.DIALS:
        return ["-dials"]
    if conv == DetectorConvention.ADXV:
        return ["-adxv"]
    if conv == DetectorConvention.DENZO:
        return ["-denzo"]
    return []


def build_nanobragg_command(
    detector_config: DetectorConfig,
    crystal_config: CrystalConfig,
    beam_config: BeamConfig,
) -> List[str]:
    """Return argv list (program name first) for the C nanoBragg binary."""
    cmd: List[str] = ["nanoBragg"]

    cmd += _conv_flag(detector_config.detector_convention)

    cc = crystal_config
    cmd += [
        "-cell",
        _fmm(cc.cell_a),
        _fmm(cc.cell_b),
        _fmm(cc.cell_c),
        _fmm(cc.cell_alpha),
        _fmm(cc.cell_beta),
        _fmm(cc.cell_gamma),
    ]

    na, nb, nc = cc.N_cells
    if na == nb == nc:
        cmd += ["-N", str(int(na))]
    else:
        cmd += ["-Na", str(int(na)), "-Nb", str(int(nb)), "-Nc", str(int(nc))]

    cmd += ["-default_F", _fmm(cc.default_F)]

    if cc.misset_random:
        cmd += ["-misset", "random"]
        if cc.misset_seed is not None:
            cmd += ["-misset_seed", str(int(cc.misset_seed))]
    else:
        mx, my, mz = cc.misset_deg
        if abs(mx) + abs(my) + abs(mz) > 0.0:
            cmd += [
                "-misset",
                _fmm(mx),
                _fmm(my),
                _fmm(mz),
            ]

    if cc.phi_start_deg != 0.0 or cc.osc_range_deg != 0.0:
        cmd += ["-phi", _fmm(cc.phi_start_deg), "-osc", _fmm(cc.osc_range_deg)]

    if cc.mosaic_spread_deg != 0.0:
        cmd += ["-mosaic", _fmm(cc.mosaic_spread_deg)]
    if cc.mosaic_domains != 1:
        cmd += ["-mosaic_dom", str(int(cc.mosaic_domains))]
    if cc.mosaic_seed is not None:
        cmd += ["-mosaic_seed", str(int(cc.mosaic_seed))]

    bc = beam_config
    cmd += ["-lambda", _fmm(bc.wavelength_A)]

    if bc.nopolar:
        cmd.append("-nopolar")
    elif float(bc.polarization_factor) != 0.0:
        cmd += ["-polar", _fmm(bc.polarization_factor)]

    flu = float(bc.fluence)
    if abs(flu - _DEFAULT_FLUENCE) > 1e-6 * max(abs(_DEFAULT_FLUENCE), 1.0):
        cmd += ["-fluence", str(flu)]

    dc = detector_config
    cmd += [
        "-distance",
        _fmm(dc.distance_mm),
        "-pixel",
        _fmm(dc.pixel_size_mm),
    ]

    if int(dc.spixels) == int(dc.fpixels):
        cmd += ["-detpixels", str(int(dc.spixels))]
    else:
        cmd += [
            "-detpixels_f",
            str(int(dc.fpixels)),
            "-detpixels_s",
            str(int(dc.spixels)),
        ]

    if dc.beam_center_f is not None and dc.beam_center_s is not None:
        cmd += [
            "-Xbeam",
            _fmm(dc.beam_center_f),
            "-Ybeam",
            _fmm(dc.beam_center_s),
        ]

    if float(dc.detector_rotx_deg) != 0.0:
        cmd += ["-detector_rotx", _fmm(dc.detector_rotx_deg)]
    if float(dc.detector_roty_deg) != 0.0:
        cmd += ["-detector_roty", _fmm(dc.detector_roty_deg)]
    if float(dc.detector_rotz_deg) != 0.0:
        cmd += ["-detector_rotz", _fmm(dc.detector_rotz_deg)]

    tt = float(dc.detector_twotheta_deg)
    if abs(tt) > _TWOTHETA_EPS:
        cmd += ["-pivot", "sample", "-twotheta", _fmt_twotheta(tt)]
    else:
        pivot = dc.detector_pivot or DetectorPivot.BEAM
        if pivot == DetectorPivot.SAMPLE:
            cmd += ["-pivot", "sample"]
        else:
            cmd += ["-pivot", "beam"]

    if int(dc.oversample) > 0:
        cmd += ["-oversample", str(int(dc.oversample))]

    return cmd


def run_c_nanobrag(cmd: List[str], **kwargs):
    """Placeholder hook for tests that patch or skip C execution."""
    raise RuntimeError("run_c_nanobrag must be monkeypatched or implemented for your site")
