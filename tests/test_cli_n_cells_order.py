"""
N_cells flags follow nanoBragg.c: default 1, applied per axis in argv order.

C starts from Na = Nb = Nc = 1 and each N flag overwrites its own axis as argv is
walked, so "-N 5 -Na 4" is (4, 5, 5) but "-Na 4 -N 5" is (5, 5, 5). The PyTorch CLI
used to let -N win outright and to default the unset axes to 5, so a run with no N
flags simulated a 5x5x5 crystal where C simulated 1x1x1.
"""
import pytest

from nanobrag_torch.__main__ import create_parser, resolve_n_cells

BASE = "-default_F 100 -cell 100 100 100 90 90 90 -lambda 1 -detpixels 32"


@pytest.mark.parametrize(
    "flags, expected",
    [
        ("", (1, 1, 1)),                       # C: double Na=1.0, Nb=1.0, Nc=1.0
        ("-N 5", (5, 5, 5)),
        ("-Na 4", (4, 1, 1)),
        ("-Nb 7", (1, 7, 1)),
        ("-Na 4 -Nb 7 -Nc 9", (4, 7, 9)),
        ("-N 5 -Na 4", (4, 5, 5)),             # per-axis flag after -N wins for that axis
        ("-Na 4 -N 5", (5, 5, 5)),             # -N after it resets every axis
    ],
)
def test_n_cells_match_c_argv_order(flags, expected):
    argv = (BASE + " " + flags).split()
    args = create_parser().parse_args(argv)
    args._argv = argv
    assert resolve_n_cells(args) == expected


def test_namespace_fallback_without_argv():
    """Without recorded argv, an explicit axis still beats -N."""
    argv = (BASE + " -N 5 -Nb 7").split()
    args = create_parser().parse_args(argv)
    assert resolve_n_cells(args) == (5, 7, 5)
