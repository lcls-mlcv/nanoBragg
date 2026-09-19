"""
Negative numeric CLI values must parse on every supported Python.

argparse only accepts a token like "-2" as a value when no option string looks like a
negative number. This CLI mirrors nanoBragg's, which includes "-4stol", and on Python 3.13+
argparse counts any option beginning with a dash and a digit as negative-number-like. That
turned every negative angle, offset and beam centre into "expected one argument" on new
interpreters, while older ones accepted them: the parity matrix passed on a 3.12 machine and
failed on a 3.14 one for the same commit.
"""
import pytest

from nanobrag_torch.__main__ import create_parser


@pytest.mark.parametrize("value", ["-2", "-2.5", "-0.75", "-1e-3", "3", "0"])
def test_negative_rotation_values_parse(value):
    args = create_parser().parse_args(["-detector_roty", value])
    assert args.detector_roty == float(value)


@pytest.mark.parametrize(
    "flag, attr",
    [("-detector_rotx", "detector_rotx"), ("-detector_rotz", "detector_rotz"),
     ("-twotheta", "twotheta"), ("-Xbeam", "Xbeam"), ("-Ybeam", "Ybeam"),
     ("-phi", "phi"), ("-osc", "osc")],
)
def test_negative_values_parse_for_every_numeric_flag(flag, attr):
    assert getattr(create_parser().parse_args([flag, "-12.5"]), attr) == -12.5


def test_negative_vector_components_parse():
    args = create_parser().parse_args(["-twotheta_axis", "0", "0", "-1"])
    assert args.twotheta_axis == [0.0, 0.0, -1.0]
    assert create_parser().parse_args(["-misset", "-10", "20", "-30"]).misset == ["-10", "20", "-30"]


def test_dash_digit_flags_still_work():
    """The fix must not stop -4stol (a real nanoBragg flag, dest 'stol') from being a flag."""
    assert create_parser().parse_args(["-4stol", "table.txt"]).stol == "table.txt"
