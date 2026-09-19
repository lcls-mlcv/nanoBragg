"""nanoBragg.c only honours -misset_seed for a random misset when it precedes -misset random."""
import warnings

import pytest

from nanobrag_torch.__main__ import warn_c_misset_seed_order


@pytest.mark.parametrize(
    "argv, warns",
    [
        (["-misset_seed", "12345", "-misset", "random"], False),
        (["-misset", "random", "-misset_seed", "12345"], True),
        (["-misset", "10", "20", "30", "-misset_seed", "1"], False),
        (["-seed", "1", "-misset", "random"], False),
    ],
)
def test_warns_when_c_would_parse_fixed_misset(argv, warns):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_c_misset_seed_order(argv)
    assert any("misset_seed" in str(w.message) for w in caught) == warns
