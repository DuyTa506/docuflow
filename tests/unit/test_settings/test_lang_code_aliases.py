"""The FE used to send ``china`` / ``russia``; stored rows keep those values."""

import pytest

from config.settings import normalize_lang_code


@pytest.mark.parametrize(
    "raw, expected",
    [("china", "zh"), ("russia", "ru"), ("China", "zh"), ("zh", "zh"), ("ru", "ru")],
)
def test_legacy_fe_codes_normalize(raw, expected):
    assert normalize_lang_code(raw) == expected
