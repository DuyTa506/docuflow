from utils.file_download import _parse_range


def test_parse_range_start_end():
    assert _parse_range("bytes=0-99", 1000) == (0, 99)


def test_parse_range_open_end():
    assert _parse_range("bytes=100-", 1000) == (100, 999)


def test_parse_range_suffix():
    assert _parse_range("bytes=-50", 1000) == (950, 999)


def test_parse_range_rejects_past_end():
    assert _parse_range("bytes=1000-1001", 1000) is None


def test_parse_range_missing():
    assert _parse_range(None, 1000) is None
