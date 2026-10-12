from utils.digest_eval import chapter_coverage, inconsistent_terms, keyword_overlap


def test_chapter_coverage_lists_the_missing_numbers():
    assert chapter_coverage([1, 2, 4, None], 4) == {"found": 3, "expected": 4, "missing": [3]}


def test_keyword_overlap_matches_on_the_english_side():
    out = keyword_overlap(["Đạn lõm (Shaped Charges)"], [["Đạn lõm", "Shaped charge"], ["X", "Y"]])

    assert (out["matched"], out["missing"]) == (1, ["Y"])


def test_one_term_written_two_ways_is_inconsistent():
    texts = ["Nội đạn học (Interior Ballistics) và Nội lực học (Interior Ballistics)."]

    assert "Interior Ballistics" in inconsistent_terms(texts, ["Interior Ballistics"])


def test_one_term_written_one_way_is_consistent():
    texts = ["Đạn đạo trong (Interior Ballistics); phần đạn đạo trong (Interior Ballistics)."]

    assert inconsistent_terms(texts, ["Interior Ballistics"]) == {}
