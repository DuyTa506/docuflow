"""Live regression (E2E DOC_006/DOC_014): the keyword validator rejected
25/30 LLM picks as ``structural_title`` — any short phrase with a capital
letter counted as a heading, so "Buck Converter" or "MOSFET" died with
"Chapter 3" and the stage failed with 0/5 keywords."""

import pytest

from utils.keyword_validation import validate_keyword_item

SOURCE = (
    "Chapter 3 Buck Converter design. The Buck Converter uses a MOSFET switch; "
    "Runge-Kutta Method integrates the model. 2.1 Introduction to topologies."
)


@pytest.mark.parametrize("term", ["Buck Converter", "MOSFET", "Runge-Kutta Method"])
def test_capitalised_subject_terms_survive(term):
    row, reason = validate_keyword_item({"keyword": term, "weight": 0.9}, source_text=SOURCE)
    assert reason is None, reason
    assert row["keyword"] == term


@pytest.mark.parametrize("heading", ["Chapter 3", "2.1 Introduction to topologies"])
def test_explicit_headings_still_rejected(heading):
    row, reason = validate_keyword_item({"keyword": heading, "weight": 0.9}, source_text=SOURCE)
    assert row is None
    assert reason == "structural_title"
