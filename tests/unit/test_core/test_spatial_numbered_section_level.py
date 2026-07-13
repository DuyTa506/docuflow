"""Regression: numbered-section prefixes ("1 Introduction", "3.1 Problem
Formulation") must produce a correct, stable heading depth -- confirmed
live on a real 2-column academic paper that bbox-geometry-only scoring can
rank a subsection ABOVE its own parent section (a 0.0004 raw-score margin
gets amplified by adaptive percentile thresholds into a structurally wrong
tree), because visual box size mostly reflects heading text length, not
semantic depth."""

from core.spatial.spatial_tree_builder import (
    extract_numbered_section_level,
    validate_with_markdown_syntax,
)


class TestExtractNumberedSectionLevel:
    def test_top_level_numeric_section(self):
        assert extract_numbered_section_level("1 Introduction") == 1
        assert extract_numbered_section_level("2 Related Work") == 1
        assert extract_numbered_section_level("3 Methodology") == 1

    def test_subsection_numeric(self):
        assert extract_numbered_section_level("3.1 Problem Formulation") == 2
        assert extract_numbered_section_level("3.2 Overview") == 2

    def test_sub_subsection_numeric(self):
        assert extract_numbered_section_level("3.1.2 Some Detail") == 3

    def test_letter_dot_number_appendix_style(self):
        assert extract_numbered_section_level("A.1 Schema Selector") == 2
        assert extract_numbered_section_level("C.5 Topology Refiner Prompt") == 2

    def test_bare_single_letter_not_matched(self):
        """A bare capital letter with no dot-number is excluded -- too easy
        to false-positive on ordinary titles ("A Detailed Algorithms")."""
        assert extract_numbered_section_level("A Detailed Algorithms") is None

    def test_no_numbering_returns_none(self):
        assert extract_numbered_section_level("Abstract") is None
        assert extract_numbered_section_level("Related Work") is None

    def test_long_text_not_treated_as_heading(self):
        long_text = "1 " + ("word " * 40)
        assert extract_numbered_section_level(long_text) is None


class TestValidateWithMarkdownSyntaxNumberedPriority:
    def test_numbered_title_overrides_noisy_spatial_level(self):
        """The exact confirmed-live bug: a subsection's spatial_level ends
        up shallower (more important) than its own parent section's --
        numbered-section detection must correct both to the right depths."""
        elements = [
            {"label": "title", "text_content": "3 Methodology", "spatial_level": 2},
            {"label": "title", "text_content": "3.1 Problem Formulation", "spatial_level": 1},
        ]
        result = validate_with_markdown_syntax(elements)

        parent, child = result
        assert parent["final_level"] == 1
        assert child["final_level"] == 2
        assert parent["final_level"] < child["final_level"]
        assert parent["level_source"] == "numbered_section"
        assert child["level_source"] == "numbered_section"

    def test_non_heading_label_not_affected_by_numbered_pattern(self):
        """A body paragraph that happens to start with a number must not
        be reinterpreted as a section heading -- only title-ish labels are
        eligible for the numbered-section override. This is the actual
        guard against false positives like "1.5 million users were
        affected..." -- the low-level regex alone can't tell a decimal
        number from a subsection prefix, so the label gate here is load-
        bearing, not incidental."""
        elements = [
            {
                "label": "text",
                "text_content": "1 Introduction repeated as a citation.",
                "spatial_level": 4,
            },
            {
                "label": "text",
                "text_content": "1.5 million users were affected by the outage.",
                "spatial_level": 4,
            },
        ]
        result = validate_with_markdown_syntax(elements)
        for elem in result:
            assert elem["final_level"] == 4
            assert elem["level_source"] == "spatial_only"

    def test_falls_back_to_spatial_when_no_numbering(self):
        elements = [
            {"label": "title", "text_content": "Abstract", "spatial_level": 1},
        ]
        result = validate_with_markdown_syntax(elements)
        assert result[0]["final_level"] == 1
        assert result[0]["level_source"] == "spatial_only"

    def test_markdown_syntax_still_works_without_numbering(self):
        elements = [
            {"label": "title", "text_content": "## Subsection", "spatial_level": 1},
        ]
        result = validate_with_markdown_syntax(elements)
        assert result[0]["level_source"] in ("spatial_validated", "blended")
