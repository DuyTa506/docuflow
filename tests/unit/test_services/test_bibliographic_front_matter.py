"""§1 must see the imprint/credits page even when a long TOC pushes it back.

E2E (DOC_014, Penetration Testing: A Survival Guide): 15 900 chars of table of
contents came first, the "## Credits / Authors" page sat past the 12 000-char
window, and §1 shipped without authors.
"""

from services.bibliographic_service import front_matter_excerpt

TOC = "Chapter heading line\n" * 800  # ~16.8k chars of table of contents
CREDITS = "## Credits\n\nAuthors\n\nWolf Halton\n\nBo Weaver\n\nReviewers\n\nPaolo Stagno\n"


def test_credits_page_past_the_window_is_appended():
    text = "Penetration Testing: A Survival Guide\n" + TOC + CREDITS + "Body " * 5000
    excerpt = front_matter_excerpt(text, max_chars=12000)
    assert "Wolf Halton" in excerpt
    assert excerpt.startswith("Penetration Testing")


def test_short_front_matter_is_unchanged():
    text = "Title\n" + CREDITS + "Body " * 100
    assert front_matter_excerpt(text, max_chars=12000) == text[:12000]


def test_no_imprint_keeps_the_plain_head():
    text = "Body text without any imprint. " * 2000
    assert front_matter_excerpt(text, max_chars=12000) == text[:12000]
