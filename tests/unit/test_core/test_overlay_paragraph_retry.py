"""Overlay paragraph translation must retry a BOUNDED number of times, then
degrade that paragraph to its source text instead of retrying forever (the old
@retry(wait_fixed(1)) had no stop condition — one persistently failing
paragraph hung the whole overlay run) or killing the document.
"""

from core.pdf_overlay.converter import translate_paragraphs


class _AlwaysFails:
    def __init__(self):
        self.calls = 0

    def translate(self, s: str) -> str:
        self.calls += 1
        raise RuntimeError("backend down")


class _Upper:
    def translate(self, s: str) -> str:
        return s.upper()


def test_failing_paragraph_returns_source_after_bounded_retries():
    adapter = _AlwaysFails()
    news, degraded = translate_paragraphs(
        ["Hello world"], skip_translate=set(), translator=adapter, thread=1
    )
    assert news == ["Hello world"]
    assert degraded == 1
    assert adapter.calls == 3


def test_one_bad_paragraph_does_not_poison_the_rest():
    class _FailsOnMarker:
        def translate(self, s: str) -> str:
            if "BAD" in s:
                raise RuntimeError("boom")
            return s.upper()

    news, degraded = translate_paragraphs(
        ["good one", "BAD one", "another good"],
        skip_translate=set(),
        translator=_FailsOnMarker(),
        thread=2,
    )
    assert news == ["GOOD ONE", "BAD one", "ANOTHER GOOD"]
    assert degraded == 1


def test_skip_and_formula_paragraphs_pass_through_untranslated():
    news, degraded = translate_paragraphs(
        ["{v0}", "  ", "keep me", "translate me"],
        skip_translate={2},
        translator=_Upper(),
        thread=1,
    )
    assert news == ["{v0}", "  ", "keep me", "TRANSLATE ME"]
    assert degraded == 0
