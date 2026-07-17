"""Stratified document sampling — head-only truncation blinded the digest
stages (keywords/research/usage) to everything past ~11k tokens on large
books. The sample must cover the whole document within the same budget.
"""

from utils.doc_sampling import build_stratified_sample


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _truncate(text: str, max_tokens: int) -> str:
    return text[: max_tokens * 4]


def _chapter(title, marker, repeat=200):
    return {
        "title": title,
        "content": f"{marker} " + ("nội dung chương lặp lại " * repeat),
        "children": [],
    }


TREE = {
    "title": "Sách thử nghiệm",
    "children": [
        _chapter("Chương 1 Mở đầu", "MARK_CH1", repeat=50),
        _chapter("Chương 2 Phương pháp", "MARK_CH2", repeat=400),
        _chapter("Chương 3 Kết luận", "MARK_CH3", repeat=50),
    ],
}


class TestTreeSample:
    def test_every_chapter_represented_and_budget_respected(self):
        sample = build_stratified_sample(
            TREE,
            "flat text unused",
            token_budget=2000,
            count_tokens=_count_tokens,
            truncate=_truncate,
        )
        for marker in ("MARK_CH1", "MARK_CH2", "MARK_CH3"):
            assert marker in sample
        # outline includes chapter titles
        assert "Chương 2 Phương pháp" in sample
        assert _count_tokens(sample) <= 2000 * 1.1

    def test_allocation_is_proportional_to_chapter_mass(self):
        sample = build_stratified_sample(
            TREE,
            "",
            token_budget=2000,
            count_tokens=_count_tokens,
            truncate=_truncate,
        )
        # Chương 2 is 8× the mass of Chương 1/3 — its excerpt must be longer.
        ch2_len = sample.split("MARK_CH2")[1].split("MARK_CH3")[0] if "MARK_CH3" in sample else ""
        ch1_len = sample.split("MARK_CH1")[1].split("MARK_CH2")[0]
        assert len(ch2_len) > len(ch1_len)


class TestManyChaptersBudget:
    def test_budget_respected_with_many_chapters(self):
        """Regression (DOC_066, 106-chapter book): the per-chapter floor was
        re-applied after rescaling AND text_budget was inflated to
        floor×n_chapters (max() picked the bigger side), so the sample grew
        ~1.5× past the caller's budget — the RESEARCH_DIRECTIONS and
        USAGE_SCOPE prompts hit 16975 tokens against a 16384-token llama
        context and failed on every long book. The floor must degrade when
        chapters are many; the total must never exceed the budget."""
        tree = {
            "title": "Sách dày",
            "children": [
                _chapter(f"Chương {i}", f"MARK_{i:03d}", repeat=100) for i in range(1, 107)
            ],
        }
        budget = 3000  # well below 150 tokens × 106 chapters
        sample = build_stratified_sample(
            tree,
            "",
            token_budget=budget,
            count_tokens=_count_tokens,
            truncate=_truncate,
        )
        assert _count_tokens(sample) <= budget
        # degraded floor still keeps every chapter represented — the fix must
        # shrink excerpts, not silently drop tail chapters
        for i in (1, 53, 106):
            assert f"MARK_{i:03d}" in sample


class TestFlatFallback:
    def test_windows_cover_head_middle_tail(self):
        text = (
            "DAU_MARKER mở đầu. "
            + ("chữ đệm " * 30000)
            + " GIUA_MARKER đoạn giữa. "
            + ("chữ đệm nữa " * 30000)
            + " CUOI_MARKER kết thúc."
        )
        sample = build_stratified_sample(
            None,
            text,
            token_budget=3000,
            count_tokens=_count_tokens,
            truncate=_truncate,
        )
        assert "DAU_MARKER" in sample
        assert "CUOI_MARKER" in sample
        assert _count_tokens(sample) <= 3000 * 1.1

    def test_short_text_returned_whole(self):
        text = "Ngắn gọn, không cần lấy mẫu."
        sample = build_stratified_sample(
            None, text, token_budget=3000, count_tokens=_count_tokens, truncate=_truncate
        )
        assert sample == text
