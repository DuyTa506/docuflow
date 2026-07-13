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
