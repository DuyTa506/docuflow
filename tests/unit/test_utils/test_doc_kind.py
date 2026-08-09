"""Mẫu tổng thuật có HAI chế độ, không phải một.

`Mau Tong thuat Book .docx` viết rõ ở cả §2.1 lẫn §2.2:

    "Nếu là Book, gồm các chương: ..."
    "Nếu là Kỷ yếu HNKH, gồm các BBKH, có thể tổng thuật theo Cụm vấn đề
     hoặc theo Từng BBKH đơn lẻ (nếu số lượng ít)"

Cho tới giờ chỉ có chế độ Book, nên một quyển kỷ yếu bị ép thành "Chương 1,
Chương 2…" — sai về thể loại chứ không chỉ sai về chữ.

Nhận diện ở đây cố tình **dè dặt**: chỉ đọc nhan đề và phần đầu tài liệu, và
mặc định là `book`. Một cuốn sách nhắc tới hội thảo ở trang 300 không được
phép đổi thể loại cả tài liệu — đó là lý do có cả đường ghi đè bằng tay.
"""

from unittest.mock import AsyncMock

import pytest

from utils.doc_kind import (
    BOOK,
    DOC_KINDS,
    PROCEEDINGS,
    detect_doc_kind,
    normalize_doc_kind,
    resolve_doc_kind,
    resolve_doc_kind_async,
)


class TestDetectFromTitle:
    @pytest.mark.parametrize(
        "title",
        [
            "Kỷ yếu Hội nghị khoa học trẻ 2025",
            "KỶ YẾU HỘI THẢO KHOA HỌC QUỐC GIA",
            "Proceedings of the 12th International Conference on Robotics",
            "2024 IEEE International Symposium on Circuits and Systems",
            "Сборник трудов конференции",
            "Материалы конференции молодых учёных",
            "Tuyển tập công trình khoa học",
            "第十二届全国计算机学术会议论文集",
            "電子情報通信学会 講演論文集",
            "한국통신학회 논문집",
            "Tagungsband der 30. Konferenz für Robotik",
            "Actes du colloque national d'automatique",
        ],
    )
    def test_proceedings_titles_are_recognised(self, title):
        kind, reason = detect_doc_kind(title=title)

        assert kind == PROCEEDINGS
        assert reason, "phải nói ra vì sao, nếu không thì lại là một suy đoán im lặng"

    @pytest.mark.parametrize(
        "title",
        [
            "Computer Networks, 5th Edition",
            "Архитектура компьютера",
            "Giáo trình Kỹ thuật điện tử",
            "Antenna-in-Package Technology and Applications",
        ],
    )
    def test_book_titles_stay_book(self, title):
        kind, _ = detect_doc_kind(title=title)

        assert kind == BOOK


class TestDetectFromFrontMatter:
    def test_front_matter_counts(self):
        text = "HỌC VIỆN KỸ THUẬT QUÂN SỰ\nKỷ yếu Hội nghị Khoa học lần thứ 20\nHà Nội, 2025"

        kind, _ = detect_doc_kind(title="HVKTQS 2025", text=text)

        assert kind == PROCEEDINGS

    def test_a_mention_deep_in_the_body_does_not_flip_the_kind(self):
        """Đây chính là dương tính giả cần chặn: sách nhắc tới hội thảo."""
        body = "Nội dung mở đầu. " * 2000 + "Kết quả này đã báo cáo tại kỷ yếu hội nghị ICRA."

        kind, _ = detect_doc_kind(title="Computer Networks", text=body)

        assert kind == BOOK


class TestNormalize:
    def test_known_values_pass_through(self):
        assert normalize_doc_kind("book") == BOOK
        assert normalize_doc_kind(" PROCEEDINGS ") == PROCEEDINGS

    def test_blank_means_auto(self):
        assert normalize_doc_kind("") is None
        assert normalize_doc_kind(None) is None

    def test_unknown_value_is_rejected(self):
        with pytest.raises(ValueError):
            normalize_doc_kind("magazine")

    def test_kinds_tuple_is_the_single_source_of_truth(self):
        assert DOC_KINDS == (BOOK, PROCEEDINGS)


class TestResolve:
    def test_explicit_beats_detection(self):
        resolved = resolve_doc_kind("book", title="Kỷ yếu Hội nghị khoa học")

        assert resolved["doc_kind"] == BOOK
        assert resolved["doc_kind_source"] == "explicit"

    def test_detection_fills_in_when_nothing_was_set(self):
        resolved = resolve_doc_kind(None, title="Proceedings of ICRA 2024")

        assert resolved["doc_kind"] == PROCEEDINGS
        assert resolved["doc_kind_source"] == "detected"
        assert resolved["doc_kind_reason"]

    def test_no_signal_falls_back_to_book_and_says_so(self):
        resolved = resolve_doc_kind(None, title="Архитектура компьютера")

        assert resolved["doc_kind"] == BOOK
        assert resolved["doc_kind_source"] == "default"


class TestLlmFallback:
    """Từ vựng chỉ phủ được những ngôn ngữ đã liệt kê; LLM phủ phần còn lại.

    Nhưng nó chạy **sau** từ vựng, không phải thay thế: khi bìa đã ghi rõ
    "KỶ YẾU" thì một câu trả lời sai của model không được phép lật ngược.
    """

    @pytest.mark.asyncio
    async def test_vocabulary_hit_never_reaches_the_llm(self):
        llm = AsyncMock()

        resolved = await resolve_doc_kind_async(llm, None, title="Kỷ yếu HNKH 2025")

        assert resolved["doc_kind"] == PROCEEDINGS
        assert resolved["doc_kind_source"] == "detected"
        llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_override_never_reaches_the_llm(self):
        llm = AsyncMock()

        resolved = await resolve_doc_kind_async(llm, "book", title="Bất kỳ")

        assert resolved["doc_kind_source"] == "explicit"
        llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_classifies_a_language_the_vocabulary_misses(self):
        llm = AsyncMock()
        llm.chat_completion.return_value = (
            '{"kind": "proceedings", "evidence": "bildiriler — tuyển tập báo cáo hội nghị"}'
        )

        resolved = await resolve_doc_kind_async(
            llm, None, title="Bildiriler Kitabı — Ulusal Otomasyon Sempozyumu 2025"
        )

        assert resolved["doc_kind"] == PROCEEDINGS
        assert resolved["doc_kind_source"] == "llm"
        assert "bildiriler" in resolved["doc_kind_reason"]

    @pytest.mark.asyncio
    async def test_llm_saying_book_is_recorded_as_llm_not_default(self):
        llm = AsyncMock()
        llm.chat_completion.return_value = '{"kind": "book", "evidence": "monograph"}'

        resolved = await resolve_doc_kind_async(
            llm, None, title="Ein Lehrbuch der Regelungstechnik"
        )

        assert resolved["doc_kind"] == BOOK
        assert resolved["doc_kind_source"] == "llm"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_open_to_book_and_flags_it(self):
        llm = AsyncMock()
        llm.chat_completion.side_effect = RuntimeError("LLM unreachable")

        resolved = await resolve_doc_kind_async(llm, None, title="Un titre quelconque")

        assert resolved["doc_kind"] == BOOK
        assert resolved["doc_kind_source"] == "default"
        assert resolved["doc_kind_degraded"] is True

    @pytest.mark.asyncio
    async def test_unparseable_answer_is_a_degradation_not_a_verdict(self):
        llm = AsyncMock()
        llm.chat_completion.return_value = "Tôi nghĩ đây là một quyển kỷ yếu."

        resolved = await resolve_doc_kind_async(llm, None, title="Un titre quelconque")

        assert resolved["doc_kind"] == BOOK
        assert resolved["doc_kind_degraded"] is True

    @pytest.mark.asyncio
    async def test_no_llm_client_is_the_sync_behaviour(self):
        resolved = await resolve_doc_kind_async(None, None, title="Un titre quelconque")

        assert resolved["doc_kind_source"] == "default"
        assert resolved["doc_kind_degraded"] is False
