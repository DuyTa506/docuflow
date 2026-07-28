"""Cổng lọc §2.2 không được phủ quyết cách đánh số của tác giả.

Chạy thật trên N4.11.160: «Глава 9. Библиография» bị gán nhãn phụ trợ và biến
thành `Chương 9. Các phần phụ trợ (Auxiliary sections).` — một tiêu đề tiếng Anh
thay cho một chương tác giả đã đánh số, trong tài liệu Nga. Lần chạy trước cùng
node đó lại được giữ nguyên, nên đây còn là nguồn dao động giữa các lần chạy.

Cổng tồn tại để loại **nhiễu không có số**: quảng cáo nhà xuất bản, mảnh mục lục,
trang bản quyền. Khi tiêu đề tự khai `Глава N` / `Приложение X` thì cấu trúc đã
được tuyên bố rồi — không còn gì để model phán đoán.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService


def _nodes(titles):
    return [
        {"node": {"title": t, "content": "nội dung " * 60, "children": []}, "number": i}
        for i, t in enumerate(titles, start=1)
    ]


def _svc(labels):
    svc = MainContentService()
    svc._classify_nodes = AsyncMock(return_value=(labels, False))
    svc._translate_titles = AsyncMock(return_value=None)

    async def fake_summarize_chapters(llm, nodes, task_id, doc_kind="book"):
        return (
            [
                {
                    "number": n["number"],
                    "title_vi": n["node"]["title"],
                    "title_original": n["node"]["title"],
                    "content": "tóm tắt",
                }
                for n in nodes
            ],
            0,
            0,
        )

    svc._summarize_chapters = fake_summarize_chapters
    return svc


class TestNumberedUnitsSurviveTheGate:
    @pytest.mark.asyncio
    async def test_a_numbered_chapter_is_never_collapsed(self):
        nodes = _nodes(["Глава 9. Библиография"])
        svc = _svc({1: "front_matter"})

        chapters, _, _, aux, _ = await svc._summarize_with_gate(MagicMock(), nodes, task_id=None)

        assert aux == 0
        assert chapters[0]["title_vi"] == "Глава 9. Библиография"

    @pytest.mark.asyncio
    async def test_a_numbered_appendix_is_never_collapsed(self):
        nodes = _nodes(["Приложение А. Двоичные числа"])
        svc = _svc({1: "toc_fragment"})

        chapters, _, _, aux, _ = await svc._summarize_with_gate(MagicMock(), nodes, task_id=None)

        assert aux == 0
        assert len(chapters) == 1

    @pytest.mark.asyncio
    async def test_unnumbered_noise_is_still_collapsed(self):
        """Không được vô hiệu hoá cổng — đây mới là việc của nó."""
        nodes = _nodes(["Издательский дом «Питер»", "Обратная связь с издательством"])
        svc = _svc({1: "front_matter", 2: "front_matter"})

        chapters, _, _, aux, _ = await svc._summarize_with_gate(MagicMock(), nodes, task_id=None)

        assert aux == 2
        assert len(chapters) == 1
        assert chapters[0]["title_vi"] == "Các mục phụ trợ"

    @pytest.mark.asyncio
    async def test_mixed_run_keeps_the_chapter_and_folds_the_rest(self):
        nodes = _nodes(["Реклама издательства", "Глава 9. Библиография", "Обратная связь"])
        svc = _svc({1: "front_matter", 2: "front_matter", 3: "front_matter"})

        chapters, _, _, aux, _ = await svc._summarize_with_gate(MagicMock(), nodes, task_id=None)

        assert aux == 2
        assert [c["title_vi"] for c in chapters] == [
            "Các mục phụ trợ",
            "Глава 9. Библиография",
            "Các mục phụ trợ",
        ]
