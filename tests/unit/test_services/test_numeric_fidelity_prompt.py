"""§2.2 đánh rơi số liệu có thật trong tài liệu.

Đo trên chương 4 của N4.11.160 (Уровень микроархитектуры), 7 dữ kiện kiểm chứng
được, 3 lượt mỗi cấu hình:

    qwen3.5-35B  prompt cũ    19/21
    gemma-4-26B  prompt cũ    13/21   ← "các tín hiệu điều khiển" thay cho "29"
    qwen3.5-35B  + khối này   21/21
    gemma-4-26B  + khối này   21/21

Câu "Preserve numbers, names, dates, and technical terms verbatim" đã có sẵn,
nhưng nằm lẫn giữa bốn ràng buộc khác. Nâng nó thành khối riêng, nêu ví dụ sai
cụ thể, và lặp lại sát điểm sinh thì cả hai model đều đạt tuyệt đối.

Với một bản tổng thuật thư viện, "một vài tín hiệu điều khiển" thay cho "29 tín
hiệu điều khiển" là mất thông tin, không phải rút gọn.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService

_LONG = "Содержание главы о вычислительной технике. " * 40


def _llm(payload="Tóm tắt."):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value=payload if isinstance(payload, str) else json.dumps(payload)
    )
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return llm


def _node():
    return {"title": "Глава 4", "content": "", "children": [{"content": _LONG, "children": []}]}


class TestBookPath:
    @pytest.mark.asyncio
    async def test_numeric_fidelity_block_is_present(self):
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _node(), 4)

        prompt = llm.chat_completion.await_args[0][0]
        assert "NUMERIC FIDELITY" in prompt
        assert "bit widths" in prompt

    @pytest.mark.asyncio
    async def test_it_names_the_actual_failure_not_just_a_rule(self):
        """Nêu ví dụ sai cụ thể mới có tác dụng — đo được 13/21 → 21/21."""
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _node(), 4)

        prompt = llm.chat_completion.await_args[0][0]
        assert "29 control signals" in prompt

    @pytest.mark.asyncio
    async def test_it_is_repeated_near_the_generation_point(self):
        """Cùng lý do câu ràng buộc ngôn ngữ được lặp: nói một lần ở xa thì bị bỏ qua."""
        llm = _llm()

        await MainContentService()._summarize_chapter(llm, _node(), 4)

        prompt = llm.chat_completion.await_args[0][0]
        first = prompt.find("NUMERIC FIDELITY")
        last = prompt.rfind("NUMERIC FIDELITY")
        assert first != -1 and last != first
        content_end = prompt.rfind(_LONG.strip()[:40])
        assert last > content_end, "lần nhắc thứ hai phải nằm SAU đoạn văn bản chương"


class TestProceedingsPath:
    @pytest.mark.asyncio
    async def test_bbkh_entries_get_the_same_guarantee(self):
        """Một BBKH cũng đầy số liệu như một chương sách."""
        llm = _llm({"summary": "s", "paper_count": 3})

        await MainContentService()._summarize_chapter(llm, _node(), 1, doc_kind="proceedings")

        assert "NUMERIC FIDELITY" in llm.chat_completion.await_args[0][0]
