"""A paper of a volume has no title node in the tree (a journal template prints it in the
page header), so §2.2 would read «BBKH 1 - Bài báo 1». The model names each paper from
its opening text, one batched call, like the chapter-title translation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService


def _llm(payload):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value=payload if isinstance(payload, str) else json.dumps(payload)
    )
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    llm.encoding = None

    def extract(text, **kwargs):
        import re

        match = re.search(r"[\[{].*[\]}]", str(text), re.DOTALL)
        if not match:
            raise ValueError("no JSON")
        return json.loads(match.group(0))

    llm.extract_json = MagicMock(side_effect=extract)
    return llm


def _paper(n, untitled=True):
    return {
        "number": n,
        "node": {
            "title": f"Bài báo {n}",
            "untitled": untitled,
            "content": "",
            "children": [
                {"title": "ABSTRACT", "content": "", "children": []},
                {"title": f"Tiêu đề thật {n}", "content": "Tác giả A, Tác giả B", "children": []},
            ],
        },
    }


@pytest.mark.asyncio
async def test_each_untitled_paper_takes_its_title_from_the_model():
    nodes = [_paper(1), _paper(2)]
    llm = _llm([{"n": 1, "title": "Giáo dục STEM"}, {"n": 2, "title": "Đạo đức nghiên cứu"}])

    await MainContentService()._name_papers(llm, nodes)

    assert [n["node"]["title"] for n in nodes] == ["Giáo dục STEM", "Đạo đức nghiên cứu"]
    assert llm.chat_completion.await_count == 1
    assert "Tiêu đề thật 1" in llm.chat_completion.await_args.args[0]


@pytest.mark.asyncio
async def test_titled_units_are_left_alone_and_cost_no_call():
    nodes = [_paper(1, untitled=False)]
    llm = _llm([])

    await MainContentService()._name_papers(llm, nodes)

    assert nodes[0]["node"]["title"] == "Bài báo 1"
    assert llm.chat_completion.await_count == 0


@pytest.mark.asyncio
async def test_a_failed_call_keeps_the_placeholder():
    nodes = [_paper(1)]
    llm = _llm("")
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("down"))

    await MainContentService()._name_papers(llm, nodes)

    assert nodes[0]["node"]["title"] == "Bài báo 1"


@pytest.mark.asyncio
async def test_the_unit_hint_is_what_the_model_reads_when_there_is_one():
    nodes = [_paper(1)]
    nodes[0]["node"]["title_hint"] = "Tiêu đề trong gợi ý"
    llm = _llm([{"n": 1, "title": "Tiêu đề trong gợi ý"}])

    await MainContentService()._name_papers(llm, nodes)

    prompt = llm.chat_completion.await_args.args[0]
    assert "Tiêu đề trong gợi ý" in prompt
    assert "Tiêu đề thật 1" not in prompt


class TestASingleNamedPaper:
    """Live (13-paper volume): every paper-unit read «..., gồm 2 BBKH» because the prompt
    asked the model to count papers in a unit that is one paper."""

    @pytest.mark.asyncio
    async def test_no_count_is_asked_for_and_none_is_reported(self):
        node = {
            "title": "Giáo dục STEM",
            "single_paper": True,
            "member_titles": ["ĐẶT VẤN ĐỀ", "KẾT LUẬN"],
            "children": [],
        }
        llm = _llm('{"summary": "Bài báo trình bày khung giáo dục STEM.", "paper_count": 2}')

        summary, count, degraded = await MainContentService()._summarize_papers(
            llm, node, "Giáo dục STEM", "Nội dung bài báo. " * 50, ""
        )

        prompt = llm.chat_completion.await_args.args[0]
        assert "count how many" not in prompt.lower()
        assert "ONE scientific paper" in prompt
        assert summary == "Bài báo trình bày khung giáo dục STEM."
        assert count is None
        assert degraded is False
