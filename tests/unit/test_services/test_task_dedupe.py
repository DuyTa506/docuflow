"""Tests for in-flight task deduplication."""

from unittest.mock import MagicMock, patch

import pytest


class TestTaskDedupe:
    def test_find_active_returns_pending_task(self):
        from data.repositories.task_repo import TaskRepository

        db = MagicMock()
        task = MagicMock()
        task.status = "PENDING"
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = task

        found = TaskRepository(db).find_active("DOC_001", "TRANSLATE")
        assert found is task

    def test_submit_returns_existing_when_dedupe(self):
        from services.task_manager import TaskManager

        tm = TaskManager()
        db = MagicMock()
        with patch.object(tm, "get_active_task_id", return_value="TRANSLATE_007"):
            task_id = tm.submit(db, "DOC_001", "TRANSLATE", _dummy_coro(), dedupe=True)
        assert task_id == "TRANSLATE_007"

    def test_submit_uses_coro_factory_with_task_id(self):
        from services.task_manager import TaskManager

        tm = TaskManager()
        db = MagicMock()
        captured: list[str] = []

        def _factory(tid: str):
            captured.append(tid)
            return _dummy_coro()

        with (
            patch.object(tm, "get_active_task_id", return_value=None),
            patch("services.task_manager.IdGenerator.next_id", return_value="TASK_099"),
            patch("services.task_manager.asyncio.create_task") as mock_create,
        ):
            task_id = tm.submit(
                db,
                "DOC_001",
                "TRANSLATE",
                coro_factory=_factory,
            )
        assert task_id == "TRANSLATE_099"
        assert captured == ["TRANSLATE_099"]
        mock_create.assert_called_once()


async def _dummy_coro():
    return {"ok": True}
