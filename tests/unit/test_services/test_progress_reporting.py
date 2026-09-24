import pytest

from services.progress_reporting import progress_context
from services.translators._parallel import run_parallel


@pytest.mark.asyncio
async def test_parallel_worker_emits_versioned_units_without_parsing_message():
    snapshots = []

    def sink(progress, message, meta):
        snapshots.append((progress, message, meta))

    async def worker(_index, value):
        return value * 2

    with progress_context(
        sink=sink,
        defaults={
            "pipeline": "translate",
            "phase": "active",
            "mode": "block_based",
            "attempt": 2,
            "target_language": "vi",
        },
    ):
        result = await run_parallel([1, 2, 3], worker, parallelism=2, progress_label="Block")

    assert result == [2, 4, 6]
    assert [snapshot[2]["units_done"] for snapshot in snapshots] == [1, 2, 3]
    assert all(snapshot[2]["version"] == 1 for snapshot in snapshots)
    assert all(snapshot[2]["units_total"] == 3 for snapshot in snapshots)
    assert all(snapshot[2]["mode"] == "block_based" for snapshot in snapshots)
