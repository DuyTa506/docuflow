"""Unit tests for pipeline helpers."""
from services.pipeline.constants import STAGE_WEIGHTS, aggregate_progress


def test_aggregate_progress_empty():
    assert aggregate_progress({}) == 0


def test_aggregate_progress_all_complete():
    assert aggregate_progress({k: 100 for k in STAGE_WEIGHTS}) == 100


def test_aggregate_progress_partial():
    stages = {k: 0 for k in STAGE_WEIGHTS}
    stages["BUILD_TREE"] = 100
    assert aggregate_progress(stages) == STAGE_WEIGHTS["BUILD_TREE"]
