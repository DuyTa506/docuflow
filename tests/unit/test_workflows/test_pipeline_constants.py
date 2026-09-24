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


def test_parallel_group_uses_bottleneck_not_independent_sum():
    """MAIN_CONTENT finishing early must not jump the bar past summarize."""
    stages = {k: 0 for k in STAGE_WEIGHTS}
    stages["BUILD_TREE"] = 100
    for name in ("BIBLIOGRAPHIC", "KEYWORDS", "RESEARCH_DIRECTIONS", "USAGE_SCOPE"):
        stages[name] = 100
    stages["HIERARCHICAL_SUMMARIZE"] = 31  # ~88/286
    stages["MAIN_CONTENT"] = 100

    # Old independent sum ≈ 15+25+7.75+35 = 82
    # Bottleneck: 15+25+60*0.31 = 58
    assert aggregate_progress(stages) == 58


def test_parallel_group_advances_with_slowest_stage():
    stages = {k: 0 for k in STAGE_WEIGHTS}
    stages["BUILD_TREE"] = 100
    for name in ("BIBLIOGRAPHIC", "KEYWORDS", "RESEARCH_DIRECTIONS", "USAGE_SCOPE"):
        stages[name] = 100
    stages["HIERARCHICAL_SUMMARIZE"] = 50
    stages["MAIN_CONTENT"] = 50
    assert aggregate_progress(stages) == 70  # 40 + 60*0.5
