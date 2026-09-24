"""Stage weights and labels for digest pipeline progress aggregation."""

STAGE_WEIGHTS = {
    "BUILD_TREE": 15,
    "BIBLIOGRAPHIC": 8,
    "KEYWORDS": 8,
    "RESEARCH_DIRECTIONS": 5,
    "USAGE_SCOPE": 4,
    "HIERARCHICAL_SUMMARIZE": 25,
    "MAIN_CONTENT": 35,
}

# Stages that run concurrently. The wave is only as done as its slowest stage:
# crediting MAIN_CONTENT's full 35% while summarize is still at 30% made the
# bar read ~82% next to "88/286 nodes", which looks fabricated.
PARALLEL_STAGE_GROUPS = (
    ("BIBLIOGRAPHIC", "KEYWORDS", "RESEARCH_DIRECTIONS", "USAGE_SCOPE"),
    ("HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"),
)

# Stages whose failure must fail the whole pipeline. Anything else degrades
# to a warning in the quality report so the digest still finalizes/exports.
CRITICAL_STAGES = {"HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"}

STAGE_LABELS = {
    "BUILD_TREE": "Xây dựng cây mục lục",
    "BIBLIOGRAPHIC": "Thư mục học (§1)",
    "KEYWORDS": "Từ khóa (§2.3)",
    "RESEARCH_DIRECTIONS": "Hướng nghiên cứu",
    "USAGE_SCOPE": "Phạm vi sử dụng (§3)",
    "HIERARCHICAL_SUMMARIZE": "Tóm tắt (§2.1)",
    "MAIN_CONTENT": "Nội dung chính (§2.2)",
    "FINALIZE": "Hoàn tất",
}


def _clamp_pct(pct: int | float) -> int:
    return min(max(int(pct), 0), 100)


def aggregate_progress(completed_stages: dict[str, int]) -> int:
    """Weighted pipeline progress with parallel-group bottlenecking.

    Sequential stages contribute ``weight × stage_pct``.
    Parallel waves contribute ``sum(weights) × min(stage_pcts)`` so a finished
    sibling cannot pull the bar ahead of the stage the UI is still counting.
    """
    accounted: set[str] = set()
    total = 0.0

    for group in PARALLEL_STAGE_GROUPS:
        accounted.update(group)
        group_weight = sum(STAGE_WEIGHTS[stage] for stage in group)
        group_pct = min(_clamp_pct(completed_stages.get(stage, 0)) for stage in group)
        total += group_weight * group_pct / 100.0

    for stage, weight in STAGE_WEIGHTS.items():
        if stage in accounted:
            continue
        total += weight * _clamp_pct(completed_stages.get(stage, 0)) / 100.0

    return min(int(total), 100)
