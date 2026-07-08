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


def aggregate_progress(completed_stages: dict[str, int]) -> int:
    """Weighted progress from per-stage 0-100 values."""
    total = 0
    for stage, weight in STAGE_WEIGHTS.items():
        pct = completed_stages.get(stage, 0)
        total += int(weight * min(max(pct, 0), 100) / 100)
    return min(total, 100)
