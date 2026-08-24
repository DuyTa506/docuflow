"""Settings hygiene: no duplicated field definitions, and the Temporal worker
activity-concurrency knob is its own setting (max_concurrent_pipelines was
mistakenly used as max_concurrent_activities, throttling every activity in the
worker to 2 and preventing the digest DAG's 4-way parallel stage group from
ever running in parallel).
"""

import re
from pathlib import Path

from config.settings import settings

SETTINGS_SRC = Path("config/settings.py").read_text()


def test_no_duplicate_field_definitions():
    field_names = re.findall(r"^    (\w+):.*=\s*Field\(", SETTINGS_SRC, re.MULTILINE)
    dupes = {name for name in field_names if field_names.count(name) > 1}
    assert not dupes, f"Duplicated Settings fields: {sorted(dupes)}"


def test_temporal_max_concurrent_activities_setting():
    assert settings.temporal_max_concurrent_activities == 8


def test_worker_uses_temporal_max_concurrent_activities():
    from workers.temporal_worker import _worker_config

    cfg = _worker_config()
    assert cfg["max_concurrent_activities"] == settings.temporal_max_concurrent_activities


def test_extraction_worker_uses_extraction_max_activities():
    """Activity fan-out is independent of the OPEN-workflow soft ceiling."""
    from workers.temporal_worker import _extraction_worker_config

    cfg = _extraction_worker_config()
    assert cfg["max_concurrent_activities"] == settings.extraction_max_activities
    assert settings.extraction_max_activities == 4
    assert settings.extraction_max_concurrent == 8
    assert settings.docling_slots == 4
    assert settings.docling_do_formula_enrichment is False
    assert settings.deepseek_formula_enrichment is True
