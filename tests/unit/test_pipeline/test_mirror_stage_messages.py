"""A finished child stage must not tell the digest parent it is "Hoàn tất".

E2E (DOC_013): §1 ended with ``_progress(100, "Hoàn tất")``; the stage sink
forwarded it verbatim, so the parent showed "20 % — Hoàn tất" while RUNNING.
"""

from unittest.mock import patch

from services.pipeline.mirror import make_stage_progress_sink, mark_stage_complete


def _sink_message(stage, progress, message):
    with patch("services.pipeline.mirror.update_pipeline_mirror") as update:
        make_stage_progress_sink("DOC_1", "DIGEST_PIPELINE_1", stage)(progress, message, {})
    return update.call_args.kwargs["message"]


def test_child_completion_message_names_the_stage():
    assert _sink_message("BIBLIOGRAPHIC", 100, "Hoàn tất") == "Đã xong: Thư mục học (§1)"


def test_child_running_message_passes_through():
    assert _sink_message("KEYWORDS", 40, "Đoạn ánh xạ từ khóa 3/13") == "Đoạn ánh xạ từ khóa 3/13"


def test_mark_stage_complete_sets_a_stage_message():
    with patch("services.pipeline.mirror.update_pipeline_mirror") as update:
        mark_stage_complete("DOC_1", "USAGE_SCOPE", "DIGEST_PIPELINE_1")
    assert update.call_args.kwargs["message"] == "Đã xong: Phạm vi sử dụng (§3)"
