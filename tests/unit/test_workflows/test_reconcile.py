"""Temporal-owned Task rows must be reconciled against Temporal's own state.

The startup sweep deliberately skips these types (an API restart must not kill
work living in the worker), and nothing else ever closed them — so a workflow
that died without running its finalize step left the row RUNNING forever and
the UI spinning. Observed: DIGEST_PIPELINE_326 stuck at 75% for 11 days.
"""

import pytest

from services.pipeline.reconcile import ReconcileAction, reconcile_decision


class TestReconcileDecision:
    @pytest.mark.parametrize("status", ["RUNNING", "CONTINUED_AS_NEW"])
    def test_live_workflow_left_alone(self, status):
        assert reconcile_decision(status, row_age_hours=999) is ReconcileAction.LEAVE

    def test_completed_workflow_closes_row(self):
        assert reconcile_decision("COMPLETED", row_age_hours=1) is ReconcileAction.COMPLETE

    @pytest.mark.parametrize("status", ["FAILED", "TERMINATED", "TIMED_OUT", "CANCELED"])
    def test_terminal_failure_marks_row_failed(self, status):
        assert reconcile_decision(status, row_age_hours=1) is ReconcileAction.FAIL

    def test_missing_workflow_young_row_left_alone(self):
        """NOT_FOUND right after create_parent_task is a race with
        start_workflow, not a dead run — closing it would kill a healthy job."""
        assert reconcile_decision(None, row_age_hours=0.01) is ReconcileAction.LEAVE

    def test_missing_workflow_older_than_retention_is_dead(self):
        """Past the retention TTL, NOT_FOUND is indistinguishable from
        never-existed — but a row still open that long cannot be alive."""
        assert (
            reconcile_decision(None, row_age_hours=48, retention_hours=24) is ReconcileAction.FAIL
        )

    def test_missing_workflow_inside_retention_left_alone(self):
        assert (
            reconcile_decision(None, row_age_hours=5, retention_hours=24) is ReconcileAction.LEAVE
        )

    def test_retention_boundary_needs_margin(self):
        """Exactly at the TTL the answer is still ambiguous — require clearly
        past it so clock skew can't close a live run."""
        assert (
            reconcile_decision(None, row_age_hours=24, retention_hours=24) is ReconcileAction.LEAVE
        )


class TestWorkflowIdForTask:
    def test_maps_each_temporal_owned_type(self):
        from services.pipeline.reconcile import workflow_ids_for_task

        assert workflow_ids_for_task("DIGEST_PIPELINE", "DOC_1", []) == ["digest-DOC_1"]
        assert workflow_ids_for_task("EXTRACT", "DOC_1", []) == ["extraction-DOC_1"]
        assert workflow_ids_for_task("KEYWORDS", "DOC_1", []) == ["stage-DOC_1-KEYWORDS"]

    def test_translate_covers_every_language_of_the_document(self):
        """A TRANSLATE row does not record its target language, so the check
        must consider each language the document has."""
        from services.pipeline.reconcile import workflow_ids_for_task

        ids = workflow_ids_for_task("TRANSLATE", "DOC_1", ["vi", "en"])
        assert ids == ["translation-DOC_1-vi", "translation-DOC_1-en"]

    def test_unknown_type_has_no_workflow(self):
        from services.pipeline.reconcile import workflow_ids_for_task

        assert workflow_ids_for_task("NORMALIZE", "DOC_1", []) == []


class TestAggregateStatus:
    def test_any_live_workflow_wins(self):
        """With several candidate workflows (multi-language translate), one
        still running means the row is alive."""
        from services.pipeline.reconcile import aggregate_status

        assert aggregate_status(["COMPLETED", "RUNNING"]) == "RUNNING"

    def test_completed_beats_missing(self):
        from services.pipeline.reconcile import aggregate_status

        assert aggregate_status([None, "COMPLETED"]) == "COMPLETED"

    def test_all_missing_stays_missing(self):
        from services.pipeline.reconcile import aggregate_status

        assert aggregate_status([None, None]) is None

    def test_no_candidates_is_missing(self):
        from services.pipeline.reconcile import aggregate_status

        assert aggregate_status([]) is None
