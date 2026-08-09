"""POST /analysis must refuse documents whose OCR hasn't completed.

The upload flow only auto-triggers OCR; digest/translation are manual and
only valid once extraction is done. The UI disables those buttons, but the
API is the enforcement point — a premature call previously started a
workflow that sat in a wait-gate showing RUNNING, which read as a stuck
pipeline (transformer-evolution-vi.pdf incident).
"""

from unittest.mock import AsyncMock, MagicMock, patch


def _doc(status):
    d = MagicMock()
    d.id = "DOC_001"
    d.processing_status = status
    return d


class TestStartAnalysisExtractionPrecondition:
    def test_rejected_with_409_before_extraction(self, client):
        with patch(
            "serving.routers.analysis_router.get_authorized_document",
            return_value=_doc("EXTRACT_IN_PROGRESS"),
        ):
            resp = client.post("/api/v2/documents/DOC_001/analysis")
        assert resp.status_code == 409
        assert "OCR" in resp.json()["detail"]

    def test_starts_when_extracted(self, client):
        with (
            patch(
                "serving.routers.analysis_router.get_authorized_document",
                return_value=_doc("EXTRACTED"),
            ),
            patch(
                "serving.routers.analysis_router.start_digest_workflow",
                new=AsyncMock(return_value=("wf-1", "TASK_1")),
            ),
        ):
            resp = client.post("/api/v2/documents/DOC_001/analysis")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_1"
