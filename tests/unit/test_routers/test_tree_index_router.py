from unittest.mock import AsyncMock, MagicMock, patch


class TestBuildTreeIndex:
    def test_success(self, client):
        mock_doc = MagicMock()
        with (
            patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo,
            patch("services.stage_dispatch.submit_stage", new_callable=AsyncMock) as mock_submit,
        ):
            MockRepo.return_value.get.return_value = mock_doc
            mock_submit.return_value = "TASK_001"
            resp = client.post("/api/v2/documents/DOC_001/tree-index")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"

    def test_caller_options_reach_the_durable_run(self, client):
        """Building a tree for a book runs for hours, so it goes through
        Temporal — but the endpoint's per-request flags must not be dropped
        in favour of the digest defaults."""
        with (
            patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo,
            patch("services.stage_dispatch.submit_stage", new_callable=AsyncMock) as mock_submit,
        ):
            MockRepo.return_value.get.return_value = MagicMock()
            mock_submit.return_value = "TASK_001"
            client.post(
                "/api/v2/documents/DOC_001/tree-index",
                json={"if_thinning": False, "if_add_node_summary": "yes"},
            )
        options = mock_submit.await_args.kwargs["options"]
        assert options["if_thinning"] is False
        assert options["if_add_node_summary"] == "yes"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.post("/api/v2/documents/DOC_999/tree-index")
        assert resp.status_code == 404


class TestGetTreeIndex:
    def test_success(self, client):
        mock_doc = MagicMock()
        mock_tree = {"root": {"title": "Doc", "children": []}}
        with (
            patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo,
            patch("serving.tree_indexing_service.TreeIndexingService") as MockTIS,
        ):
            MockRepo.return_value.get.return_value = mock_doc
            MockTIS.return_value.get_tree_index.return_value = mock_tree
            resp = client.get("/api/v2/documents/DOC_001/tree-index")
        assert resp.status_code == 200

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/tree-index")
        assert resp.status_code == 404

    def test_no_tree_returns_404(self, client):
        mock_doc = MagicMock()
        with (
            patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo,
            patch("serving.tree_indexing_service.TreeIndexingService") as MockTIS,
        ):
            MockRepo.return_value.get.return_value = mock_doc
            MockTIS.return_value.get_tree_index.return_value = None
            resp = client.get("/api/v2/documents/DOC_001/tree-index")
        assert resp.status_code == 404
