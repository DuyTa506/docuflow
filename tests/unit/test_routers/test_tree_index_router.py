from unittest.mock import MagicMock, patch


class TestBuildTreeIndex:
    def test_success(self, client):
        mock_doc = MagicMock()
        with (
            patch("serving.routers.tree_index_router.DocumentRepository") as MockRepo,
            patch("serving.routers.tree_index_router.task_manager") as mock_tm,
        ):
            MockRepo.return_value.get.return_value = mock_doc
            mock_tm.submit.return_value = "TASK_001"
            resp = client.post("/api/v2/documents/DOC_001/tree-index")
        # Close any unawaited coroutine passed to submit
        call_kwargs = mock_tm.submit.call_args
        if call_kwargs and "coro" in call_kwargs.kwargs:
            call_kwargs.kwargs["coro"].close()
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"

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
