from unittest.mock import patch


class TestGetTask:
    def test_success(self, client):
        mock_status = {"task_id": "TASK_001", "status": "COMPLETED", "progress": 100}
        with patch("serving.routers.tasks_router.task_manager") as mock_tm:
            mock_tm.get_status.return_value = mock_status
            resp = client.get("/api/v2/tasks/TASK_001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "COMPLETED"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.tasks_router.task_manager") as mock_tm:
            mock_tm.get_status.return_value = None
            resp = client.get("/api/v2/tasks/TASK_999")
        assert resp.status_code == 404


class TestListTasks:
    def test_success(self, client):
        mock_tasks = [{"task_id": "TASK_001", "status": "RUNNING"}]
        with patch("serving.routers.tasks_router.task_manager") as mock_tm:
            mock_tm.list_tasks.return_value = mock_tasks
            resp = client.get("/api/v2/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_filter_by_document_id(self, client):
        with patch("serving.routers.tasks_router.task_manager") as mock_tm:
            mock_tm.list_tasks.return_value = []
            resp = client.get("/api/v2/tasks?document_id=DOC_001")
        assert resp.status_code == 200
        mock_tm.list_tasks.assert_called_once()
        _, kwargs = mock_tm.list_tasks.call_args
        assert kwargs.get("document_id") == "DOC_001"
