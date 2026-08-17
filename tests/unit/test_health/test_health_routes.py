from fastapi import FastAPI
from fastapi.testclient import TestClient

from serving.health import router


def test_live_returns_ok():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
