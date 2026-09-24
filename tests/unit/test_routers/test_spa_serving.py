"""The API serves the built frontend, so both live on one origin.

The frontend used to be served separately on :4200 and reach the API through
an absolute `apiUrl` baked into `assets/env.json`. That URL can only ever be
right for one caller: `http://localhost:8022/api/v2/` works for an SSH
port-forward and fails for every colleague on the LAN, whose browser resolves
`localhost` to their own machine — which is exactly the ten blank red toasts
reported on 2026-08-12.

Serving both from one origin lets `apiUrl` become the relative `/api/v2/`,
which resolves against whatever host the browser actually used. One build then
works through a LAN IP, a port-forward, and a future hostname or reverse proxy
with no per-machine configuration.

The catch-all that makes client-side routing work is the risk here: it must
never swallow an API path (a mistyped endpoint has to stay a JSON 404, not
become an HTML page) and must never serve a file from outside the build.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    """A minimal Angular build: index, a hashed bundle, an asset."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<html><base href='/'>SPA</html>")
    (root / "main.abc123.js").write_text("console.log(1)")
    (root / "assets" / "env.json").write_text('{"apiUrl": "/api/v2/"}')
    (tmp_path / "secret.txt").write_text("NOT PART OF THE BUILD")
    return root


@pytest.fixture
def client(dist: Path) -> TestClient:
    from serving.spa import mount_spa

    app = FastAPI()

    @app.get("/api/v2/ping")
    async def ping():
        return {"ok": True}

    mount_spa(app, dist)
    return TestClient(app)


class TestTheAppIsServed:
    def test_root_returns_the_spa(self, client):
        resp = client.get("/")

        assert resp.status_code == 200
        assert "SPA" in resp.text
        assert resp.headers["content-type"].startswith("text/html")

    def test_a_client_route_returns_the_spa_not_404(self, client):
        """`/document` exists only in the Angular router. A hard refresh on it
        has to reach index.html or the app 404s on its own pages."""
        resp = client.get("/document")

        assert resp.status_code == 200
        assert "SPA" in resp.text

    def test_hashed_bundles_are_served_from_the_build_root(self, client):
        resp = client.get("/main.abc123.js")

        assert resp.status_code == 200
        assert resp.text == "console.log(1)"

    def test_assets_are_served(self, client):
        resp = client.get("/assets/env.json")

        assert resp.status_code == 200
        assert resp.json()["apiUrl"] == "/api/v2/"


class TestTheApiStillWins:
    def test_an_api_route_is_not_shadowed(self, client):
        resp = client.get("/api/v2/ping")

        assert resp.json() == {"ok": True}

    def test_an_unknown_api_path_stays_a_json_404(self, client):
        """The whole point of the guard: a typo in an endpoint must not come
        back as 200 text/html, which is unreadable to any HTTP client and
        turns a clear 404 into a mystery."""
        resp = client.get("/api/v2/does-not-exist")

        assert resp.status_code == 404
        assert not resp.headers["content-type"].startswith("text/html")

    @pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
    def test_the_api_docs_still_work(self, dist, path):
        from serving.spa import mount_spa

        app = FastAPI()
        mount_spa(app, dist)

        assert TestClient(app).get(path).status_code == 200


class TestItCannotServeWhatIsNotInTheBuild:
    @pytest.mark.parametrize(
        "path",
        ["/../secret.txt", "/..%2fsecret.txt", "/assets/../../secret.txt"],
    )
    def test_traversal_never_escapes_the_build_directory(self, client, path):
        resp = client.get(path)

        assert "NOT PART OF THE BUILD" not in resp.text


class TestWithoutABuild:
    def test_a_missing_dist_leaves_the_api_working(self, tmp_path):
        """Developers run the API without building the frontend; that must
        start, not crash, and must not pretend to serve a page."""
        from serving.spa import mount_spa

        app = FastAPI()

        @app.get("/api/v2/ping")
        async def ping():
            return {"ok": True}

        mount_spa(app, tmp_path / "nope")
        client = TestClient(app)

        assert client.get("/api/v2/ping").json() == {"ok": True}
        assert client.get("/").status_code == 404
