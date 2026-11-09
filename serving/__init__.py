"""
Serving package — FastAPI application and supporting services.

Public exports (lazy-loaded to avoid import cycles with services layer):
  app              — the FastAPI application instance (for uvicorn)
  startup_event    — the startup coroutine (for test harnesses)
"""


def __getattr__(name: str):
    if name == "app":
        from .workflow_api import app as _app

        return _app
    if name == "startup_event":
        from .workflow_api import startup_event as _startup

        return _startup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "startup_event"]
