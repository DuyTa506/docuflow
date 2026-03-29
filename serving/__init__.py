"""
Serving package — FastAPI application and supporting services.

Public exports:
  app              — the FastAPI application instance (for uvicorn)
  startup_event    — the startup coroutine (for test harnesses)
"""
from .workflow_api import app, startup_event

__all__ = ["app", "startup_event"]
