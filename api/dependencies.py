"""
API Dependencies — Dependency injection for FastAPI.

Provides reusable dependencies for:
- Database sessions
- OCR service
- JWT authentication
- Role-based access control
- LLM client factory
"""
from typing import Generator, Callable, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from data.database import get_db_manager
from data.db_models import User
from services.ocr_service import OCRService
from config.settings import settings

# OAuth2 scheme — token URL matches the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v2/auth/login", auto_error=False)


# ── Database dependency ─────────────────────────────────────────────

def get_db() -> Generator:
    """Dependency for database session."""
    db_manager = get_db_manager()
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── OCR dependencies (unchanged) ────────────────────────────────────

def get_ocr_client() -> AsyncOpenAI:
    """Dependency for OCR client."""
    return AsyncOpenAI(
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_server_url,
    )


def get_ocr_service(client: AsyncOpenAI = None) -> OCRService:
    """Dependency for OCR service."""
    if client is None:
        client = get_ocr_client()
    return OCRService(
        client=client,
        api_key=settings.vllm_api_key,
        server_url=settings.vllm_server_url,
        model=settings.vllm_model,
    )


# ── Authentication dependencies ─────────────────────────────────────

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Decode JWT and return the authenticated User.

    Raises 401 if token is missing / invalid / user not found.
    """
    from services.auth_service import AuthService

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth = AuthService()
    payload = auth.decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user.status != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status}",
        )

    return user


def require_role(*roles: str) -> Callable:
    """
    Returns a dependency that ensures the current user has one of the
    specified roles.

    Usage::

        @router.get("/admin-only", dependencies=[Depends(require_role("ADMIN"))])
        async def admin_endpoint(): ...
    """
    async def _check_role(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' not authorized. Required: {', '.join(roles)}",
            )
        return user

    return _check_role


# ── LLM client dependency ───────────────────────────────────────────

_llm_cache: dict = {}


def get_llm_client():
    """
    Return a cached LLM client for AI services (translation, summarisation, etc.).

    The client is cached by (provider, model, ollama_base_url) so repeated calls
    within the same process never re-initialise the underlying HTTP session.
    """
    from pageindex.llm.client_factory import LLMClientFactory

    key = (settings.ai_provider, settings.ai_model, settings.ai_ollama_base_url, settings.ai_openai_base_url)
    if key not in _llm_cache:
        _llm_cache[key] = LLMClientFactory.create_client(
            provider=settings.ai_provider,
            model=settings.ai_model,
            ollama_base_url=settings.ai_ollama_base_url,
            openai_base_url=settings.ai_openai_base_url,
        )
    return _llm_cache[key]
