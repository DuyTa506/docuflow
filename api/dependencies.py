"""
API Dependencies - Dependency injection for FastAPI.

Provides reusable dependencies for database sessions, services, etc.
"""
from typing import Generator
from openai import AsyncOpenAI

from data.database import get_db_manager
from services.ocr_service import OCRService
from config.settings import settings


def get_db() -> Generator:
    """
    Dependency for database session.
    
    Yields:
        SQLAlchemy session
    """
    db_manager = get_db_manager()
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_ocr_client() -> AsyncOpenAI:
    """
    Dependency for OCR client.
    
    Returns:
        AsyncOpenAI client configured for vLLM
    """
    return AsyncOpenAI(
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_server_url
    )


def get_ocr_service(client: AsyncOpenAI = None) -> OCRService:
    """
    Dependency for OCR service.
    
    Args:
        client: AsyncOpenAI client (optional, will create if not provided)
    
    Returns:
        OCRService instance
    """
    if client is None:
        client = get_ocr_client()
    
    return OCRService(
        client=client,
        api_key=settings.vllm_api_key,
        server_url=settings.vllm_server_url,
        model=settings.vllm_model
    )
