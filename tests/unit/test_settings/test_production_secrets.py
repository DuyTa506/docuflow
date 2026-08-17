import pytest

from config.settings import Settings


def test_prod_refuses_default_jwt(monkeypatch):
    monkeypatch.setenv("DOCUFLOW_PROD", "1")
    with pytest.raises(ValueError, match="PRODUCTION STARTUP REFUSED"):
        Settings(
            jwt_secret_key="change-me-in-production",
            minio_access_key="ok",
            minio_secret_key="ok",
            database_url="postgresql+psycopg2://u:p@localhost:5433/db",
            admin_password="not-admin",
        )


def test_prod_refuses_default_minio(monkeypatch):
    monkeypatch.setenv("DOCUFLOW_PROD", "1")
    with pytest.raises(ValueError, match="MINIO"):
        Settings(
            jwt_secret_key="a-real-secret-value",
            minio_access_key="minioadmin",
            minio_secret_key="minioadmin",
            database_url="postgresql+psycopg2://u:p@localhost:5433/db",
            admin_password="not-admin",
        )


def test_dev_allows_defaults(monkeypatch):
    monkeypatch.setenv("DOCUFLOW_PROD", "0")
    Settings(
        jwt_secret_key="change-me-in-production",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        database_url="postgresql+psycopg2://docuflow:docuflow@localhost:5433/docuflow",
        admin_password="admin",
    )
