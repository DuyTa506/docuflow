"""MinIO / S3-compatible object storage wrapper."""

from __future__ import annotations

import mimetypes
import os
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from minio import Minio
from minio.error import S3Error

from config.settings import settings


class ObjectStorage:
    """Thin wrapper around the MinIO client."""

    def __init__(self) -> None:
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        self._client.put_object(
            self.bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def put_file(self, key: str, file_path: str, *, content_type: str | None = None) -> str:
        ct = content_type or mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        self._client.fput_object(self.bucket, key, file_path, content_type=ct)
        return key

    def exists(self, key: str) -> bool:
        if not key:
            return False
        try:
            self._client.stat_object(self.bucket, key)
            return True
        except S3Error:
            return False

    def get_bytes(self, key: str) -> bytes:
        response = self._client.get_object(self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_stream(self, key: str) -> BinaryIO:
        return self._client.get_object(self.bucket, key)

    def iter_stream(self, key: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        response = self.get_stream(key)
        try:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            response.close()
            response.release_conn()

    def delete(self, key: str) -> None:
        if not key:
            return
        try:
            self._client.remove_object(self.bucket, key)
        except S3Error:
            pass

    def list_keys(self, prefix: str) -> list[str]:
        if not prefix:
            return []
        return [
            obj.object_name
            for obj in self._client.list_objects(self.bucket, prefix=prefix, recursive=True)
        ]

    def delete_prefix(self, prefix: str) -> None:
        if not prefix:
            return
        for obj in self._client.list_objects(self.bucket, prefix=prefix, recursive=True):
            self._client.remove_object(self.bucket, obj.object_name)

    def materialize_to_temp(self, key: str, *, suffix: str | None = None) -> str:
        """Download object to a temp file; caller must delete when done."""
        data = self.get_bytes(key)
        ext = suffix if suffix is not None else Path(key).suffix
        fd, path = tempfile.mkstemp(suffix=ext or "")
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        return path

    def is_object_key(self, path_or_key: str | None) -> bool:
        """True when value is a MinIO object key (not a legacy local path)."""
        if not path_or_key:
            return False
        return path_or_key.startswith("documents/") and not os.path.isfile(path_or_key)

    def resolve_local_or_key(self, path_or_key: str) -> str:
        """Return a local filesystem path for processing (materialize if needed)."""
        if self.is_object_key(path_or_key):
            return self.materialize_to_temp(path_or_key)
        if not path_or_key or not os.path.isfile(path_or_key):
            raise FileNotFoundError(f"File not found: {path_or_key}")
        return path_or_key


@lru_cache
def get_object_storage() -> ObjectStorage:
    return ObjectStorage()
