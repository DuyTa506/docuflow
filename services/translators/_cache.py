"""Persistent per-run translation unit cache (MinIO-backed).

Every translation mode funnels through StructuredTranslator's
translate_text/translate_title, so one content-addressed cache under the
translation run's storage prefix gives crash/retry resume to all of them:
on a re-run, already-translated units are cache hits and only the missing
tail is re-translated. Keys are hashed over (kind, target_lang, text) —
a changed source text or target language never reuses a stale entry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from utils.storage_keys import translation_run_prefix

logger = logging.getLogger(__name__)


class TranslationUnitCache:
    def __init__(self, document_id: str, translation_id: str, *, target_lang: str, storage=None):
        if storage is None:
            from services.object_storage import get_object_storage

            storage = get_object_storage()
        self._storage = storage
        self._target_lang = target_lang
        self.prefix = translation_run_prefix(document_id, translation_id) + "cache/"
        self._mem: dict[str, str] = {}

    def _key(self, kind: str, text: str) -> str:
        digest = hashlib.sha1(
            f"{kind}\x00{self._target_lang}\x00{text}".encode("utf-8")
        ).hexdigest()
        return f"{self.prefix}{digest}.json"

    def load(self) -> int:
        """Pull all persisted entries into memory (one list + N gets at run
        start beats a storage round-trip per unit). Returns entry count."""
        try:
            for key in self._storage.list_keys(self.prefix):
                try:
                    payload = json.loads(self._storage.get_bytes(key).decode("utf-8"))
                    self._mem[key] = payload["translated"]
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Translation cache load failed (starting cold): %s", exc)
        return len(self._mem)

    def get(self, kind: str, text: str) -> Optional[str]:
        return self._mem.get(self._key(kind, text))

    def put(self, kind: str, text: str, translated: str) -> None:
        key = self._key(kind, text)
        self._mem[key] = translated
        try:
            self._storage.put_bytes(
                key, json.dumps({"translated": translated}, ensure_ascii=False).encode("utf-8")
            )
        except Exception as exc:
            # Persistence is a checkpoint, not correctness — never fail a
            # translation because MinIO hiccuped.
            logger.warning("Translation cache write failed (non-fatal): %s", exc)
