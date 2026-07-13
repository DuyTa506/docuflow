"""Key conventions for Temporal translation workflow payload offload —
activities pass MinIO keys + indices, never inline unit lists (Temporal
payloads cap at ~2MB; a 700-page block list cannot ride inline).
"""

from utils.storage_keys import (
    translation_batch_key,
    translation_run_prefix,
    translation_units_key,
)


def test_units_key_shape():
    assert (
        translation_units_key("DOC_001", "TRN_002")
        == "documents/DOC_001/translations/TRN_002/units.json"
    )


def test_batch_key_shape_includes_fingerprint_and_padded_index():
    key = translation_batch_key("DOC_001", "TRN_002", "abc123", 7)
    assert key == "documents/DOC_001/translations/TRN_002/batches/abc123/0007.json"


def test_run_prefix_covers_units_and_batches():
    prefix = translation_run_prefix("DOC_001", "TRN_002")
    assert translation_units_key("DOC_001", "TRN_002").startswith(prefix)
    assert translation_batch_key("DOC_001", "TRN_002", "f", 0).startswith(prefix)
