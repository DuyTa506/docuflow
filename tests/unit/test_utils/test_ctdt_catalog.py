"""§3 was empty because the catalog filter compared raw strings.

`usage_scope_service` kept only LLM answers that matched a catalog entry
byte-for-byte, against a catalog whose entries carry a mandatory "Ngành "
prefix and an en-dash (U+2013). One character off and the item vanished —
no log, no exception, an empty §3 reported as success.

The catalog is also operational data that many deployments simply do not
have, so loading must tolerate its absence instead of pretending.
"""

import json

import pytest

from config.settings import settings
from utils.ctdt_catalog import (
    CATALOG_KEYS,
    catalog_source,
    empty_catalog,
    has_entries,
    load_catalog,
    normalize_name,
    resolve_items,
    save_catalog,
    validate_catalog,
)

CATALOG = {
    "undergraduate": ["Ngành Khoa học máy tính", "Ngành Kỹ thuật điện, điện tử"],
    "master": ["Ngành Kỹ thuật ra đa – dẫn đường"],
    "phd": [],
    "strong_research_groups": ["Trí tuệ nhân tạo và khoa học dữ liệu"],
}


class TestNormalizeName:
    def test_dash_variants_collapse(self):
        """The catalog uses an en-dash; models type a plain hyphen."""
        assert normalize_name("Kỹ thuật ra đa – dẫn đường") == normalize_name(
            "Kỹ thuật ra đa - dẫn đường"
        )

    def test_spacing_around_dash_is_irrelevant(self):
        assert normalize_name("ra đa – dẫn đường") == normalize_name("ra đa-dẫn đường")

    def test_nganh_prefix_is_optional(self):
        assert normalize_name("Ngành Khoa học máy tính") == normalize_name("Khoa học máy tính")

    def test_case_and_whitespace_are_irrelevant(self):
        assert normalize_name("  KHOA  HỌC   máy tính ") == normalize_name("Khoa học máy tính")

    def test_unicode_composition_is_irrelevant(self):
        """Vietnamese text arrives both NFC and NFD depending on the source."""
        import unicodedata

        nfc = "Ngành Kỹ thuật điện"
        nfd = unicodedata.normalize("NFD", nfc)
        assert nfd != nfc
        assert normalize_name(nfd) == normalize_name(nfc)

    def test_distinct_programmes_stay_distinct(self):
        assert normalize_name("Khoa học máy tính") != normalize_name("Kỹ thuật phần mềm")


class TestResolveItems:
    def test_near_miss_resolves_to_the_catalog_string(self):
        kept, dropped = resolve_items(CATALOG, "master", ["Kỹ thuật ra đa - dẫn đường"])

        assert kept == ["Ngành Kỹ thuật ra đa – dẫn đường"], "must return the catalog spelling"
        assert dropped == []

    def test_exact_match_still_works(self):
        kept, _ = resolve_items(CATALOG, "undergraduate", ["Ngành Khoa học máy tính"])
        assert kept == ["Ngành Khoa học máy tính"]

    def test_invented_programme_is_dropped_and_reported(self):
        kept, dropped = resolve_items(CATALOG, "undergraduate", ["Ngành Y khoa"])

        assert kept == []
        assert dropped == ["Ngành Y khoa"], "dropped items must be visible, not silently gone"

    def test_duplicates_collapse_to_one(self):
        kept, _ = resolve_items(
            CATALOG, "undergraduate", ["Khoa học máy tính", "Ngành Khoa học máy tính"]
        )
        assert kept == ["Ngành Khoa học máy tính"]

    def test_unknown_key_yields_nothing(self):
        kept, dropped = resolve_items(CATALOG, "diploma", ["anything"])
        assert kept == []
        assert dropped == ["anything"]

    def test_non_string_items_are_dropped_not_crashing(self):
        kept, dropped = resolve_items(CATALOG, "undergraduate", [None, 42, {"a": 1}])
        assert kept == []
        assert len(dropped) == 3


class TestLoading:
    def test_bundled_catalog_loads_by_default(self):
        catalog = load_catalog()

        assert has_entries(catalog)
        assert all(k in catalog for k in CATALOG_KEYS)

    def test_override_file_wins_over_bundled(self, tmp_path, monkeypatch):
        override = tmp_path / "ctdt_catalog.json"
        override.write_text(
            json.dumps({"undergraduate": ["Ngành Chỉ có một"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(override))

        catalog = load_catalog()

        assert catalog["undergraduate"] == ["Ngành Chỉ có một"]
        assert catalog_source() == "uploaded"

    def test_missing_override_falls_back_to_bundled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(tmp_path / "absent.json"))

        assert has_entries(load_catalog())
        assert catalog_source() == "bundled"

    def test_unreadable_override_degrades_to_bundled(self, tmp_path, monkeypatch):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(broken))

        assert has_entries(load_catalog()), "a corrupt upload must not blank out §3"

    def test_empty_catalog_reports_no_entries(self):
        assert not has_entries(empty_catalog())
        assert not has_entries({})
        assert not has_entries({"undergraduate": [], "master": []})


class TestValidateAndSave:
    def test_validate_keeps_known_keys_only(self):
        cleaned = validate_catalog(
            {"undergraduate": ["Ngành A"], "nonsense": ["B"], "_source": "phòng đào tạo"}
        )

        assert cleaned["undergraduate"] == ["Ngành A"]
        assert "nonsense" not in cleaned
        assert cleaned["_source"] == "phòng đào tạo"
        assert all(k in cleaned for k in CATALOG_KEYS)

    def test_validate_strips_blank_and_duplicate_entries(self):
        cleaned = validate_catalog({"master": ["  Ngành A  ", "", "Ngành A", None]})
        assert cleaned["master"] == ["Ngành A"]

    def test_validate_rejects_non_dict(self):
        with pytest.raises(ValueError):
            validate_catalog(["Ngành A"])

    def test_validate_rejects_non_list_value(self):
        with pytest.raises(ValueError):
            validate_catalog({"undergraduate": "Ngành A"})

    def test_saved_catalog_is_what_load_returns(self, tmp_path, monkeypatch):
        target = tmp_path / "nested" / "ctdt_catalog.json"
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(target))

        save_catalog({"phd": ["Ngành Toán ứng dụng"]})

        assert load_catalog()["phd"] == ["Ngành Toán ứng dụng"]
        assert catalog_source() == "uploaded"

    def test_save_never_touches_the_bundled_file(self, tmp_path, monkeypatch):
        from pathlib import Path

        bundled = Path("config/ctdt_catalog.json").resolve()
        before = bundled.read_text(encoding="utf-8")
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(tmp_path / "c.json"))

        save_catalog({"phd": ["Ngành Toán ứng dụng"]})

        assert bundled.read_text(encoding="utf-8") == before
