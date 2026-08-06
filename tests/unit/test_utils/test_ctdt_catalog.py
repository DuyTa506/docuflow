"""The §3 catalog is now the BGD one, and the discipline code is what matches.

It used to be the Academy's 42 programme names, matched on the normalised
string. Switching to Phụ lục I of Thông tư 09/2022 breaks that structurally: in
the national catalog 364 of 604 discipline names sit under more than one code,
and the word "Khác" alone sits under 73. A name no longer identifies a
discipline; only a code does.

So `resolve_items` takes discipline codes. It still accepts a name as a fallback
— the model does not always comply — but only when that name is unique within
the level being resolved; ambiguous names are dropped and reported, because
guessing at a code is worse than leaving the slot empty.

The catalog is still operational data that many deployments will not have, so
loading has to tolerate a missing file rather than pretend.
"""

import json

import pytest

from config.settings import settings
from utils.ctdt_catalog import (
    CATALOG_KEYS,
    catalog_source,
    catalog_text_block,
    count_programmes,
    empty_catalog,
    has_entries,
    iter_programmes,
    load_catalog,
    name_key,
    normalize_name,
    research_area_names,
    resolve_items,
    save_catalog,
    validate_catalog,
)

CATALOG = {
    "undergraduate": [
        {
            "code": "74801",
            "name": "Máy tính",
            "children": [
                {"code": "7480101", "name": "Khoa học máy tính"},
                {"code": "7480102", "name": "Mạng máy tính và truyền thông dữ liệu"},
            ],
        },
        {
            "code": "78601",
            "name": "An ninh và trật tự xã hội",
            "children": [{"code": "7860103", "name": "Trinh sát kỹ thuật"}],
        },
        {
            "code": "78602",
            "name": "Quân sự",
            # Cùng tên với 7860103 — chính là ca nhập nhằng mà mã phải xử lý.
            "children": [{"code": "7860231", "name": "Trinh sát kỹ thuật"}],
        },
    ],
    "master": [
        {
            "code": "85202",
            "name": "Kỹ thuật điện, điện tử và viễn thông",
            "children": [{"code": "8520204", "name": "Kỹ thuật rađa - dẫn đường"}],
        }
    ],
    "phd": [],
}


class TestNormalizeName:
    """Still used for the name fallback, and by research_direction_service."""

    def test_dash_variants_collapse(self):
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
        import unicodedata

        nfc = "Ngành Kỹ thuật điện"
        nfd = unicodedata.normalize("NFD", nfc)
        assert nfd != nfc
        assert normalize_name(nfd) == normalize_name(nfc)

    def test_distinct_programmes_stay_distinct(self):
        assert normalize_name("Khoa học máy tính") != normalize_name("Kỹ thuật phần mềm")

    def test_name_key_is_word_order_insensitive(self):
        assert name_key("Giao thức PPP và HDLC") == name_key("Giao thức HDLC và PPP")


class TestIterProgrammes:
    def test_walks_leaves_in_order(self):
        assert list(iter_programmes(CATALOG, "undergraduate"))[:2] == [
            ("7480101", "Khoa học máy tính"),
            ("7480102", "Mạng máy tính và truyền thông dữ liệu"),
        ]

    def test_counts_leaves_not_groups(self):
        assert count_programmes(CATALOG, "undergraduate") == 4
        assert count_programmes(CATALOG, "phd") == 0
        assert count_programmes(CATALOG, "diploma") == 0


class TestResearchAreaNames:
    """Groups replace the 18 hand-written names the catalog itself called unofficial."""

    def test_group_names_deduplicated_across_levels(self):
        names = research_area_names(
            {"undergraduate": CATALOG["undergraduate"], "master": CATALOG["undergraduate"]}
        )

        assert names == ["Máy tính", "An ninh và trật tự xã hội", "Quân sự"]

    def test_group_without_programmes_is_not_an_area(self):
        assert research_area_names({"phd": [{"code": "74801", "name": "Máy tính"}]}) == []

    def test_bundled_catalog_yields_academy_areas(self):
        names = research_area_names(load_catalog())

        assert "Quân sự" in names
        assert "Máy tính" in names
        assert not any("Thú y" in n for n in names)


class TestResolveItems:
    def test_code_resolves_to_the_official_name(self):
        kept, dropped = resolve_items(CATALOG, "master", ["8520204"])

        assert kept == ["Kỹ thuật rađa - dẫn đường"]
        assert dropped == []

    def test_code_of_another_level_is_dropped(self):
        """8520204 is a master's code; it does not exist at undergraduate level."""
        kept, dropped = resolve_items(CATALOG, "undergraduate", ["8520204"])

        assert kept == []
        assert dropped == ["8520204"]

    def test_ambiguous_name_is_dropped_not_guessed(self):
        """'Trinh sát kỹ thuật' exists under two codes — guessing is worse than dropping."""
        kept, dropped = resolve_items(CATALOG, "undergraduate", ["Trinh sát kỹ thuật"])

        assert kept == []
        assert dropped == ["Trinh sát kỹ thuật"]

    def test_unique_name_still_resolves_as_a_fallback(self):
        """The model does not always return a code."""
        kept, dropped = resolve_items(CATALOG, "master", ["Kỹ thuật ra đa - dẫn đường"])

        assert kept == ["Kỹ thuật rađa - dẫn đường"], "phải trả về chính tả của danh mục"
        assert dropped == []

    def test_code_with_name_attached_resolves(self):
        """The LLM often returns 'code — name' despite the prompt asking for codes."""
        kept, _ = resolve_items(CATALOG, "master", ["8520204 - Kỹ thuật rađa - dẫn đường"])
        assert kept == ["Kỹ thuật rađa - dẫn đường"]

    def test_group_code_is_not_a_programme(self):
        """74801 is a group heading, not a discipline."""
        kept, dropped = resolve_items(CATALOG, "undergraduate", ["74801"])

        assert kept == []
        assert dropped == ["74801"]

    def test_invented_code_is_dropped_and_reported(self):
        kept, dropped = resolve_items(CATALOG, "undergraduate", ["7999999"])

        assert kept == []
        assert dropped == ["7999999"], "mục bị bỏ phải nhìn thấy được, không biến mất"

    def test_duplicates_collapse_to_one(self):
        kept, _ = resolve_items(
            CATALOG, "undergraduate", ["7480101", "Khoa học máy tính", "7480101"]
        )
        assert kept == ["Khoa học máy tính"]

    def test_unknown_key_yields_nothing(self):
        kept, dropped = resolve_items(CATALOG, "diploma", ["7480101"])
        assert kept == []
        assert dropped == ["7480101"]

    def test_non_string_items_are_dropped_not_crashing(self):
        kept, dropped = resolve_items(CATALOG, "undergraduate", [None, 42, {"a": 1}])
        assert kept == []
        assert len(dropped) == 3


class TestCatalogTextBlock:
    def test_block_shows_group_then_indented_programmes(self):
        block = catalog_text_block(CATALOG, "master")

        assert "85202 Kỹ thuật điện, điện tử và viễn thông" in block
        assert "  8520204 Kỹ thuật rađa - dẫn đường" in block

    def test_empty_level_gives_empty_block(self):
        assert catalog_text_block(CATALOG, "phd") == ""


class TestLoading:
    def test_bundled_catalog_loads_by_default(self):
        catalog = load_catalog()

        assert has_entries(catalog)
        assert all(k in catalog for k in CATALOG_KEYS)

    def test_bundled_catalog_keeps_the_academy_disciplines(self):
        """The BGD filter must keep the disciplines the Academy actually teaches."""
        catalog = load_catalog()
        codes = {code for code, _ in iter_programmes(catalog, "master")}

        assert "8480101" in codes, "Khoa học máy tính"
        assert "8520204" in codes, "Kỹ thuật rađa - dẫn đường"
        assert "8860220" in codes, "Chỉ huy, quản lý kỹ thuật"

    def test_bundled_catalog_drops_unrelated_fields(self):
        """Veterinary medicine, aquaculture… are outside the Academy's scope."""
        catalog = load_catalog()
        names = {name.casefold() for _, name in iter_programmes(catalog, "undergraduate")}

        assert "thú y" not in names
        assert "nuôi trồng thủy sản" not in names
        assert "chăn nuôi" not in names

    def test_bundled_catalog_drops_the_placeholder_bucket(self):
        """'Khác' sits under 73 codes — it maps to nothing."""
        catalog = load_catalog()
        names = {name.casefold() for _, name in iter_programmes(catalog, "undergraduate")}

        assert "khác" not in names

    def test_override_file_wins_over_bundled(self, tmp_path, monkeypatch):
        override = tmp_path / "ctdt_catalog.json"
        override.write_text(json.dumps(CATALOG, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(override))

        catalog = load_catalog()

        assert count_programmes(catalog, "undergraduate") == 4
        assert catalog_source() == "uploaded"

    def test_missing_override_falls_back_to_bundled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(tmp_path / "absent.json"))

        assert has_entries(load_catalog())
        assert catalog_source() == "bundled"

    def test_unreadable_override_degrades_to_bundled(self, tmp_path, monkeypatch):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(broken))

        assert has_entries(load_catalog()), "upload hỏng không được làm trống §3"

    def test_empty_catalog_reports_no_entries(self):
        assert not has_entries(empty_catalog())
        assert not has_entries({})
        assert not has_entries({"undergraduate": [], "master": []})

    def test_group_without_children_is_not_an_entry(self):
        assert not has_entries({"undergraduate": [{"code": "74801", "name": "Máy tính"}]})


class TestValidateAndSave:
    def test_validate_keeps_known_keys_only(self):
        cleaned = validate_catalog(
            {"undergraduate": CATALOG["undergraduate"], "nonsense": [], "_source": "phòng đào tạo"}
        )

        assert count_programmes(cleaned, "undergraduate") == 4
        assert "nonsense" not in cleaned
        assert cleaned["_source"] == "phòng đào tạo"
        assert all(k in cleaned for k in CATALOG_KEYS)

    def test_validate_drops_strong_research_groups(self):
        """Strong research groups are no longer a fixed pick-list."""
        cleaned = validate_catalog({"strong_research_groups": ["Trí tuệ nhân tạo"]})
        assert "strong_research_groups" not in cleaned

    def test_validate_requires_code_and_name(self):
        with pytest.raises(ValueError):
            validate_catalog({"undergraduate": [{"name": "Không có mã", "children": []}]})

    def test_validate_trims_and_collapses_duplicate_codes(self):
        cleaned = validate_catalog(
            {
                "master": [
                    {
                        "code": " 85202 ",
                        "name": "  Kỹ thuật điện  ",
                        "children": [
                            {"code": "8520204", "name": "Kỹ thuật rađa - dẫn đường"},
                            {"code": "8520204", "name": "Trùng mã"},
                        ],
                    }
                ]
            }
        )

        group = cleaned["master"][0]
        assert group["code"] == "85202"
        assert group["name"] == "Kỹ thuật điện"
        assert [c["code"] for c in group["children"]] == ["8520204"]

    def test_validate_rejects_a_blank_code_rather_than_dropping_it(self):
        """Silent skipping is exactly how §3 ends up empty with nobody knowing why."""
        with pytest.raises(ValueError):
            validate_catalog(
                {
                    "master": [
                        {"code": "85202", "name": "X", "children": [{"code": "", "name": "Y"}]}
                    ]
                }
            )

    def test_validate_rejects_non_dict(self):
        with pytest.raises(ValueError):
            validate_catalog(["Ngành A"])

    def test_validate_rejects_non_list_value(self):
        with pytest.raises(ValueError):
            validate_catalog({"undergraduate": "Ngành A"})

    def test_validate_rejects_flat_string_list(self):
        """The old format (a list of names) no longer identifies a discipline."""
        with pytest.raises(ValueError):
            validate_catalog({"undergraduate": ["Ngành Khoa học máy tính"]})

    def test_saved_catalog_is_what_load_returns(self, tmp_path, monkeypatch):
        target = tmp_path / "nested" / "ctdt_catalog.json"
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(target))

        save_catalog({"phd": CATALOG["master"]})

        assert count_programmes(load_catalog(), "phd") == 1
        assert catalog_source() == "uploaded"

    def test_save_never_touches_the_bundled_file(self, tmp_path, monkeypatch):
        from pathlib import Path

        bundled = Path("config/ctdt_catalog.json").resolve()
        before = bundled.read_text(encoding="utf-8")
        monkeypatch.setattr(settings, "ctdt_catalog_path", str(tmp_path / "c.json"))

        save_catalog({"phd": CATALOG["master"]})

        assert bundled.read_text(encoding="utf-8") == before
