"""A digest must reach a library catalogue in a format it already imports.

The library system integrating with DocuFlow is unknown beyond "MSSQL", so the
record is offered in the two formats every catalogue speaks: MARC21 (MARCXML)
and simple Dublin Core (oai_dc). Both are built from the same DigestResult the
§-template export uses.
"""

import xml.etree.ElementTree as ET

import pytest

from services.digest_service import ChapterEntry, DigestResult, KeywordEntry, ResearchDirectionEntry
from utils.library_formats import split_authors, to_dublin_core, to_marcxml

MARC = {"marc": "http://www.loc.gov/MARC21/slim"}
DC = {
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _digest(**overrides):
    base = dict(
        document_id="DOC_042",
        title="Ballistics",
        source_language="en",
        original_filename="ballistics.pdf",
        bibliographic={
            "title_display": "Ballistics: Theory and Design (Đạn đạo: lý thuyết và thiết kế)",
            "authors": "Donald E. Carlucci, Sidney S. Jacobson",
            "publisher": "CRC Press",
            "year": "2008",
            "isbn": "978-1-4200-6618-0",
            "doi": "",
            "pages": "502",
        },
        abstract="Tài liệu gồm 3 phần & trình bày <đạn đạo> trong, ngoài và tận cùng.",
        chapters=[
            ChapterEntry(1, "Đạn đạo trong", "Interior Ballistics", "…"),
            ChapterEntry(2, "Đạn đạo ngoài", "Exterior Ballistics", "…"),
        ],
        keywords=[
            KeywordEntry("interior ballistics", 0.9, "đạn đạo trong (interior ballistics)"),
            KeywordEntry("projectile", 0.7, ""),
        ],
        usage_scope={
            "undergraduate": ["Kỹ thuật vũ khí"],
            "master": [],
            "phd": ["Cơ học"],
            "strong_research_groups": [],
        },
        research_directions=[ResearchDirectionEntry("Mô phỏng đạn đạo", 0.8)],
    )
    base.update(overrides)
    return DigestResult(**base)


def _marc_fields(xml: str) -> dict:
    root = ET.fromstring(xml)
    out: dict = {}
    for f in root.iter("{http://www.loc.gov/MARC21/slim}datafield"):
        subs = {s.get("code"): s.text for s in f.findall("marc:subfield", MARC)}
        out.setdefault(f.get("tag"), []).append((f.get("ind1"), f.get("ind2"), subs))
    for f in root.iter("{http://www.loc.gov/MARC21/slim}controlfield"):
        out[f.get("tag")] = f.text
    return out


class TestSplitAuthors:
    def test_comma_separated_names(self):
        assert split_authors("Donald E. Carlucci, Sidney S. Jacobson") == [
            "Donald E. Carlucci",
            "Sidney S. Jacobson",
        ]

    def test_semicolons_win_so_inverted_names_survive(self):
        assert split_authors("Carlucci, Donald E.; Jacobson, Sidney S.") == [
            "Carlucci, Donald E.",
            "Jacobson, Sidney S.",
        ]

    def test_empty(self):
        assert split_authors("") == []
        assert split_authors(None) == []


class TestMarcXml:
    def test_core_catalogue_fields(self):
        f = _marc_fields(to_marcxml(_digest()))

        assert f["001"] == "DOC_042"
        assert f["020"][0][2] == {"a": "978-1-4200-6618-0"}
        assert f["041"][0][2] == {"a": "eng"}
        assert f["100"][0][2] == {"a": "Donald E. Carlucci"}
        assert f["700"][0][2] == {"a": "Sidney S. Jacobson"}
        assert f["245"][0][2]["a"].startswith("Ballistics: Theory and Design")
        assert f["264"][0][2] == {"b": "CRC Press", "c": "2008"}
        assert f["300"][0][2] == {"a": "502 tr."}

    def test_digest_sections_go_to_notes_and_subjects(self):
        f = _marc_fields(to_marcxml(_digest()))

        assert "<đạn đạo>" in f["520"][0][2]["a"]  # escaped on the way out, intact on the way in
        assert f["505"][0][2]["a"] == "1. Đạn đạo trong (Interior Ballistics) -- 2. Đạn đạo ngoài (Exterior Ballistics)"
        assert [s["a"] for _, _, s in f["653"]] == [
            "đạn đạo trong (interior ballistics)",
            "projectile",
        ]
        assert f["521"][0][2]["a"] == "Đại học: Kỹ thuật vũ khí; Tiến sĩ: Cơ học"
        assert "Mô phỏng đạn đạo" in f["500"][0][2]["a"]

    def test_title_indicator_follows_main_entry(self):
        with_author = _marc_fields(to_marcxml(_digest()))
        no_author = _marc_fields(
            to_marcxml(_digest(bibliographic={"title_display": "Untitled", "authors": ""}))
        )
        assert with_author["245"][0][0] == "1"
        assert no_author["245"][0][0] == "0"
        assert "100" not in no_author

    def test_empty_values_produce_no_empty_fields(self):
        f = _marc_fields(
            to_marcxml(
                _digest(
                    bibliographic={"title_display": "T"},
                    abstract=None,
                    chapters=[],
                    keywords=[],
                    usage_scope={},
                    research_directions=[],
                )
            )
        )
        for tag in ("020", "024", "100", "264", "300", "500", "505", "520", "521", "653"):
            assert tag not in f, tag

    def test_doi_is_a_standard_identifier(self):
        f = _marc_fields(to_marcxml(_digest(bibliographic={"title_display": "T", "doi": "10.1/x"})))
        assert f["024"][0][:2] == ("7", " ")
        assert f["024"][0][2] == {"a": "10.1/x", "2": "doi"}

    @pytest.mark.parametrize("lang,code", [("vi", "vie"), ("ru", "rus"), ("zh", "chi"), ("xx", None)])
    def test_language_code(self, lang, code):
        f = _marc_fields(to_marcxml(_digest(source_language=lang)))
        assert (f["041"][0][2]["a"] if "041" in f else None) == code


class TestDublinCore:
    def test_oai_dc_record(self):
        root = ET.fromstring(to_dublin_core(_digest()))

        def values(el):
            return [e.text for e in root.findall(f"dc:{el}", DC)]

        assert root.tag == "{http://www.openarchives.org/OAI/2.0/oai_dc/}dc"
        assert values("title")[0].startswith("Ballistics")
        assert values("creator") == ["Donald E. Carlucci", "Sidney S. Jacobson"]
        assert values("publisher") == ["CRC Press"]
        assert values("date") == ["2008"]
        assert values("language") == ["en"]
        assert values("type") == ["Text"]
        assert "docuflow:DOC_042" in values("identifier")
        assert "ISBN:978-1-4200-6618-0" in values("identifier")
        assert values("subject") == ["đạn đạo trong (interior ballistics)", "projectile"]
        assert "<đạn đạo>" in values("description")[0]
