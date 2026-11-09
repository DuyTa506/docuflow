"""Catalogue formats for a digest: MARC21 (MARCXML) and Dublin Core (oai_dc).

A library system integrating with DocuFlow imports the digest as a record it
already understands rather than a DocuFlow-specific JSON. The mapping keeps to
fields every MARC21 catalogue indexes:

  001 document id   020 ISBN        024 7_ DOI ($2 doi)   041 language
  100 / 700 authors 245 title       264 _1 publisher/year 300 pages
  500 research directions (note)    505 0_ chapter list   520 abstract
  521 intended audience (usage scope)                     653 keywords
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional, Tuple

MARC_NS = "http://www.loc.gov/MARC21/slim"
OAI_DC_NS = "http://www.openarchives.org/OAI/2.0/oai_dc/"
DC_NS = "http://purl.org/dc/elements/1.1/"

# ISO 639-2/B, the code MARC 041 expects.
_MARC_LANG = {"en": "eng", "vi": "vie", "ru": "rus", "zh": "chi", "fr": "fre", "de": "ger", "ja": "jpn"}

_AUDIENCE = (
    ("undergraduate", "Đại học"),
    ("master", "Thạc sĩ"),
    ("phd", "Tiến sĩ"),
    ("strong_research_groups", "Nhóm nghiên cứu mạnh"),
)

_LEADER = "00000nam a2200000 a 4500"


def split_authors(authors: Optional[str]) -> List[str]:
    """Split the bibliographic `authors` string into names.

    Semicolons separate names when present, so inverted forms
    ("Carlucci, Donald E.") survive; otherwise commas do.
    """
    if not authors:
        return []
    sep = ";" if ";" in authors else ","
    return [a.strip() for a in authors.split(sep) if a.strip()]


def _bib(digest, key: str) -> str:
    return str((digest.bibliographic or {}).get(key) or "").strip()


def _title(digest) -> str:
    return _bib(digest, "title_display") or (digest.title or "").strip()


def _keywords(digest) -> List[str]:
    return [(k.display or k.keyword).strip() for k in digest.keywords if (k.display or k.keyword).strip()]


def _contents_note(digest) -> str:
    parts = []
    for ch in digest.chapters:
        title = ch.title_vi.strip()
        if ch.title_original and ch.title_original.strip() and ch.title_original.strip() != title:
            title = f"{title} ({ch.title_original.strip()})"
        parts.append(f"{ch.number}. {title}")
    return " -- ".join(parts)


def _audience_note(digest) -> str:
    scope = digest.usage_scope or {}
    parts = [f"{label}: {', '.join(scope[key])}" for key, label in _AUDIENCE if scope.get(key)]
    return "; ".join(parts)


def _directions_note(digest) -> str:
    names = [d.direction_name.strip() for d in digest.research_directions if d.direction_name.strip()]
    return f"Hướng nghiên cứu gợi ý: {'; '.join(names)}" if names else ""


def _pages(digest) -> str:
    pages = _bib(digest, "pages")
    return f"{pages} tr." if re.fullmatch(r"\d+", pages) else pages


def to_marcxml(digest) -> str:
    """One MARCXML <record> for the digest; empty values produce no field."""
    ET.register_namespace("marc", MARC_NS)
    record = ET.Element(f"{{{MARC_NS}}}record")
    ET.SubElement(record, f"{{{MARC_NS}}}leader").text = _LEADER
    ET.SubElement(record, f"{{{MARC_NS}}}controlfield", tag="001").text = digest.document_id

    def field(tag: str, subs: Iterable[Tuple[str, str]], ind1: str = " ", ind2: str = " ") -> None:
        subs = [(code, value) for code, value in subs if value]
        if not subs:
            return
        df = ET.SubElement(record, f"{{{MARC_NS}}}datafield", tag=tag, ind1=ind1, ind2=ind2)
        for code, value in subs:
            ET.SubElement(df, f"{{{MARC_NS}}}subfield", code=code).text = value

    authors = split_authors(_bib(digest, "authors"))
    field("020", [("a", _bib(digest, "isbn"))])
    if _bib(digest, "doi"):
        field("024", [("a", _bib(digest, "doi")), ("2", "doi")], ind1="7")
    field("041", [("a", _MARC_LANG.get((digest.source_language or "").lower(), ""))])
    if authors:
        field("100", [("a", authors[0])], ind1="1")
    field("245", [("a", _title(digest))], ind1="1" if authors else "0", ind2="0")
    field("264", [("b", _bib(digest, "publisher")), ("c", _bib(digest, "year"))], ind2="1")
    field("300", [("a", _pages(digest))])
    field("500", [("a", _directions_note(digest))])
    field("505", [("a", _contents_note(digest))], ind1="0")
    field("520", [("a", (digest.abstract or "").strip())])
    field("521", [("a", _audience_note(digest))])
    for kw in _keywords(digest):
        field("653", [("a", kw)])
    for name in authors[1:]:
        field("700", [("a", name)], ind1="1")

    return ET.tostring(record, encoding="unicode", xml_declaration=False)


def to_dublin_core(digest) -> str:
    """One simple Dublin Core record (oai_dc), as OAI-PMH harvesters expect."""
    ET.register_namespace("oai_dc", OAI_DC_NS)
    ET.register_namespace("dc", DC_NS)
    root = ET.Element(f"{{{OAI_DC_NS}}}dc")

    def dc(name: str, value: str) -> None:
        if value:
            ET.SubElement(root, f"{{{DC_NS}}}{name}").text = value

    dc("title", _title(digest))
    for name in split_authors(_bib(digest, "authors")):
        dc("creator", name)
    for kw in _keywords(digest):
        dc("subject", kw)
    dc("description", (digest.abstract or "").strip())
    dc("publisher", _bib(digest, "publisher"))
    dc("date", _bib(digest, "year"))
    dc("type", "Text")
    dc("identifier", f"docuflow:{digest.document_id}")
    if _bib(digest, "isbn"):
        dc("identifier", f"ISBN:{_bib(digest, 'isbn')}")
    if _bib(digest, "doi"):
        dc("identifier", f"doi:{_bib(digest, 'doi')}")
    dc("language", (digest.source_language or "").strip())
    return ET.tostring(root, encoding="unicode")
