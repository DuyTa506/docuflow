"""Build `config/ctdt_catalog.json` from Phụ lục I of Thông tư 09/2022/TT-BGDĐT.

Why a script rather than a hand-copied JSON file:

1. The source catalog is **legislation** and will be superseded by a new
   circular. Keeping the path from .docx to JSON means the next revision only
   needs a new source file, not 400 retyped lines.
2. **The filter is a business decision, not data.** The national catalog holds
   ~1000 disciplines including veterinary medicine, aquaculture and tourism —
   the Academy does not teach them and the library holds nothing on them.
   Keeping the filter right here, as two annotated sets of codes, means anyone
   can read and change it; burying it in generated JSON does not.

Run:  python scripts/build_ctdt_catalog.py [--source FILE] [--out FILE]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_SOURCE = REPO / "Phu_luc_Thong_tu_09_2022_TT_BGDDT.docx"
DEFAULT_OUT = REPO / "config" / "ctdt_catalog.json"

# Phụ lục I holds three tables, one per level. The leading digit of a code gives
# the level (7 = ĐH, 8 = ThS, 9 = TS); the rest is identical across the three
# tables, so the filter below strips that digit and is shared by all three.
TABLES = {
    "undergraduate": (0, "7"),
    "master": (1, "8"),
    "phd": (2, "9"),
}

# Fields (3-digit code) kept whole — 2 digits once the level digit is stripped.
KEEP_FIELDS = {
    "44": "Khoa học tự nhiên",  # vật lý, cơ học, hóa học, KH vật liệu, trắc địa, khí tượng
    "46": "Toán và thống kê",  # Toán ứng dụng, Cơ sở toán học cho tin học
    "48": "Máy tính và công nghệ thông tin",
    "51": "Công nghệ kỹ thuật",
    "52": "Kỹ thuật",
    "58": "Kiến trúc và xây dựng",
    "86": "An ninh, Quốc phòng",
}

# Groups (5-digit code) kept individually, where the field as a whole is not
# relevant but one group inside it is — 4 digits once the level digit is
# stripped.
KEEP_GROUPS = {
    "3404": "Quản trị - Quản lý",  # chứa Quản lý khoa học và công nghệ (x340412)
    "8502": "Dịch vụ an toàn lao động và vệ sinh công nghiệp",
}

# Every field and group carries a "Khác" entry ending in 90 — the trap described
# in utils/ctdt_catalog.py: the word "Khác" alone appears under 73 different
# codes. It is not a discipline and maps to nothing, so drop it outright.
PLACEHOLDER_SUFFIX = "90"

SOURCE_NOTE = (
    "Phụ lục I Thông tư 09/2022/TT-BGDĐT (Danh mục thống kê ngành đào tạo ĐH/ThS/TS), "
    "đã lọc giữ các lĩnh vực và nhóm ngành liên quan Học viện KTQS. "
    "Sinh bằng scripts/build_ctdt_catalog.py — sửa bộ lọc trong script rồi chạy lại, "
    "đừng sửa tay file này."
)


def _keep(code: str) -> bool:
    """Is this code within the Academy's scope? (level digit already stripped)"""
    if code.endswith(PLACEHOLDER_SUFFIX):
        return False
    return code[1:3] in KEEP_FIELDS or code[1:5] in KEEP_GROUPS


def _read_rows(table, name_col: int) -> list[tuple[str, str]]:
    """(code, name) in order, skipping merged-cell duplicates and header rows."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for row in table.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        code = cells[0]
        if not code.isdigit() or code in seen:
            continue
        seen.add(code)
        rows.append((code, cells[name_col]))
    return rows


def build_level(table, name_col: int, digit: str) -> list[dict]:
    """The filtered group → discipline tree for one level."""
    groups: list[dict] = []
    current: dict | None = None
    for code, name in _read_rows(table, name_col):
        if not code.startswith(digit):
            continue
        if len(code) == 5:
            current = {"code": code, "name": name, "children": []} if _keep(code) else None
            if current:
                groups.append(current)
        elif len(code) == 7 and current and _keep(code):
            current["children"].append({"code": code, "name": name})
    # A group left with no disciplines (all filtered out) offers nothing to pick.
    return [g for g in groups if g["children"]]


def build(source: Path) -> dict:
    from docx import Document

    doc = Document(str(source))
    catalog: dict = {"_source": SOURCE_NOTE}
    for key, (index, digit) in TABLES.items():
        # The ĐH table has merged cells so names sit in column 2; the other two
        # tables use column 1.
        catalog[key] = build_level(doc.tables[index], 2 if index == 0 else 1, digit)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"Không thấy file nguồn: {args.source}", file=sys.stderr)
        return 1

    catalog = build(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for key, (_, _) in TABLES.items():
        groups = catalog[key]
        leaves = sum(len(g["children"]) for g in groups)
        print(f"{key:>14}: {len(groups):3d} groups, {leaves:3d} disciplines")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
