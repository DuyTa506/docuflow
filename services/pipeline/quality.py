"""Quality gates for digest pipeline output."""

from typing import Any, Optional

from config.settings import settings
from data.database import get_db_manager
from data.db_models import MainContent, Summary, TreeIndex
from services.digest_service import DigestService
from services.pipeline.constants import STAGE_LABELS
from utils.ctdt_catalog import has_entries, load_catalog
from utils.digest_format import is_chapter_schema
from utils.doc_kind import BOOK, PROCEEDINGS

# The official template lists exactly this many keywords in §2.3.
KEYWORD_TARGET = 20


def build_quality_report(
    document_id: str,
    stage_failures: Optional[dict[str, str]] = None,
    tree_fallback: bool = False,
) -> dict[str, Any]:
    db_manager = get_db_manager()
    with db_manager.session() as db:
        digest = DigestService().assemble(db, document_id)

        has_tree = (
            db.query(TreeIndex).filter(TreeIndex.document_id == document_id).first() is not None
        )

        mc = (
            db.query(MainContent)
            .filter(MainContent.document_id == document_id, MainContent.status == "COMPLETED")
            .order_by(MainContent.created_at.desc())
            .first()
        )
        tree_used = bool(has_tree and mc and is_chapter_schema(mc.details))

        summary = (
            db.query(Summary)
            .filter(Summary.document_id == document_id, Summary.status == "COMPLETED")
            .order_by(Summary.created_at.desc())
            .first()
        )
        abstract_sentences = 0
        if summary and summary.content:
            abstract_sentences = len(
                [
                    s
                    for s in summary.content.replace("!", ".").replace("?", ".").split(".")
                    if s.strip()
                ]
            )

        warnings: list[str] = []
        for stage, err in (stage_failures or {}).items():
            label = STAGE_LABELS.get(stage, stage)
            warnings.append(f"Giai đoạn {label} thất bại — {err[:200]}")
        if tree_fallback:
            warnings.append("Dựng cây mục lục thất bại — tóm tắt/nội dung chạy chế độ fallback")
        if not has_tree:
            warnings.append(
                "Không có TreeIndex — tóm tắt/nội dung có thể dùng fallback chunk/heading"
            )
        elif not tree_used:
            warnings.append("§2.2 có thể dùng heading fallback thay vì tree-walk")
        if abstract_sentences and (abstract_sentences < 8 or abstract_sentences > 20):
            warnings.append(f"Tóm tắt có ~{abstract_sentences} câu (mục tiêu 10–15)")
        if "bibliographic" in digest.missing:
            warnings.append("§1 thư mục học thiếu tác giả/NXB/ISBN")
        if "main_content" in digest.missing:
            warnings.append("§2.2 nội dung chính theo chương trống")
        if "keywords" in digest.missing:
            warnings.append("§2.3 từ khóa trống")
        if "usage_scope" in digest.missing:
            # "No programme matched" and "no programme list was ever loaded"
            # look identical in the digest; only the second is actionable.
            if has_entries(load_catalog()):
                warnings.append("§3 phạm vi CTĐT/NNC trống")
            else:
                warnings.append(
                    "§3 phạm vi CTĐT/NNC trống vì chưa nạp danh mục chương trình đào tạo "
                    "— tải lên qua PUT /api/v2/catalog/ctdt"
                )
        if "research_directions" in digest.missing:
            warnings.append("§3 hướng nghiên cứu trống")

        # §2.3 — the template asks for 20 keywords in `Tiếng Việt (nguyên bản)`
        # form. `LIMIT 20` in the assembler is a ceiling, so a short or
        # monolingual list shipped without comment.
        keywords = list(getattr(digest, "keywords", None) or [])
        if keywords and len(keywords) < KEYWORD_TARGET:
            warnings.append(f"§2.3 chỉ có {len(keywords)}/{KEYWORD_TARGET} từ khóa theo mẫu")
        if keywords and (getattr(digest, "source_language", "") or "").lower() != "vi":
            monolingual = [
                k for k in keywords if (k.display or "").strip() == (k.keyword or "").strip()
            ]
            if len(monolingual) * 2 >= len(keywords):
                warnings.append(
                    f"§2.3 {len(monolingual)}/{len(keywords)} từ khóa chưa có dạng song ngữ "
                    "«Tiếng Việt (nguyên bản)»"
                )

        # §2.2 unit selection — a tree whose levels come from a percentile cut can
        # yield hundreds of pseudo-chapters, and that used to ship silently.
        details = (mc.details or {}) if mc else {}
        unit_tier = details.get("unit_selection_tier")
        median_unit_chars = details.get("median_unit_chars") or 0
        coverage_ratio = details.get("coverage_ratio")
        chapter_count = len(digest.chapters)
        # The ceiling scales with document length, so read the one actually applied
        # rather than re-deriving it from settings.
        max_units = details.get("max_units") or settings.main_content_max_units
        if max_units and chapter_count > max_units:
            warnings.append(
                f"§2.2 có {chapter_count} mục (ngưỡng {max_units}) — "
                "nghi ngờ phân mạch theo tiêu đề"
            )
        if median_unit_chars and median_unit_chars < settings.main_content_min_unit_chars:
            warnings.append(
                f"§2.2 các mục quá nhỏ (trung vị {median_unit_chars} ký tự) — "
                "nội dung có thể bị phân mảnh"
            )
        if unit_tier == "mass_segmentation":
            warnings.append(
                "§2.2 không nhận diện được tiêu đề chương — đã chia theo khối lượng nội dung"
            )
        if coverage_ratio is not None and coverage_ratio < settings.main_content_unit_coverage_min:
            warnings.append(f"§2.2 chỉ bao phủ {coverage_ratio:.0%} nội dung tài liệu")
        degraded_chapters = (mc.details or {}).get("degraded_chapters", 0) if mc else 0
        if degraded_chapters:
            warnings.append(f"{degraded_chapters} chương dùng văn bản gốc do lỗi LLM khi tóm tắt")
        raw_passthrough_chapters = (
            (mc.details or {}).get("raw_passthrough_chapters", 0) if mc else 0
        )
        if raw_passthrough_chapters:
            warnings.append(
                f"{raw_passthrough_chapters} chương chỉ có tiêu đề gốc do nội dung quá ngắn (chưa qua tóm tắt LLM)"
            )
        auxiliary_sections = (mc.details or {}).get("auxiliary_sections", 0) if mc else 0
        gate_degraded = bool((mc.details or {}).get("gate_degraded")) if mc else False
        if gate_degraded:
            warnings.append(
                "Bước phân loại đề mục §2.2 thất bại — giữ nguyên toàn bộ đề mục, "
                "không gộp phần phụ trợ"
            )

        # Thể loại — mẫu tổng thuật có hai chế độ và chọn nhầm thì §2.1/§2.2
        # sai về thể loại chứ không chỉ sai câu chữ.
        doc_kind = details.get("doc_kind") or BOOK
        doc_kind_source = details.get("doc_kind_source")
        if details.get("doc_kind_degraded"):
            warnings.append(
                "Không phân loại được thể loại tài liệu (LLM lỗi) — đang tổng thuật "
                "theo dạng sách; đặt lại qua PUT /api/v2/documents/{id}/digest/kind"
            )
        if doc_kind == PROCEEDINGS:
            if unit_tier == "mass_segmentation":
                warnings.append(
                    "§2.2 kỷ yếu nhưng ranh giới BBKH cắt theo khối lượng — "
                    "các mục nhiều khả năng không trùng ranh giới bài báo"
                )
            if doc_kind_source == "llm":
                # §2.1 chốt thể loại bằng bằng chứng tất định vì chạy song song
                # với §2.2 — nguồn `llm` nghĩa là từ vựng không bắt được gì, nên
                # tóm tắt chắc chắn vẫn đang viết theo dạng sách.
                warnings.append(
                    "Thể loại «kỷ yếu» do LLM suy đoán (không có từ khóa nào trên bìa), "
                    "nên §2.1 vẫn viết theo dạng sách — xác nhận qua PUT "
                    "/api/v2/documents/{id}/digest/kind rồi chạy lại summarize"
                )

        ok = "abstract" not in digest.missing and "main_content" not in digest.missing

        return {
            "ok": ok,
            "missing": digest.missing,
            "warnings": warnings,
            "doc_kind": doc_kind,
            "doc_kind_source": doc_kind_source,
            "tree_used": tree_used,
            "has_tree_index": has_tree,
            "abstract_sentence_count": abstract_sentences,
            "chapter_count": chapter_count,
            "unit_selection_tier": unit_tier,
            "median_unit_chars": median_unit_chars,
            "coverage_ratio": coverage_ratio,
            "max_units": max_units,
            "raw_passthrough_chapters": raw_passthrough_chapters,
            "auxiliary_sections": auxiliary_sections,
            "gate_degraded": gate_degraded,
            "stage_failures": dict(stage_failures or {}),
            "tree_fallback": tree_fallback,
        }
