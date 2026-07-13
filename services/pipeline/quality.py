"""Quality gates for digest pipeline output."""

from typing import Any, Optional

from data.database import get_db_manager
from data.db_models import MainContent, Summary, TreeIndex
from services.digest_service import DigestService
from services.pipeline.constants import STAGE_LABELS
from utils.digest_format import is_chapter_schema


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
            warnings.append("§3 phạm vi CTĐT/NNC trống")
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

        ok = "abstract" not in digest.missing and "main_content" not in digest.missing

        return {
            "ok": ok,
            "missing": digest.missing,
            "warnings": warnings,
            "tree_used": tree_used,
            "has_tree_index": has_tree,
            "abstract_sentence_count": abstract_sentences,
            "chapter_count": len(digest.chapters),
            "raw_passthrough_chapters": raw_passthrough_chapters,
            "stage_failures": dict(stage_failures or {}),
            "tree_fallback": tree_fallback,
        }
