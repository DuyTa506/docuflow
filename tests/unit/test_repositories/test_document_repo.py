"""Unit tests for DocumentRepository bulk-delete helpers."""

import uuid

from sqlalchemy import func, select

from data.db_models import (
    DigitizedText,
    Document,
    DocumentKeyword,
    DocumentResearchDirection,
    Keyword,
    KeywordExtraction,
    LayoutElement,
    MainContent,
    Page,
    ResearchDirection,
    ResearchExtraction,
    Summary,
    Task,
    Translation,
    TreeIndex,
    TreeNode,
)
from data.repositories.document_repo import DocumentRepository


def _doc_id() -> str:
    return f"DOC_{uuid.uuid4().hex[:8]}"


def _seed_document(session, doc_id: str | None = None) -> str:
    """Insert a document with extraction artifacts and downstream rows."""
    doc_id = doc_id or _doc_id()
    session.add(
        Document(
            id=doc_id,
            title="Cascade test",
            original_filename="test.pdf",
            total_pages=1,
            processing_status="EXTRACTED",
        )
    )
    session.flush()

    page = Page(document_id=doc_id, page_number=1, markdown_content="# Page")
    session.add(page)
    session.flush()

    session.add(
        LayoutElement(
            page_id=page.id,
            label="title",
            text_content="Heading",
            bbox_x1=0,
            bbox_y1=0,
            bbox_x2=100,
            bbox_y2=20,
            sequence_order=1,
        )
    )
    session.add(
        DigitizedText(
            document_id=doc_id,
            ocr_content="raw",
            normalized_content="normalized",
        )
    )

    tree = TreeIndex(document_id=doc_id, tree_data={"nodes": []})
    session.add(tree)
    session.flush()
    session.add(
        TreeNode(
            tree_index_id=tree.id,
            node_id="n1",
            node_type="section",
            title="Intro",
        )
    )

    session.add(
        Translation(document_id=doc_id, target_language="vi", translated_content="dịch")
    )
    session.add(Summary(document_id=doc_id, summary_type="short", content="summary"))
    session.add(MainContent(document_id=doc_id, details={"key_points": []}))

    kw = Keyword(keyword_name=f"kw-{doc_id}")
    session.add(kw)
    session.flush()
    session.add(DocumentKeyword(document_id=doc_id, keyword_id=kw.id, weight=1.0))
    session.add(KeywordExtraction(document_id=doc_id, status="COMPLETED"))

    direction = ResearchDirection(direction_name=f"dir-{doc_id}")
    session.add(direction)
    session.flush()
    session.add(
        DocumentResearchDirection(document_id=doc_id, direction_id=direction.id, confidence=0.9)
    )
    session.add(ResearchExtraction(document_id=doc_id, status="COMPLETED"))
    session.add(
        Task(id=f"TASK_{doc_id}", document_id=doc_id, task_type="OCR", status="COMPLETED")
    )
    session.commit()
    return doc_id


def _count_for_document(session, model, doc_id: str) -> int:
    return session.scalar(
        select(func.count()).select_from(model).where(model.document_id == doc_id)
    )


class TestDeleteCascade:
    def test_removes_document_and_related_rows(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        repo = DocumentRepository(test_db_session)

        assert repo.delete_cascade(doc_id) is True
        assert repo.get(doc_id) is None
        assert _count_for_document(test_db_session, Page, doc_id) == 0
        assert _count_for_document(test_db_session, DigitizedText, doc_id) == 0
        assert _count_for_document(test_db_session, Translation, doc_id) == 0
        assert _count_for_document(test_db_session, Summary, doc_id) == 0
        assert _count_for_document(test_db_session, Task, doc_id) == 0
        assert (
            test_db_session.scalar(
                select(func.count()).select_from(TreeIndex).where(TreeIndex.document_id == doc_id)
            )
            == 0
        )

    def test_missing_document_returns_false(self, test_db_session):
        repo = DocumentRepository(test_db_session)
        assert repo.delete_cascade("DOC_missing") is False


class TestClearExtractionArtifacts:
    def test_removes_extraction_rows_keeps_document(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        repo = DocumentRepository(test_db_session)

        repo.clear_extraction_artifacts(doc_id)

        assert repo.get(doc_id) is not None
        assert _count_for_document(test_db_session, Page, doc_id) == 0
        assert _count_for_document(test_db_session, DigitizedText, doc_id) == 0
        assert _count_for_document(test_db_session, Translation, doc_id) == 1
        assert _count_for_document(test_db_session, Summary, doc_id) == 1
        assert _count_for_document(test_db_session, Task, doc_id) == 1

    def test_update_digitized_text_sets_overridden_flag(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        repo = DocumentRepository(test_db_session)

        updated = repo.update_digitized_text(doc_id, "user correction")

        assert updated is not None
        assert updated.normalized_content == "user correction"
        assert updated.text_overridden is True


class TestGetPages:
    def test_get_pages_without_limit_returns_all_ordered(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        test_db_session.add(Page(document_id=doc_id, page_number=2, markdown_content="# Two"))
        test_db_session.add(Page(document_id=doc_id, page_number=3, markdown_content="# Three"))
        test_db_session.commit()

        repo = DocumentRepository(test_db_session)
        pages = repo.get_pages(doc_id)

        assert [p.page_number for p in pages] == [1, 2, 3]

    def test_get_pages_with_limit_fetches_only_requested_pages(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        test_db_session.add(Page(document_id=doc_id, page_number=2, markdown_content="# Two"))
        test_db_session.add(Page(document_id=doc_id, page_number=3, markdown_content="# Three"))
        test_db_session.commit()

        repo = DocumentRepository(test_db_session)
        pages = repo.get_pages(doc_id, limit=2)

        assert [p.page_number for p in pages] == [1, 2]


class TestCountElements:
    def test_count_elements_returns_total(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        repo = DocumentRepository(test_db_session)
        assert repo.count_elements(doc_id) == 1

    def test_count_elements_filters_by_label(self, test_db_session):
        doc_id = _seed_document(test_db_session)
        repo = DocumentRepository(test_db_session)
        page = test_db_session.query(Page).filter(Page.document_id == doc_id).first()
        test_db_session.add(
            LayoutElement(
                page_id=page.id,
                label="text",
                text_content="Body",
                bbox_x1=0,
                bbox_y1=20,
                bbox_x2=100,
                bbox_y2=40,
                sequence_order=2,
            )
        )
        test_db_session.commit()
        assert repo.count_elements(doc_id) == 2
        assert repo.count_elements(doc_id, label="title") == 1
        assert repo.count_elements(doc_id, label="text") == 1
