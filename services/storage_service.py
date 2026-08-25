"""
Document Storage Service

Handles CRUD operations for documents, pages, layout elements, and tree indices.
"""

import base64
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image
from sqlalchemy.orm import Session

from data.db_models import Document, LayoutElement, Page, TreeIndex, TreeNode
from services.object_storage import get_object_storage
from serving.logic import ServicePageResult
from utils.storage_keys import layout_crop_key, page_image_key
from utils.storage_keys import tree_data_key as tree_object_key


class DocumentStorageService:
    """Service for storing and retrieving OCR documents."""

    def __init__(self, session: Session):
        """
        Initialize storage service.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        self._storage = get_object_storage()

    def _upload_page_image(
        self,
        document_id: str,
        page_number: int,
        image_b64: str,
    ) -> tuple[bytes | None, str | None]:
        if not image_b64:
            return None, None
        try:
            img_data = base64.b64decode(image_b64)
            key = page_image_key(document_id, page_number)
            self._storage.put_bytes(key, img_data, content_type="image/jpeg")
            return img_data, key
        except Exception as e:
            print(f"Warning: Could not upload page image to MinIO: {e}")
            return None, None

    def _upload_crop_image(
        self,
        document_id: str,
        page_number: int,
        sequence_order: int,
        crop_b64: str,
    ) -> str | None:
        if not crop_b64:
            return None
        try:
            crop_data = base64.b64decode(crop_b64)
            key = layout_crop_key(document_id, page_number, sequence_order)
            self._storage.put_bytes(key, crop_data, content_type="image/jpeg")
            return key
        except Exception as e:
            print(f"Warning: Could not upload crop image to MinIO: {e}")
            return None

    def _replace_layout_elements(
        self,
        page: Page,
        layout_items: list,
        img_width: Optional[int],
        img_height: Optional[int],
        *,
        document_id: str,
        page_number: int,
    ) -> None:
        """Drop existing layout rows for *page* and write *layout_items*."""
        self.session.query(LayoutElement).filter(LayoutElement.page_id == page.id).delete(
            synchronize_session=False
        )
        for idx, elem in enumerate(layout_items or []):
            self._save_layout_element(
                page.id,
                elem,
                idx,
                img_width,
                img_height,
                document_id=document_id,
                page_number=page_number,
            )

    def save_page_result(
        self,
        document_id: str,
        page_result: ServicePageResult,
        page_type: str = "scanned",
    ) -> Page:
        """
        Save a page result from OCR processing.

        Upserts by ``(document_id, page_number)`` so Temporal retries do not
        insert duplicate page rows.

        Args:
            document_id: ID of parent document
            page_result: ServicePageResult from OCR processing

        Returns:
            Created or updated Page object
        """
        # Decode image to get dimensions and upload to MinIO
        img_width, img_height = None, None
        image_key = None
        img_data = None
        if page_result.image_base64:
            img_data, image_key = self._upload_page_image(
                document_id,
                page_result.page_num,
                page_result.image_base64,
            )
            if img_data is None:
                try:
                    img_data = base64.b64decode(page_result.image_base64)
                except Exception:
                    img_data = None
            if img_data:
                try:
                    img = Image.open(BytesIO(img_data))
                    img_width, img_height = img.size
                except Exception as e:
                    print(f"Warning: Could not decode image for dimension extraction: {e}")

        page = (
            self.session.query(Page)
            .filter(
                Page.document_id == document_id,
                Page.page_number == page_result.page_num,
            )
            .first()
        )
        if page is None:
            page = Page(
                document_id=document_id,
                page_number=page_result.page_num,
                page_type=page_type,
                markdown_content=page_result.markdown,
                image_base64=None if image_key else page_result.image_base64,
                image_key=image_key,
                image_width=img_width,
                image_height=img_height,
            )
            self.session.add(page)
            self.session.flush()
        else:
            page.page_type = page_type
            page.markdown_content = page_result.markdown
            page.image_base64 = None if image_key else page_result.image_base64
            if image_key:
                page.image_key = image_key
            if img_width is not None:
                page.image_width = img_width
            if img_height is not None:
                page.image_height = img_height
            self.session.flush()

        self._replace_layout_elements(
            page,
            page_result.layout_elements or [],
            img_width or page.image_width,
            img_height or page.image_height,
            document_id=document_id,
            page_number=page_result.page_num,
        )

        self.session.commit()
        self.session.refresh(page)
        return page

    def _save_layout_element(
        self,
        page_id: str,
        element: Dict,
        sequence_order: int,
        img_width: Optional[int],
        img_height: Optional[int],
        *,
        document_id: str | None = None,
        page_number: int | None = None,
    ):
        """Save a layout element with bounding box."""
        # Handle both dict and object formats
        if isinstance(element, dict):
            label = element.get("label", "")
            text_full = element.get("text_full", "")
            text_content = element.get("text_content", element.get("text", ""))
            if label.lower() not in ("title", "sub_title", "heading"):
                text_content = text_full or text_content
            x1, y1 = element.get("bbox_x1", element.get("x1", 0)), element.get(
                "bbox_y1", element.get("y1", 0)
            )
            x2, y2 = element.get("bbox_x2", element.get("x2", 0)), element.get(
                "bbox_y2", element.get("y2", 0)
            )
            crop_image = element.get("crop_image", "")
        else:
            # Assume it's a LayoutElement-like object
            label = getattr(element, "label", "")
            text_content = getattr(element, "text_content", getattr(element, "text", ""))
            text_full = getattr(element, "text_full", "")
            if label.lower() not in ("title", "sub_title", "heading"):
                text_content = text_full or text_content
            x1 = getattr(element, "bbox_x1", getattr(element, "x1", 0))
            y1 = getattr(element, "bbox_y1", getattr(element, "y1", 0))
            x2 = getattr(element, "bbox_x2", getattr(element, "x2", 0))
            y2 = getattr(element, "bbox_y2", getattr(element, "y2", 0))
            crop_image = getattr(element, "crop_image", "")

        # Calculate normalized coordinates (reverse of scaling)
        norm_x1 = norm_y1 = norm_x2 = norm_y2 = None
        if img_width and img_height:
            norm_x1 = (x1 / img_width) * 999.0
            norm_y1 = (y1 / img_height) * 999.0
            norm_x2 = (x2 / img_width) * 999.0
            norm_y2 = (y2 / img_height) * 999.0

        crop_image_key_val = None
        if crop_image and document_id and page_number is not None:
            crop_image_key_val = self._upload_crop_image(
                document_id,
                page_number,
                sequence_order,
                crop_image,
            )

        layout_elem = LayoutElement(
            page_id=page_id,
            label=label,
            text_content=text_content,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            bbox_norm_x1=norm_x1,
            bbox_norm_y1=norm_y1,
            bbox_norm_x2=norm_x2,
            bbox_norm_y2=norm_y2,
            crop_image_base64=None if crop_image_key_val else crop_image,
            crop_image_key=crop_image_key_val,
            sequence_order=sequence_order,
        )
        self.session.add(layout_elem)

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document object or None
        """
        return self.session.query(Document).filter(Document.id == document_id).first()

    def get_document_markdown(self, document_id: str) -> str:
        """
        Get complete markdown for a document (all pages concatenated).

        Args:
            document_id: Document ID

        Returns:
            Combined markdown content
        """
        pages = (
            self.session.query(Page)
            .filter(Page.document_id == document_id)
            .order_by(Page.page_number)
            .all()
        )

        markdown_parts = []
        for page in pages:
            markdown_parts.append(f"# Page {page.page_number}\n\n{page.markdown_content}")

        return "\n\n---\n\n".join(markdown_parts)

    def get_document_elements(
        self, document_id: str, label_filter: Optional[str] = None
    ) -> List[LayoutElement]:
        """
        Get all layout elements for a document.

        Args:
            document_id: Document ID
            label_filter: Optional label to filter by

        Returns:
            List of LayoutElement objects
        """
        query = self.session.query(LayoutElement).join(Page).filter(Page.document_id == document_id)

        if label_filter:
            query = query.filter(LayoutElement.label == label_filter)

        return query.order_by(Page.page_number, LayoutElement.sequence_order).all()

    def save_tree_index(
        self, document_id: str, tree_data: Dict, config: Optional[Dict] = None
    ) -> TreeIndex:
        """
        Save a tree index for a document.

        Args:
            document_id: Document ID
            tree_data: Tree structure as JSON/dict
            config: PageIndex configuration used

        Returns:
            Created TreeIndex object
        """
        tree_index = TreeIndex(document_id=document_id, tree_data=tree_data, config=config or {})
        self.session.add(tree_index)
        self.session.flush()

        # Optionally offload large tree JSON to MinIO
        try:
            import json

            payload = json.dumps(tree_data, ensure_ascii=False)
            if len(payload) > 200_000:
                key = tree_object_key(document_id, tree_index.id)
                self._storage.put_bytes(
                    key,
                    payload.encode("utf-8"),
                    content_type="application/json",
                )
                tree_index.tree_data_key = key
                tree_index.tree_data = None
        except Exception as e:
            print(f"Warning: tree_data MinIO offload skipped: {e}")

        # Extract and save individual nodes for querying
        self._extract_tree_nodes(tree_index.id, tree_data)

        self.session.commit()
        self.session.refresh(tree_index)
        return tree_index

    def _extract_tree_nodes(
        self, tree_index_id: str, tree_data: Dict, parent_node_id: Optional[str] = None
    ):
        """Recursively extract tree nodes for storage."""
        # Extract node information
        node_id = tree_data.get("node_id", tree_data.get("id", ""))
        if not node_id:
            return

        node = TreeNode(
            tree_index_id=tree_index_id,
            node_id=node_id,
            node_type=tree_data.get("type", tree_data.get("node_type")),
            title=tree_data.get("title", tree_data.get("name")),
            summary=tree_data.get("summary", tree_data.get("node_summary")),
            parent_node_id=parent_node_id,
            page_start=tree_data.get("page_start", tree_data.get("start_page")),
            page_end=tree_data.get("page_end", tree_data.get("end_page")),
            token_count=tree_data.get("token_count", tree_data.get("tokens")),
        )
        self.session.add(node)

        # Process children recursively
        children = tree_data.get("children", tree_data.get("child_nodes", []))
        for child in children:
            self._extract_tree_nodes(tree_index_id, child, node_id)

    def get_tree_index(self, document_id: str) -> Optional[TreeIndex]:
        """
        Get the most recent tree index for a document.

        Args:
            document_id: Document ID

        Returns:
            TreeIndex object or None
        """
        return (
            self.session.query(TreeIndex)
            .filter(TreeIndex.document_id == document_id)
            .order_by(TreeIndex.created_at.desc())
            .first()
        )

    def save_unified_elements(
        self,
        document_id: str,
        page_number: int,
        markdown_content: str,
        layout_dicts: list,
        image_width: int = None,
        image_height: int = None,
        page_type: str = None,
        page_image_b64: str | None = None,
    ) -> Page:
        """
        Save a page produced by the unified extraction pipeline.

        Args:
            document_id: ID of parent document.
            page_number: 1-based page number.
            markdown_content: Aggregated text for this page.
            layout_dicts: List of dicts in build_spatial_tree() format
                          (keys: label, bbox_x1/y1/x2/y2, text_content, text_full, …).
            image_width/height: Page coordinate dimensions (PDF points when page_image_b64
                          is rendered at 72 DPI — aligns bboxes with page raster for export).
            page_image_b64: Optional full-page JPEG base64 for bbox crop fallback in export.

        Returns:
            Created or updated Page object.
        """
        image_key = None
        if page_image_b64:
            _, image_key = self._upload_page_image(document_id, page_number, page_image_b64)
            if image_key and (image_width is None or image_height is None):
                try:
                    img = Image.open(BytesIO(base64.b64decode(page_image_b64)))
                    image_width, image_height = img.size
                except Exception:
                    pass

        page = (
            self.session.query(Page)
            .filter(Page.document_id == document_id, Page.page_number == page_number)
            .first()
        )
        if page is None:
            page = Page(
                document_id=document_id,
                page_number=page_number,
                page_type=page_type,
                markdown_content=markdown_content,
                image_base64=None,
                image_key=image_key,
                image_width=int(image_width) if image_width is not None else None,
                image_height=int(image_height) if image_height is not None else None,
            )
            self.session.add(page)
            self.session.flush()
        else:
            page.page_type = page_type
            page.markdown_content = markdown_content
            if image_key:
                page.image_key = image_key
            if image_width is not None:
                page.image_width = int(image_width)
            if image_height is not None:
                page.image_height = int(image_height)
            self.session.flush()

        self._replace_layout_elements(
            page,
            layout_dicts or [],
            page.image_width,
            page.image_height,
            document_id=document_id,
            page_number=page_number,
        )

        self.session.commit()
        self.session.refresh(page)
        return page
