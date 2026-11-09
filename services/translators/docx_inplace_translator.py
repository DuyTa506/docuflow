"""Translate DOCX/DOC files in-place while preserving document structure."""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, List, Optional, Set

from config.settings import settings
from core.pageindex.enrichment.translator import StructuredTranslator
from services.extractors.doc_converter import convert_doc_to_docx
from services.extractors.docx_extractor import (
    _NS_W,
    _NS_WPS,
)
from services.extractors.docx_extractor import _extract_omath_text as extract_omath_text
from services.extractors.docx_extractor import (
    _get_para_text,
    _has_ole_math,
    _has_page_break,
)

ProgressCallback = Optional[Callable[[int, str], Any]]


class DocxInPlaceTranslator:
    """Walk DOCX XML regions and translate paragraph text in place."""

    def __init__(self, translator: StructuredTranslator):
        self.translator = translator

    async def translate_file(
        self,
        source_path: str,
        output_path: str,
        *,
        doc_format: str = "docx",
        on_progress: ProgressCallback = None,
    ) -> dict:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if doc_format == "doc":
            docx_path = convert_doc_to_docx(
                source_path,
                libreoffice_path=settings.libreoffice_path,
            )
            shutil.copy2(docx_path, output_path)
        else:
            shutil.copy2(source_path, output_path)

        from docx import Document as DocxDocument

        doc = DocxDocument(output_path)
        body = doc.element.body
        regions = self._collect_regions(body)
        paragraphs = []
        for region in regions:
            for child in region:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "p":
                    paragraphs.append(child)
                elif tag == "tbl":
                    for tr in child.findall(f"{{{_NS_W}}}tr"):
                        for tc in tr.findall(f"{{{_NS_W}}}tc"):
                            for p in tc.findall(f".//{{{_NS_W}}}p"):
                                paragraphs.append(p)

        # Pass 1: collect translatable paragraphs. Pass 2: translate with
        # bounded concurrency (order preserved by run_parallel). Pass 3:
        # apply results serially — XML mutation stays single-threaded.
        from services.translators._parallel import run_parallel

        translatable: List[tuple] = []
        for p_el in paragraphs:
            if _has_page_break(p_el):
                continue
            if _has_ole_math(p_el) or extract_omath_text(p_el) is not None:
                continue
            text = _get_para_text(p_el)
            if not text or not text.strip():
                continue
            translatable.append((p_el, text))

        async def worker(_idx: int, item: tuple) -> str:
            return await self.translator.translate_text(item[1])

        translations = await run_parallel(
            translatable,
            worker,
            parallelism=max(1, settings.translation_parallelism),
            on_progress=on_progress,
            progress_label="Đoạn",
        )

        flat_parts: List[str] = []
        for (p_el, _), translated in zip(translatable, translations):
            self._set_para_text(p_el, translated)
            flat_parts.append(translated)

        doc.save(output_path)

        return {
            "translation_mode": "docx_inplace",
            "translated_elements": None,
            "translated_content": "\n\n".join(flat_parts),
            "translated_file_path": output_path,
        }

    @staticmethod
    def _collect_regions(body) -> List:
        regions = [body]
        seen_txbx_ids: Set[int] = set()
        for txbx in body.findall(f".//{{{_NS_WPS}}}txbx") + body.findall(f".//{{{_NS_W}}}txbx"):
            content = txbx.find(f"{{{_NS_W}}}txbxContent")
            if content is None:
                for child in txbx:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "txbxContent":
                        content = child
                        break
            if content is not None and id(content) not in seen_txbx_ids:
                seen_txbx_ids.add(id(content))
                regions.append(content)
        return regions

    @staticmethod
    def _set_para_text(p_el, new_text: str) -> None:
        """Distribute the translated paragraph across its existing runs
        proportionally to each run's source-text share (cut at word
        boundaries), so bold/italic/hyperlink runs keep their formatting
        instead of being blanked into run 0. Boundaries are approximate —
        acceptable versus losing intra-paragraph formatting entirely.
        """
        t_nodes = p_el.findall(f".//{{{_NS_W}}}t")
        if not t_nodes:
            return
        texted = [t for t in t_nodes if (t.text or "").strip()]

        if len(texted) <= 1:
            target = texted[0] if texted else t_nodes[0]
            target.text = new_text
            for t in t_nodes:
                if t is not target:
                    t.text = ""
            return

        lengths = [len(t.text) for t in texted]
        total = sum(lengths)
        words = new_text.split()
        pieces: List[str] = []
        start = 0
        cum = 0
        for i, length in enumerate(lengths):
            if i == len(lengths) - 1:
                pieces.append(" ".join(words[start:]))
                break
            cum += length
            cut = round(len(words) * cum / total)
            cut = max(start + 1, min(cut, len(words) - (len(lengths) - 1 - i)))
            pieces.append(" ".join(words[start:cut]))
            start = cut

        _XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
        for i, (t, piece) in enumerate(zip(texted, pieces)):
            if i < len(pieces) - 1 and piece:
                piece += " "
                t.set(_XML_SPACE, "preserve")
            t.text = piece
        for t in t_nodes:
            if t not in texted:
                t.text = ""
