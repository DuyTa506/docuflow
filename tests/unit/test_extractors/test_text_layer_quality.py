"""Text-layer quality gate: fluency over en/zh/ru/vi character n-grams."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from services.extractors.pdf_text_extractor import classify_pages
from services.extractors.text_layer_quality import (
    classify_extracted_text,
    evaluate_text_layer,
    reset_model_cache,
)

# Excerpt shaped like 2-2024-cq-271.PDF_ocr.docx — long Latin, zero Vietnamese tones.
VI_GARBAGE = (
    "THANH TRA CHNH PHU CONG HOA XA HQI CHU NGHIA VIET NAM Mic lap - Tir do - Hanh pluic "
    "Sa:02.14/TB-TTCP Ha A T ngetycl6thang 02 nam 2024 THONG BAO KET LUAN THANH TRA "
    "Cong tfic quan 157, sir dung dat dai theo tinh than Nghi quytt 73/NQ-CP va 116/NQ-CP "
    "cua Chinh phii; cong tfic quy hoach va thuc hien quy hoach xay dun cua UBND tinh Hung Yen "
    "Ngdy 28/12/2023, Tong Thanh tra Chinh phii ban hanh Ket Juan thanh tra so 3136/KL-TTCP "
    "ye cong tic quan VT, sir dung cat dai theo tinh thin Nghi guy& 73/NQ-CP va 116/NQ-CP "
    "cua Chinh pha; cong tic quy hoach va thuc hien quy hoach xay dung"
)

VI_GOOD = (
    "Thanh tra Chính phủ ban hành thông báo kết luận thanh tra về công tác "
    "quản lý, sử dụng đất đai theo tinh thần Nghị quyết của Chính phủ; "
    "công tác quy hoạch và thực hiện quy hoạch xây dựng của Ủy ban nhân dân tỉnh. "
    "Kiến nghị giao địa phương tổ chức khắc phục hạn chế, thiếu sót đã nêu."
)

EN_GOOD = (
    "This paper presents a scalable architecture for document understanding "
    "that combines layout analysis with neural optical character recognition. "
    "Experimental results on multilingual corpora show consistent gains over "
    "prior baselines in both precision and recall for scanned page recovery."
)

ZH_GOOD = (
    "本文提出一种面向文档理解的可扩展架构，将版面分析与神经光学字符识别相结合。"
    "在多语种语料上的实验结果表明，该方法在准确率与召回率方面均优于已有基线。"
    "城市规划与行政管理文件同样可用于评估跨域泛化能力。"
) * 2

RU_GOOD = (
    "В статье представлена масштабируемая архитектура понимания документов, "
    "сочетающая анализ макета с нейронным оптическим распознаванием символов. "
    "Эксперименты на многоязычных корпусах показывают устойчивый рост точности "
    "и полноты по сравнению с существующими базовыми методами распознавания."
)

NUMERIC_TABLE = "1 2 3 4 5 10 20 30 100 200 300 1.5 2.5 3.5 99.9 " * 12


@pytest.fixture(autouse=True)
def _fresh_models():
    reset_model_cache()
    yield
    reset_model_cache()


class TestEvaluateTextLayer:
    def test_broken_vietnamese_layer_is_rejected(self):
        q = evaluate_text_layer(VI_GARBAGE)
        assert q.usable is False
        assert q.reason in {"vietnamese_unaccented", "high_perplexity"}
        assert classify_extracted_text(VI_GARBAGE) == "scanned"

    def test_real_vietnamese_passes(self):
        q = evaluate_text_layer(VI_GOOD)
        assert q.usable is True
        assert classify_extracted_text(VI_GOOD) == "text"

    def test_real_english_passes(self):
        assert classify_extracted_text(EN_GOOD) == "text"
        assert evaluate_text_layer(EN_GOOD).best_lang == "en"

    def test_real_chinese_passes(self):
        assert classify_extracted_text(ZH_GOOD) == "text"
        assert evaluate_text_layer(ZH_GOOD).best_lang == "zh"

    def test_real_russian_passes(self):
        assert classify_extracted_text(RU_GOOD) == "text"
        assert evaluate_text_layer(RU_GOOD).best_lang == "ru"

    def test_numeric_table_is_accepted(self):
        """Too little script to judge — OCR would not recover more text."""
        q = evaluate_text_layer(NUMERIC_TABLE)
        assert q.usable is True
        assert q.reason == "insufficient_script"
        assert classify_extracted_text(NUMERIC_TABLE) == "text"

    def test_short_page_is_scanned(self):
        assert classify_extracted_text("hello world", min_chars=50) == "scanned"

    def test_gate_off_falls_back_to_length(self):
        assert classify_extracted_text(VI_GARBAGE, quality_gate=False) == "text"
        assert evaluate_text_layer(VI_GARBAGE, quality_gate=False).reason == "gate_off"

    def test_structural_pua_is_rejected(self):
        soup = "Title " + "".join(chr(0xE000 + i) for i in range(40)) + " footer text here"
        q = evaluate_text_layer(soup * 3)
        assert q.usable is False
        assert q.reason == "structural_corrupt"

    def test_vietnamese_english_bilingual_passes(self):
        """Same-script mix: sentence split lets each half pick en or vi."""
        mixed = f"{VI_GOOD} {EN_GOOD}"
        assert classify_extracted_text(mixed) == "text"

    def test_english_russian_bilingual_passes(self):
        """Cross-script mix: latin and cyrillic buckets scored separately."""
        mixed = f"{EN_GOOD}\n\n{RU_GOOD}"
        assert classify_extracted_text(mixed) == "text"

    def test_chinese_russian_bilingual_passes(self):
        mixed = f"{ZH_GOOD}\n\n{RU_GOOD}"
        assert classify_extracted_text(mixed) == "text"

    def test_all_four_scripts_pass(self):
        mixed = f"{VI_GOOD}\n{EN_GOOD}\n{ZH_GOOD}\n{RU_GOOD}"
        assert classify_extracted_text(mixed) == "text"

    def test_garbage_latin_still_rejected_beside_chinese(self):
        """Broken VI Latin must not ride along on a fluent CJK bucket."""
        mixed = f"{VI_GARBAGE}\n\n{ZH_GOOD}"
        assert classify_extracted_text(mixed) == "scanned"


class TestClassifyPagesFitz:
    def _write_pdf(self, tmp_path: Path, text: str) -> str:
        doc = fitz.open()
        page = doc.new_page()
        # Word-wrap manually — insert_text is single-line; TextWriter keeps extractable Unicode.
        tw = fitz.TextWriter(page.rect)
        font = fitz.Font("helv")
        y = 50.0
        words = text.split()
        line: list[str] = []
        for w in words:
            trial = (" ".join(line + [w])).strip()
            if font.text_length(trial, fontsize=10) > 480 and line:
                tw.append((50, y), " ".join(line), font=font, fontsize=10)
                y += 14
                line = [w]
                if y > 780:
                    break
            else:
                line.append(w)
        if line and y <= 780:
            tw.append((50, y), " ".join(line), font=font, fontsize=10)
        tw.write_text(page)
        path = tmp_path / "page.pdf"
        doc.save(path)
        doc.close()
        return str(path)

    def test_garbage_latin_pdf_is_scanned(self, tmp_path, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "pdf_text_quality_gate", True)
        path = self._write_pdf(tmp_path, VI_GARBAGE)
        result = classify_pages(path, threshold=50)
        assert result[1] == "scanned"

    def test_english_pdf_is_text(self, tmp_path, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "pdf_text_quality_gate", True)
        path = self._write_pdf(tmp_path, EN_GOOD)
        result = classify_pages(path, threshold=50)
        assert result[1] == "text"

    def test_gate_off_accepts_garbage_by_length(self, tmp_path, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "pdf_text_quality_gate", False)
        path = self._write_pdf(tmp_path, VI_GARBAGE)
        result = classify_pages(path, threshold=50)
        assert result[1] == "text"
