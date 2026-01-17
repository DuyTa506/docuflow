# Legacy Code

This folder contains deprecated code that is no longer used in the main pipeline.

## Deprecated Components

### PDF Processor (`processors/pdf_processor.py`)
- **Status**: DEPRECATED
- **Reason**: We now use Deepseek OCR to convert PDF to Markdown files directly
- **Replacement**: Use `MarkdownProcessor` with OCR-generated `.md` files
- **Date Deprecated**: 2026-01-17

### PDF-Specific Core Components (`core/`)
All TOC (Table of Contents) related components have been moved here:

- **TOCDetector** (`toc_detector.py`) - Detects TOC pages in PDF
- **TOCExtractor** (`toc_extractor.py`) - Extracts TOC content
- **TOCTransformer** (`toc_transformer.py`) - Transforms TOC to JSON
- **PageMapper** (`page_mapper.py`) - Maps logical to physical page numbers
- **TOCVerifier** (`verifier.py`) - Verifies TOC accuracy

**Status**: DEPRECATED  
**Reason**: These are only used by PDFProcessor, which is deprecated  
**Date Deprecated**: 2026-01-17

> **Note**: `TreeBuilder` remains in the main `core/` as it's still a base component.

## Usage Flow

### Old Flow (Deprecated):
```
PDF → pdf_processor → TOC Detection → TOC Extraction → Tree Building
```

### New Flow (Current):
```
PDF → Deepseek OCR → .md file → MarkdownProcessor → Tree Building
```

## Benefits of New Approach:
1. ✅ Simpler pipeline
2. ✅ Better OCR quality from Deepseek
3. ✅ Direct markdown structure parsing
4. ✅ No need for complex TOC detection
5. ✅ Faster processing

## Note
These files are kept for reference only. Do not use them in production code.
