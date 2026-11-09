"""Return the GPU memory PyTorch holds in its pool back to the driver.

Docling loads three models onto the GPU during extraction (layout, TableFormer,
CodeFormula). Once it finishes they are garbage as far as Python is concerned —
`DocumentConverter` is constructed fresh each time and Docling's pipeline cache
is an instance attribute — but PyTorch's caching allocator does not return freed
blocks to the driver, so `nvidia-smi` still shows the space as taken.

Observed consequence on 2026-08-06: the worker held 4.7 GB for over a day, vLLM
OCR no longer had the 9.41 GB it needed to start, and docuflow-backend
crash-looped 447 times.
"""

import logging

logger = logging.getLogger(__name__)


def release_cached_gpu_memory() -> bool:
    """Call `torch.cuda.empty_cache()`. True when released, False when skipped.

    Absolutely silent by design: this runs at the end of every extraction,
    failed ones included, so a host without torch/CUDA or a driver having a bad
    day must never turn a finished extraction into a failed one.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
        logger.info("Returned PyTorch's pooled GPU memory to the driver")
        return True
    except Exception as exc:
        logger.warning("Could not release GPU memory (%s) — skipping", exc)
        return False
