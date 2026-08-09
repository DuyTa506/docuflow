"""Docling holds the GPU after extraction, and that is what strangled vLLM OCR.

Incident of 2026-08-06: a worker running since the previous day still held 4.7 GB
of GPU for Docling's three models (layout, TableFormer, CodeFormula) long after
extraction had finished. Together with llama.cpp's 17.3 GB, vLLM OCR no longer
had the 9.41 GB it needed to start — docuflow-backend crash-looped 447 times.

The models are garbage *as far as Python is concerned*: `DocumentConverter` is
constructed fresh each time and Docling's pipeline cache is an instance
attribute. What holds the space is **PyTorch's caching allocator**, which does
not return freed blocks to the driver. Only `torch.cuda.empty_cache()` does.

This function runs at the end of every extraction, failed ones included, so it
has to be absolutely silent: a host without torch, without CUDA, or with a driver
having a bad day must never break an extraction that already succeeded.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from utils.gpu_memory import release_cached_gpu_memory


def _fake_torch(available=True, empty_cache=None, collect=None):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        empty_cache=empty_cache or MagicMock(),
        ipc_collect=collect or MagicMock(),
    )
    return SimpleNamespace(cuda=cuda)


class TestReleases:
    def test_empties_the_cache_when_cuda_is_present(self, monkeypatch):
        empty = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(empty_cache=empty))

        assert release_cached_gpu_memory() is True
        empty.assert_called_once()

    def test_skips_when_cuda_is_unavailable(self, monkeypatch):
        empty = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, empty_cache=empty))

        assert release_cached_gpu_memory() is False
        empty.assert_not_called()


class TestNeverBreaksTheCaller:
    """Called at the end of extraction — a failure here must not take the run down."""

    def test_missing_torch_is_not_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        assert release_cached_gpu_memory() is False

    def test_driver_failure_is_swallowed_and_logged(self, monkeypatch, caplog):
        boom = MagicMock(side_effect=RuntimeError("CUDA driver shutting down"))
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(empty_cache=boom))

        import logging

        with caplog.at_level(logging.WARNING, logger="utils.gpu_memory"):
            assert release_cached_gpu_memory() is False

        assert "CUDA driver shutting down" in caplog.text, "nuốt lỗi thì phải để lại dấu vết"
