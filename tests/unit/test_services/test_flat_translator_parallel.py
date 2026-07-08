"""FlatTranslator should translate chunks concurrently, preserving order."""
import asyncio

import pytest
from unittest.mock import MagicMock, patch

from services.translators.flat_translator import FlatTranslator


class TestFlatTranslatorParallel:
    @pytest.mark.asyncio
    async def test_chunks_translate_concurrently_and_in_order(self):
        translator = MagicMock()
        translator.chunk_size = 100
        translator.chunk_text = MagicMock(return_value=["c1", "c2", "c3"])

        delays = {"c1": 0.03, "c2": 0.01, "c3": 0.01}
        peak = [0]
        in_flight = []

        async def fake_translate_text(chunk: str) -> str:
            in_flight.append(chunk)
            peak[0] = max(peak[0], len(in_flight))
            await asyncio.sleep(delays[chunk])
            in_flight.remove(chunk)
            return f"VI:{chunk}"

        translator.translate_text = fake_translate_text

        with patch("services.translators.flat_translator.settings") as mock_settings:
            mock_settings.translation_parallelism = 3
            result = await FlatTranslator(translator).translate_text("irrelevant")

        assert peak[0] >= 2, "chunks should overlap, not translate strictly sequentially"
        assert result["translated_content"] == "VI:c1\n\nVI:c2\n\nVI:c3"
        assert result["translation_mode"] == "flat"
