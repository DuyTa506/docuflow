"""Text-layer extraction converts in slices instead of one opaque call.

Docling used to convert a whole PDF in a single blocking call sitting directly
on the activity's event loop. On DOC_001 (761 pages) that produced the worst
possible failure shape:

  * the loop was frozen, so `_with_heartbeat` never ran — Temporal killed the
    attempt on its 5-minute `heartbeat_timeout` while the conversion was still
    working, and the finished result was thrown away;
  * the same frozen loop could not poll the activity queue either, so the
    retry Temporal scheduled sat in the backlog forever (dispatch rate 0) —
    a deadlock the worker cannot leave on its own;
  * no page was persisted and no progress row was written until the *whole*
    book had converted, so the task stayed PENDING at 0% and read as dead.

Slicing fixes all three: the blocking work moves to a thread (the loop stays
free for heartbeats and polling), each slice persists its pages, and progress
advances per page from the first slice on.

`page_range` was verified against the installed Docling: page numbers stay
absolute, per-page markdown is byte-identical to a full-document conversion,
and the converter's pipeline cache stays at one entry so models load once.
"""

import asyncio
from unittest.mock import patch

import pytest

from services.document_service import DOCLING_PAGE_CHUNK, DocumentService


@pytest.fixture(autouse=True)
def no_page_raster():
    """Rasterising a page needs a real PDF; the slicing under test does not."""
    with patch("utils.image_utils.render_pdf_page_to_base64", return_value=""):
        yield


class FakeExtractor:
    """Stands in for DoclingLayoutExtractor, recording what it was asked for.

    `block` is how long each conversion blocks the calling thread — the real
    one blocks for minutes, which is the whole point of the bug.
    """

    def __init__(self, file_path, block: float = 0.0, fail_on=None):
        self.file_path = file_path
        self.block = block
        self.fail_on = fail_on
        self.ranges: list[tuple[int, int]] = []

    def convert(self, page_range=None):
        if self.fail_on is not None and page_range == self.fail_on:
            raise RuntimeError("docling exploded")
        self.ranges.append(page_range)
        if self.block:
            import time

            time.sleep(self.block)

    def extract_page(self, page_number):
        return []

    def page_size(self, page_number):
        return 595.0, 842.0

    def page_markdown(self, page_number):
        return f"page {page_number}"


def _run(pages, extractor, saved=None, bumped=None):
    """Drive `_extract_text_pages` with every side effect captured."""
    saved = [] if saved is None else saved
    bumped = [] if bumped is None else bumped

    def on_page_saved(page_number, **kwargs):
        saved.append(page_number)

    def on_page_done():
        bumped.append(len(bumped) + 1)

    return DocumentService()._extract_text_pages(
        file_path="/tmp/x.pdf",
        pages=pages,
        extractor=extractor,
        save_page=on_page_saved,
        on_page_done=on_page_done,
    )


class TestSlicing:
    @pytest.mark.asyncio
    async def test_conversion_is_split_into_slices(self):
        """One call over 761 pages is one opaque hour; slices are countable."""
        pages = list(range(1, 26))
        extractor = FakeExtractor("/tmp/x.pdf")

        await _run(pages, extractor)

        assert extractor.ranges == [(1, 10), (11, 20), (21, 25)]

    @pytest.mark.asyncio
    async def test_a_sparse_page_list_does_not_widen_the_slice(self):
        """Docling's cost is the span it converts, not the pages we read back.

        Taking (min, max) of the next ten pending pages looks harmless until
        the pending pages are scattered — a scanned book with occasional text
        pages, or a resume with holes. `[3, 4, 5, 40, 41]` would convert 39
        pages to use 5, and several such slices would re-convert most of the
        book more than once: worse than the single whole-book call this
        replaced. Bounding the window instead keeps every slice cheap.
        """
        extractor = FakeExtractor("/tmp/x.pdf")

        await _run([3, 4, 5, 40, 41], extractor)

        assert extractor.ranges == [(3, 5), (40, 41)]

    @pytest.mark.asyncio
    async def test_a_window_never_converts_more_than_the_chunk(self):
        pages = [1, 2, 3, 50, 100, 101, 700]
        extractor = FakeExtractor("/tmp/x.pdf")

        await _run(pages, extractor)

        assert all(hi - lo + 1 <= DOCLING_PAGE_CHUNK for lo, hi in extractor.ranges)
        assert sorted(p for lo, hi in extractor.ranges for p in pages if lo <= p <= hi) == pages

    @pytest.mark.asyncio
    async def test_nothing_pending_converts_nothing(self):
        extractor = FakeExtractor("/tmp/x.pdf")

        await _run([], extractor)

        assert extractor.ranges == []


class TestProgress:
    @pytest.mark.asyncio
    async def test_early_pages_are_saved_before_the_last_slice_converts(self):
        """The 0%-until-the-end symptom: a page reaching the DB is what flips
        the task PENDING -> RUNNING, so the first slice must persist."""
        order = []
        pages = list(range(1, 26))

        class Recorder(FakeExtractor):
            def convert(self, page_range=None):
                order.append(("convert", page_range))
                super().convert(page_range)

        extractor = Recorder("/tmp/x.pdf")

        def on_page_saved(page_number, **kwargs):
            order.append(("save", page_number))

        await DocumentService()._extract_text_pages(
            file_path="/tmp/x.pdf",
            pages=pages,
            extractor=extractor,
            save_page=on_page_saved,
            on_page_done=lambda: None,
        )

        first_save = order.index(("save", 1))
        last_convert = order.index(("convert", (21, 25)))
        assert first_save < last_convert

    @pytest.mark.asyncio
    async def test_every_page_reports_progress_once(self):
        bumped: list[int] = []

        await _run(list(range(1, 26)), FakeExtractor("/tmp/x.pdf"), bumped=bumped)

        assert len(bumped) == 25

    @pytest.mark.asyncio
    async def test_a_failed_slice_keeps_the_pages_already_stored(self):
        """Temporal retries with resume=True, which only pays off if the
        slices before the failure are already in the DB."""
        saved: list[int] = []
        extractor = FakeExtractor("/tmp/x.pdf", fail_on=(11, 20))

        with pytest.raises(RuntimeError):
            await _run(list(range(1, 26)), extractor, saved=saved)

        assert saved == list(range(1, 11))


class TestTheLoopStaysFree:
    @pytest.mark.asyncio
    async def test_heartbeat_can_run_while_docling_converts(self):
        """The regression test for the deadlock.

        `_with_heartbeat` is an asyncio task on the same loop as the activity.
        With the conversion called inline it never got a slot, Temporal timed
        the attempt out, and the retry could not be polled. A ticker standing
        in for the heartbeat must tick while conversion is in flight.
        """
        ticks = []

        async def heartbeat():
            while True:
                await asyncio.sleep(0.005)
                ticks.append(1)

        beat = asyncio.create_task(heartbeat())
        try:
            # Three slices blocking 40ms each: a loop that is free ticks
            # several times, a blocked one cannot tick at all.
            await _run(list(range(1, 26)), FakeExtractor("/tmp/x.pdf", block=0.04))
        finally:
            beat.cancel()

        assert len(ticks) >= 3


class TestSliceSize:
    def test_the_chunk_is_small_enough_to_show_movement(self):
        """761 pages / 10 = 76 progress steps. A chunk in the hundreds would
        put us back where we started."""
        assert 1 <= DOCLING_PAGE_CHUNK <= 25
