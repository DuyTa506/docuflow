"""Thể loại tài liệu cho tổng thuật: sách hay kỷ yếu / số tạp chí.

`Mau Tong thuat Book .docx` mô tả hai chế độ khác nhau ở cả §2.1 và §2.2 —
một quyển sách được thuật theo *chương*, một kỷ yếu được thuật theo *BBKH*
(gộp thành cụm vấn đề, hoặc từng bài một nếu số lượng ít).

Hai quyết định đáng nói:

1. **Nhận diện chỉ đọc nhan đề và phần đầu tài liệu.** Quét cả 800 trang thì
   một cuốn sách nhắc tới "kỷ yếu hội nghị" ở phần tài liệu tham khảo sẽ bị
   đổi thể loại. Trang bìa mới là chỗ thể loại được tuyên bố.

2. **Nhận diện không bao giờ là tiếng nói cuối cùng.** Nó có thể sai, và một
   suy đoán sai không được phép chạy im lặng — nên kết quả luôn kèm lý do, và
   `documents.digest_doc_kind` cho phép sửa tay đè lên.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

BOOK = "book"
PROCEEDINGS = "proceedings"
DOC_KINDS = (BOOK, PROCEEDINGS)

# Bìa + trang tên sách. Đủ rộng để bắt trang bìa lót, đủ hẹp để không chạm
# vào thân tài liệu.
FRONT_MATTER_CHARS = 4000

# Cụm từ tuyên bố thể loại, không phải cụm từ *nhắc tới* thể loại. Ví dụ
# "conference" một mình bị loại: sách nào cũng có thể nói về hội nghị.
_PROCEEDINGS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"kỷ yếu", "kỷ yếu"),
    (r"tuyển tập\s+(?:các\s+)?(?:công trình|bài báo|báo cáo)", "tuyển tập công trình"),
    (r"hội (?:nghị|thảo)\s+khoa học", "hội nghị/hội thảo khoa học"),
    (r"proceedings", "proceedings"),
    (r"\bsymposium\b", "symposium"),
    (r"book of abstracts", "book of abstracts"),
    (r"conference\s+papers", "conference papers"),
    (r"сборник\s+(?:трудов|статей|материалов|научных)", "сборник трудов"),
    (r"материалы\s+(?:конференции|конгресса|семинара)", "материалы конференции"),
    (r"труды\s+конференции", "труды конференции"),
    # CJK không có khoảng trắng giữa từ nên khớp thẳng cụm. Cố ý KHÔNG nhận
    # `文集` trần (tuyển tập trước tác của một tác giả — vẫn là sách) và
    # `学术会议`/`学術大会` trần (chỉ là nhắc tới hội nghị).
    (r"论文集|論文集", "论文集 / 論文集"),
    (r"予稿集|講演集", "予稿集"),
    (r"논문집", "논문집"),
    (r"tagungsband|konferenzband|kongressband", "Tagungsband"),
    (r"actes\s+d[eu]\s+(?:colloque|congrès|la conférence)", "actes du colloque"),
    (r"\bprosiding\b|\batti\s+del\b|\banais\s+d[oe]\b", "prosiding / atti / anais"),
)

# Textbooks have editors too: "под редакцией" alone made Ru_Book a «kỷ yếu».
_EDITOR_ONLY_RE = re.compile(
    r"^\W*(?:под\s+(?:общей\s+)?ред(?:акцией|\.)|edited\s+by|editors?|eds?\.|"
    r"chủ\s+biên|biên\s+tập|主编|編)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\w{4,}")
# The work calls itself a textbook / monograph / manual: one continuous work,
# however many authors share its chapters (DOC_010: «учебного пособия … (гл. 1, 2, 3)»).
_BOOK_MARKER_RE = re.compile(
    r"учебн\w*\s+пособи|учебник|монограф|textbook|monograph|lehrbuch|"
    r"giáo\s+trình|sách\s+giáo\s+khoa|教材|教科书|专著",
    re.IGNORECASE,
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), label) for pattern, label in _PROCEEDINGS_PATTERNS
)


def normalize_doc_kind(value) -> str | None:
    """Kiểm tra giá trị đặt tay. Rỗng nghĩa là "để hệ thống tự nhận diện"."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in DOC_KINDS:
        raise ValueError(f"doc_kind phải là một trong {DOC_KINDS}, nhận được '{value}'")
    return text


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or ""))


def detect_doc_kind(title: str = "", text: str = "") -> tuple[str, str]:
    """Đoán thể loại từ nhan đề + phần đầu tài liệu.

    Trả về ``(kind, reason)``; ``reason`` rỗng nghĩa là không có tín hiệu nào
    và giá trị trả về chỉ là mặc định.
    """
    haystack = f"{_fold(title)}\n{_fold(text)[:FRONT_MATTER_CHARS]}"
    for pattern, label in _COMPILED:
        if pattern.search(haystack):
            return PROCEEDINGS, label
    return BOOK, ""


def _result(kind: str, source: str, reason: str = "", degraded: bool = False) -> dict:
    return {
        "doc_kind": kind,
        "doc_kind_source": source,
        "doc_kind_reason": reason,
        "doc_kind_degraded": degraded,
    }


def resolve_doc_kind(explicit, title: str = "", text: str = "") -> dict:
    """Chốt thể loại bằng bằng chứng tất định, không gọi LLM.

    ``doc_kind_source``: ``explicit`` (người dùng đặt) | ``detected`` (từ vựng
    trúng) | ``default`` (không có tín hiệu, coi là sách).
    """
    chosen = normalize_doc_kind(explicit)
    if chosen:
        return _result(chosen, "explicit")

    kind, reason = detect_doc_kind(title=title, text=text)
    return _result(kind, "detected" if reason else "default", reason)


async def resolve_doc_kind_async(llm, explicit, title: str = "", text: str = "") -> dict:
    """Như trên, nhưng hỏi LLM khi từ vựng không bắt được gì.

    Bảng từ vựng chỉ phủ những ngôn ngữ đã liệt kê; một kỷ yếu tiếng Thổ, Ba
    Lan hay Indonesia rơi hết về ``book``. LLM lấp phần đó — nhưng **chỉ khi
    không còn bằng chứng nào khác**: khi bìa đã ghi "KỶ YẾU" thì một câu trả
    lời sai không được phép lật ngược nó, và một lượt gọi mạng không được
    thêm vào đường chạy vốn đã có câu trả lời chắc chắn.

    Fail-open: lỗi mạng hoặc phản hồi không parse được → ``book`` kèm
    ``doc_kind_degraded=True``, để báo cáo chất lượng nói ra chứ không giấu.
    """
    resolved = resolve_doc_kind(explicit, title=title, text=text)
    if resolved["doc_kind_source"] != "default" or llm is None:
        return resolved

    prompt = (
        "You are a library cataloguer.\n\n"
        "TASK: Decide whether the document below is a single authored work or a "
        "collection of independent papers.\n"
        '- "book": a monograph, textbook or manual — chapters of one continuous work.\n'
        '- "proceedings": conference/symposium proceedings, a journal issue, or any '
        "volume made of separate papers by different authors.\n\n"
        "RULES:\n"
        "- Judge ONLY from the text below. It may be in any language.\n"
        "- Proceedings need evidence of SEPARATE papers: a conference/symposium name, "
        "or a contents list where each entry has its own authors. An editor "
        "('edited by', 'под редакцией', 'chủ biên') alone does NOT — textbooks have "
        "editors too.\n"
        "- A textbook or monograph whose chapters were written by different authors "
        "is still a book.\n"
        "- The evidence must be quoted verbatim from the text.\n"
        '- If the text does not say, answer "book".\n\n'
        'OUTPUT: JSON only — {"kind": "book"|"proceedings", "evidence": "<short quote '
        'or phrase from the text, in its original language>"}\n\n'
        f"TITLE: {_fold(title)}\n\n"
        f"FRONT MATTER:\n{_fold(text)[:FRONT_MATTER_CHARS]}\n\n"
        "JSON:"
    )

    try:
        raw = await llm.chat_completion(prompt, max_tokens=200)
        parsed = json.loads(_JSON_OBJECT_RE.search(str(raw)).group(0))
        kind = normalize_doc_kind(parsed.get("kind"))
    except Exception as exc:  # network, JSON, shape — all the same outcome here
        logger.warning("Không phân loại được thể loại tài liệu bằng LLM (%s) — coi là sách", exc)
        return _result(BOOK, "default", degraded=True)

    if not kind:
        logger.warning("LLM trả về thể loại rỗng — coi là sách")
        return _result(BOOK, "default", degraded=True)

    evidence = str(parsed.get("evidence") or "").strip()[:200]
    if kind == PROCEEDINGS and not _supports_proceedings(evidence, title, text):
        logger.info(
            "LLM đoán «kỷ yếu» nhưng bằng chứng không đứng vững (%r) — coi là sách", evidence
        )
        return _result(BOOK, "llm", evidence)
    return _result(kind, "llm", evidence)


def _supports_proceedings(evidence: str, title: str, text: str) -> bool:
    """The quote must come from the front matter and say more than "edited by"."""
    if not evidence or _EDITOR_ONLY_RE.match(evidence):
        return False
    if _BOOK_MARKER_RE.search(_fold(evidence)) or _BOOK_MARKER_RE.search(
        f"{_fold(title)}\n{_fold(text)[:FRONT_MATTER_CHARS]}"
    ):
        return False
    haystack = f"{_fold(title)}\n{_fold(text)[:FRONT_MATTER_CHARS]}".casefold()
    return any(word.casefold() in haystack for word in _WORD_RE.findall(_fold(evidence)))
