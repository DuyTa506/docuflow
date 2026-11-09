"""Character 5-gram LM with stupid backoff — trainable, numpy-serializable.

Used at PDF classify time to reject unreadable text layers (broken ToUnicode /
custom-font encoding) while accepting fluent en/zh/ru/vi. Character n-grams
avoid word tokenization (Chinese has no spaces; Vietnamese tones live in chars).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ORDER = 5
UNK = "<unk>"
# Characters rarer than this become UNK during training.
_MIN_CHAR_COUNT = 3
# N-grams with count below this are pruned (except unigrams).
_MIN_NGRAM_COUNT = 2
# Floor for backoff / unseen events.
_LOG_EPS = -20.0


def model_dir() -> Path:
    return Path(__file__).resolve().parent


def _normalize(text: str) -> str:
    """Lowercase Latin/Cyrillic; collapse whitespace; keep CJK as-is."""
    out: list[str] = []
    prev_space = False
    for ch in text.replace("\r", "\n"):
        if ch.isalpha() and ("A" <= ch <= "Z" or "А" <= ch <= "Я" or "Ё" == ch):
            ch = ch.lower()
        if ch in "\t\n\r\f\v" or ch == " ":
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
            continue
        prev_space = False
        out.append(ch)
    return "".join(out).strip()


class CharNgramLM:
    """Stupid-backoff character n-gram model (order <= 5)."""

    def __init__(
        self,
        *,
        order: int = ORDER,
        char_to_id: dict[str, int],
        # log P(c | context) keyed by packed context+token; flat parallel arrays
        contexts: np.ndarray,
        tokens: np.ndarray,
        log_probs: np.ndarray,
        unigram_log: np.ndarray,
        backoff: float = 0.4,
    ):
        self.order = order
        self.char_to_id = char_to_id
        self.id_to_char = {i: c for c, i in char_to_id.items()}
        self.unk_id = char_to_id[UNK]
        self.backoff = backoff
        self.log_backoff = math.log(backoff)
        self.unigram_log = unigram_log
        # Lookup: (context_key_tuple encoded as bytes) -> log_prob is too slow;
        # use a dict of int tuples -> float built once.
        self._table: dict[tuple[int, ...], float] = {}
        for ctx, tok, lp in zip(contexts.tolist(), tokens.tolist(), log_probs.tolist()):
            # ctx is a list of length order-1 with -1 padding on the left for short n
            key = tuple(int(x) for x in ctx if x >= 0) + (int(tok),)
            self._table[key] = float(lp)

    def encode(self, text: str) -> list[int]:
        norm = _normalize(text)
        return [self.char_to_id.get(ch, self.unk_id) for ch in norm]

    def log_prob(self, ids: list[int]) -> float:
        if not ids:
            return 0.0
        total = 0.0
        for i, tok in enumerate(ids):
            total += self._cond_log_prob(ids, i, tok)
        return total

    def _cond_log_prob(self, ids: list[int], i: int, tok: int) -> float:
        # Try longest context first (stupid backoff).
        max_ctx = min(self.order - 1, i)
        for ctx_len in range(max_ctx, 0, -1):
            key = tuple(ids[i - ctx_len : i]) + (tok,)
            lp = self._table.get(key)
            if lp is not None:
                # Back off penalty for each length we skipped below full order-1
                # is already absorbed by using observed conditional; for unseen
                # higher orders we multiply backoff when falling through.
                skipped = max_ctx - ctx_len
                return lp + skipped * self.log_backoff
            # Fall through with backoff factor for this missing order.
        # Unigram
        skipped = max_ctx  # all higher contexts missed
        uni = float(self.unigram_log[tok]) if tok < len(self.unigram_log) else _LOG_EPS
        return uni + skipped * self.log_backoff

    def unk_ratio(self, text: str) -> float:
        ids = self.encode(text)
        if not ids:
            return 1.0
        return sum(1 for i in ids if i == self.unk_id) / len(ids)

    def perplexity(self, text: str) -> float:
        ids = self.encode(text)
        if len(ids) < 8:
            return float("inf")
        # Private-use / glyph soup collapses to UNK and can look "fluent" under
        # a heavy UNK unigram; treat that as infinitely surprising instead.
        if sum(1 for i in ids if i == self.unk_id) / len(ids) > 0.25:
            return float("inf")
        lp = self.log_prob(ids)
        return math.exp(-lp / len(ids))

    def save(self, path: Path) -> None:
        path = Path(path)
        # Rebuild parallel arrays from table for compact storage.
        rows = sorted(self._table.items(), key=lambda kv: (len(kv[0]), kv[0]))
        max_ctx = self.order - 1
        contexts = np.full((len(rows), max_ctx), -1, dtype=np.int16)
        tokens = np.empty(len(rows), dtype=np.int16)
        log_probs = np.empty(len(rows), dtype=np.float32)
        for i, (key, lp) in enumerate(rows):
            *ctx, tok = key
            start = max_ctx - len(ctx)
            for j, c in enumerate(ctx):
                contexts[i, start + j] = c
            tokens[i] = tok
            log_probs[i] = lp
        # char vocab as parallel arrays
        chars = sorted(self.char_to_id.items(), key=lambda x: x[1])
        char_ids = np.array([i for _, i in chars], dtype=np.int16)
        # Store chars as UTF-8 bytes with offsets
        encoded = [c.encode("utf-8") for c, _ in chars]
        char_bytes = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        offsets = np.zeros(len(encoded) + 1, dtype=np.int32)
        for i, b in enumerate(encoded):
            offsets[i + 1] = offsets[i] + len(b)
        np.savez_compressed(
            path,
            order=np.int32(self.order),
            backoff=np.float32(self.backoff),
            contexts=contexts,
            tokens=tokens,
            log_probs=log_probs,
            unigram_log=self.unigram_log.astype(np.float32),
            char_ids=char_ids,
            char_bytes=char_bytes,
            char_offsets=offsets,
        )

    @classmethod
    def load(cls, path: Path) -> "CharNgramLM":
        data = np.load(path, allow_pickle=False)
        order = int(data["order"])
        backoff = float(data["backoff"])
        offsets = data["char_offsets"]
        char_bytes = data["char_bytes"].tobytes()
        char_ids = data["char_ids"]
        char_to_id: dict[str, int] = {}
        for i in range(len(char_ids)):
            raw = char_bytes[int(offsets[i]) : int(offsets[i + 1])]
            char_to_id[raw.decode("utf-8")] = int(char_ids[i])
        if UNK not in char_to_id:
            raise ValueError(f"model missing {UNK}: {path}")
        return cls(
            order=order,
            char_to_id=char_to_id,
            contexts=data["contexts"],
            tokens=data["tokens"],
            log_probs=data["log_probs"],
            unigram_log=data["unigram_log"],
            backoff=backoff,
        )


def train_char_lm(texts: Iterable[str], *, order: int = ORDER) -> CharNgramLM:
    """Train a character n-gram LM from an iterable of documents."""
    char_counts: Counter[str] = Counter()
    docs: list[str] = []
    for raw in texts:
        norm = _normalize(raw)
        if len(norm) < 20:
            continue
        docs.append(norm)
        char_counts.update(norm)

    # Vocab: frequent chars + UNK
    kept = {ch for ch, n in char_counts.items() if n >= _MIN_CHAR_COUNT}
    char_to_id: dict[str, int] = {UNK: 0}
    for ch in sorted(kept):
        if ch == UNK:
            continue
        char_to_id[ch] = len(char_to_id)

    def enc(s: str) -> list[int]:
        return [char_to_id.get(ch, 0) for ch in s]

    ngram_counts: list[Counter[tuple[int, ...]]] = [Counter() for _ in range(order)]
    for doc in docs:
        ids = enc(doc)
        for i, tok in enumerate(ids):
            for n in range(1, order + 1):
                if i + 1 < n:
                    break
                gram = tuple(ids[i + 1 - n : i + 1])
                ngram_counts[n - 1][gram] += 1

    # Prune rare higher-order n-grams
    for n in range(2, order + 1):
        ngram_counts[n - 1] = Counter(
            {g: c for g, c in ngram_counts[n - 1].items() if c >= _MIN_NGRAM_COUNT}
        )

    # Unigram log probs (add-one smoothed over vocab size)
    uni = ngram_counts[0]
    vocab_size = len(char_to_id)
    total_uni = sum(uni.values()) + vocab_size  # add-one
    unigram_log = np.full(vocab_size, _LOG_EPS, dtype=np.float32)
    for (tok,), c in uni.items():
        unigram_log[tok] = math.log((c + 1) / total_uni)
    # Unseen ids already at _LOG_EPS; also give UNK a mass from discarded chars
    unk_extra = sum(n for ch, n in char_counts.items() if ch not in kept)
    unigram_log[0] = math.log((uni.get((0,), 0) + 1 + unk_extra) / (total_uni + unk_extra))

    # Conditional log P(last | prefix) for n >= 2
    table: dict[tuple[int, ...], float] = {}
    for n in range(2, order + 1):
        # Denominators: counts of prefixes
        prefix_totals: Counter[tuple[int, ...]] = Counter()
        for gram, c in ngram_counts[n - 1].items():
            prefix_totals[gram[:-1]] += c
        for gram, c in ngram_counts[n - 1].items():
            denom = prefix_totals[gram[:-1]]
            if denom <= 0:
                continue
            table[gram] = math.log(c / denom)

    max_ctx = order - 1
    contexts = np.full((len(table), max_ctx), -1, dtype=np.int16)
    tokens = np.empty(len(table), dtype=np.int16)
    log_probs = np.empty(len(table), dtype=np.float32)
    for i, (key, lp) in enumerate(table.items()):
        *ctx, tok = key
        start = max_ctx - len(ctx)
        for j, c in enumerate(ctx):
            contexts[i, start + j] = c
        tokens[i] = tok
        log_probs[i] = lp

    return CharNgramLM(
        order=order,
        char_to_id=char_to_id,
        contexts=contexts,
        tokens=tokens,
        log_probs=log_probs,
        unigram_log=unigram_log,
    )


def load_model(lang: str, directory: Optional[Path] = None) -> CharNgramLM:
    directory = directory or model_dir()
    path = directory / f"{lang}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing text-layer LM for {lang}: {path}")
    return CharNgramLM.load(path)


def load_thresholds(directory: Optional[Path] = None) -> dict:
    directory = directory or model_dir()
    path = directory / "thresholds.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
