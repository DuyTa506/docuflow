#!/usr/bin/env python3
"""Train character 5-gram LMs for PDF text-layer quality gating.

Data: Wikipedia only (MediaWiki API). No embedded seed corpus.

Produces:
  services/extractors/text_layer_lm/{en,zh,ru,vi}.npz
  services/extractors/text_layer_lm/thresholds.json

Requires: pip install requests

Usage:
  python scripts/train_text_layer_lm.py
  python scripts/train_text_layer_lm.py --chars-per-lang 300000 --max-pages 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.extractors.text_layer_lm.char_lm import (  # noqa: E402
    model_dir,
    train_char_lm,
)

LANGS = ("en", "zh", "ru", "vi")
# ~0.2–0.5 MB text/lang is enough for char-5gram fluency; not GB-scale.
_MIN_CHARS_PER_LANG = 80_000
_DEFAULT_CHARS = 300_000
_DEFAULT_MAX_PAGES = 300

# Long, stable articles (fill first — random zh pages are often stubs).
_SEED_TITLES: dict[str, list[str]] = {
    "en": [
        "Computer",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "History",
        "Geography",
        "China",
        "Russia",
        "Vietnam",
        "United States",
        "World War II",
        "Internet",
        "Machine learning",
        "Operating system",
        "Database",
        "Algorithm",
        "Artificial intelligence",
        "Economics",
        "Law",
        "Urban planning",
        "Agriculture",
        "Medicine",
        "Engineering",
        "Literature",
        "Philosophy",
        "Astronomy",
        "Climate change",
        "Electricity",
        "Genetics",
        "Statistics",
        "Linear algebra",
        "Probability",
        "Software engineering",
        "Computer network",
        "Cryptography",
        "Natural language processing",
        "Optical character recognition",
        "Document",
        "Public administration",
    ],
    "zh": [
        "计算机",
        "数学",
        "物理学",
        "化学",
        "生物学",
        "历史",
        "地理",
        "中国",
        "俄罗斯",
        "越南",
        "美国",
        "第二次世界大战",
        "互联网",
        "机器学习",
        "操作系统",
        "数据库",
        "算法",
        "人工智能",
        "经济学",
        "法律",
        "城市规划",
        "农业",
        "医学",
        "工程学",
        "文学",
        "哲学",
        "天文学",
        "气候变化",
        "电",
        "遗传学",
        "统计学",
        "线性代数",
        "概率论",
        "软件工程",
        "计算机网络",
        "密码学",
        "自然语言处理",
        "光学字符识别",
        "文档",
        "公共行政",
        "汉字",
        "中文",
        "北京",
        "上海",
        "清朝",
        "明朝",
        "宋朝",
        "唐朝",
        "中国共产党",
        "中华人民共和国",
    ],
    "ru": [
        "Компьютер",
        "Математика",
        "Физика",
        "Химия",
        "Биология",
        "История",
        "География",
        "Китай",
        "Россия",
        "Вьетнам",
        "Соединённые_Штаты_Америки",
        "Вторая_мировая_война",
        "Интернет",
        "Машинное_обучение",
        "Операционная_система",
        "База_данных",
        "Алгоритм",
        "Искусственный_интеллект",
        "Экономика",
        "Право",
        "Градостроительство",
        "Сельское_хозяйство",
        "Медицина",
        "Инженерия",
        "Литература",
        "Философия",
        "Астрономия",
        "Изменение_климата",
        "Электричество",
        "Генетика",
        "Статистика",
        "Линейная_алгебра",
        "Теория_вероятностей",
        "Программная_инженерия",
        "Компьютерная_сеть",
        "Криптография",
        "Обработка_естественного_языка",
        "Оптическое_распознавание_символов",
        "Документ",
        "Государственное_управление",
    ],
    "vi": [
        "Máy_tính",
        "Toán_học",
        "Vật_lý",
        "Hóa_học",
        "Sinh_học",
        "Lịch_sử",
        "Địa_lý",
        "Trung_Quốc",
        "Nga",
        "Việt_Nam",
        "Hoa_Kỳ",
        "Chiến_tranh_thế_giới_thứ_hai",
        "Internet",
        "Học_máy",
        "Hệ_điều_hành",
        "Cơ_sở_dữ_liệu",
        "Thuật_toán",
        "Trí_tuệ_nhân_tạo",
        "Kinh_tế_học",
        "Luật",
        "Quy_hoạch_đô_thị",
        "Nông_nghiệp",
        "Y_học",
        "Kỹ_thuật",
        "Văn_học",
        "Triết_học",
        "Thiên_văn_học",
        "Biến_đổi_khí_hậu",
        "Điện",
        "Di_truyền_học",
        "Thống_kê",
        "Đại_số_tuyến_tính",
        "Xác_suất",
        "Kỹ_nghệ_phần_mềm",
        "Mạng_máy_tính",
        "Mật_mã_học",
        "Xử_lý_ngôn_ngữ_tự_nhiên",
        "Nhận_dạng_ký_tự_quang_học",
        "Tài_liệu",
        "Hành_chính_công",
        "Hà_Nội",
        "Thành_phố_Hồ_Chí_Minh",
        "Đất_đai",
        "Xây_dựng",
        "Chính_phủ_Việt_Nam",
    ],
}


def _has_vietnamese_diacritics(text: str) -> bool:
    return any("\u1ea0" <= c <= "\u1ef9" or c in "ăâđêôơưĂÂĐÊÔƠƯ" for c in text)


def _get_json(session, url: str, params: dict, *, lang: str) -> dict | None:
    """GET with retry/backoff on 429 / transient errors."""
    delay = 2.0
    for attempt in range(8):
        try:
            r = session.get(url, params=params, timeout=45)
            if r.status_code == 429:
                wait = delay * (attempt + 1)
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, float(retry_after))
                print(f"  {lang}: 429, sleep {wait:.0f}s (attempt {attempt + 1})", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt >= 7:
                print(f"  wiki fetch failed for {lang}: {exc}", file=sys.stderr)
                return None
            print(f"  {lang}: {exc}; retry in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 1.8, 40.0)
    return None


def _accept_extract(lang: str, extract: str) -> bool:
    if len(extract) < 200:
        return False
    if lang == "vi" and not _has_vietnamese_diacritics(extract):
        return False
    letters = sum(1 for c in extract if c.isalpha() or "\u4e00" <= c <= "\u9fff")
    return letters >= 80


def _fetch_titles(lang: str, titles: list[str], session, *, budget: int) -> list[str]:
    """Fetch full plain-text extracts for known article titles."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    out: list[str] = []
    used = 0
    for i in range(0, len(titles), 10):
        if used >= budget:
            break
        batch = titles[i : i + 10]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": "max",
            "redirects": 1,
        }
        data = _get_json(session, api, params, lang=lang)
        time.sleep(0.8)
        if not data:
            continue
        for page in (data.get("query") or {}).get("pages", {}).values():
            if page.get("missing") is not None:
                continue
            extract = (page.get("extract") or "").strip()
            if not _accept_extract(lang, extract):
                continue
            take = extract[: max(0, budget - used)]
            out.append(take)
            used += len(take)
            if used >= budget:
                break
    return out


def _fetch_wikipedia_until(
    lang: str,
    *,
    target_chars: int,
    max_pages: int,
    session,
) -> list[str]:
    """Curated long articles first, then random pages until target_chars."""
    print(f"  {lang}: fetching curated titles...")
    out = _fetch_titles(lang, _SEED_TITLES.get(lang, []), session, budget=target_chars)
    used = sum(len(c) for c in out)
    print(f"  {lang}: {used:,} chars from curated titles")
    if used >= target_chars:
        return out

    api = f"https://{lang}.wikipedia.org/w/api.php"
    fetched = 0
    empty_streak = 0
    while used < target_chars and fetched < max_pages:
        batch = min(10, max_pages - fetched)
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": batch,
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 0,
            "exlimit": batch,
        }
        data = _get_json(session, api, params, lang=lang)
        if data is None:
            break

        pages_map = (data.get("query") or {}).get("pages") or {}
        got = 0
        for page in pages_map.values():
            extract = (page.get("extract") or "").strip()
            if not _accept_extract(lang, extract):
                continue
            take = extract[: max(0, target_chars - used)]
            if not take:
                break
            out.append(take)
            used += len(take)
            got += 1
            if used >= target_chars:
                break

        fetched += batch
        if got == 0:
            empty_streak += 1
            if empty_streak >= 12:
                print(f"  stopping {lang}: too many empty random batches", file=sys.stderr)
                break
        else:
            empty_streak = 0
        time.sleep(0.8)
        if fetched % 50 == 0 or used >= target_chars:
            print(f"  {lang}: {used:,} chars (random ~{fetched} requests)...")

    return out


def _calibration_positives() -> dict[str, str]:
    """Tiny held-out snippets for threshold fitting only (not training)."""
    return {
        "en": (
            "This paper presents a scalable architecture for document understanding "
            "that combines layout analysis with neural optical character recognition. "
            "Experimental results on multilingual corpora show consistent gains over "
            "prior baselines in both precision and recall."
        ),
        "zh": (
            "本文提出一种面向文档理解的可扩展架构，将版面分析与神经光学字符识别相结合。"
            "在多语种语料上的实验结果表明，该方法在准确率与召回率方面均优于已有基线。"
            "城市规划与行政管理文件同样可用于评估跨域泛化能力。"
        ),
        "ru": (
            "В статье представлена масштабируемая архитектура понимания документов, "
            "сочетающая анализ макета с нейронным оптическим распознаванием символов. "
            "Эксперименты на многоязычных корпусах показывают устойчивый рост точности "
            "и полноты по сравнению с существующими базовыми методами."
        ),
        "vi": (
            "Thanh tra Chính phủ ban hành thông báo kết luận thanh tra về công tác "
            "quản lý, sử dụng đất đai theo tinh thần Nghị quyết của Chính phủ; "
            "công tác quy hoạch và thực hiện quy hoạch xây dựng của Ủy ban nhân dân tỉnh. "
            "Kiến nghị giao địa phương tổ chức khắc phục hạn chế, thiếu sót đã nêu."
        ),
    }


def _calibration_negatives() -> list[tuple[str, str]]:
    garbage = (
        "THANH TRA CHNH PHU CONG HOA XA HQI CHU NGHIA VIET NAM Mic lap - Tir do - Hanh pluic "
        "Sa:02.14/TB-TTCP Ha A T ngetycl6thang 02 nam 2024 THONG BAO KET LUAN THANH TRA "
        "Cong tfic quan 157, sir dung dat dai theo tinh than Nghi quytt 73/NQ-CP va 116/NQ-CP "
        "cua Chinh phii; cong tfic quy hoach va thuc hien quy hoach xay dun cua UBND tinh Hung Yen "
        "Ngdy 28/12/2023, Tong Thanh tra Chinh phii ban hanh Ket Juan thanh tra so 3136/KL-TTCP "
        "ye cong tic quan VT, sir dung cat dai theo tinh thin Nghi guy& 73/NQ-CP va 116/NQ-CP"
    )
    consonants = "CHNH PHU HQI NGHIA VIET NAM TIR DO HANH PLUIC QHXD CNQSD " * 20
    random_latin = (
        "xqz vbb nngg ffjj wwww ykpk mzrt plxq bbfg nnmh "
        "qwrk vzxp mnnb jjff hhgg kkll ppqq " * 25
    )
    return [
        ("vi_garbage", garbage),
        ("consonant_salad", consonants),
        ("random_latin", random_latin),
    ]


def _fit_thresholds(models: dict, margin: float = 2.0) -> dict:
    from services.extractors.text_layer_quality import eligible_langs

    positives = _calibration_positives()
    negatives = _calibration_negatives()
    thresholds: dict[str, float] = {}
    report: dict = {"positives": {}, "negatives": {}, "thresholds": thresholds}

    for lang, text in positives.items():
        ppl = models[lang].perplexity(text)
        report["positives"][lang] = ppl
        thresholds[lang] = max(ppl * margin, ppl + 8.0)

    for name, text in negatives:
        elig = eligible_langs(text) or list(LANGS)
        scores = {lang: models[lang].perplexity(text) for lang in elig}
        report["negatives"][name] = {
            "best": min(scores.values()) if scores else float("inf"),
            "by_lang": scores,
            "eligible": elig,
        }

    # Only tighten using script-eligible languages (Latin junk must not fight zh).
    for _ in range(8):
        violated = False
        for name, text in negatives:
            for lang in eligible_langs(text):
                ppl = models[lang].perplexity(text)
                if ppl < thresholds[lang]:
                    violated = True
                    pos = report["positives"][lang]
                    mid = (pos + ppl) / 2.0
                    thresholds[lang] = max(pos * 1.15, min(thresholds[lang], mid))
                    print(
                        f"  tighten {lang} for {name}: neg_ppl={ppl:.2f} -> T={thresholds[lang]:.2f}",
                        file=sys.stderr,
                    )
        if not violated:
            break
    else:
        print("WARNING: could not separate all negatives from positives", file=sys.stderr)

    for lang, text in positives.items():
        ppl = models[lang].perplexity(text)
        if ppl >= thresholds[lang]:
            thresholds[lang] = ppl * 1.2
            print(
                f"  widen {lang} for positive: ppl={ppl:.2f} -> T={thresholds[lang]:.2f}",
                file=sys.stderr,
            )

    report["global_threshold"] = float(min(thresholds.values()))
    report["thresholds"] = thresholds
    return report


def _finite(x: float) -> float | None:
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return float(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chars-per-lang", type=int, default=_DEFAULT_CHARS)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=_DEFAULT_MAX_PAGES,
        help="Max Wikipedia page requests per language",
    )
    parser.add_argument("--out-dir", type=Path, default=model_dir())
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        print("pip install requests  # required to download Wikipedia", file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers["User-Agent"] = (
        "DocuFlow-TextLayerLM/1.0 (research training; contact: docuflow-local)"
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for lang in LANGS:
        target = max(args.chars_per_lang, _MIN_CHARS_PER_LANG)
        print(f"Downloading Wikipedia for {lang} (target {target:,} chars)...")
        chunks = _fetch_wikipedia_until(
            lang,
            target_chars=target,
            max_pages=args.max_pages,
            session=session,
        )
        used = sum(len(c) for c in chunks)
        if used < _MIN_CHARS_PER_LANG:
            print(
                f"ERROR: only {used:,} chars for {lang} (need >= {_MIN_CHARS_PER_LANG:,}). "
                f"Increase --max-pages or retry later (Wikimedia rate limits).",
                file=sys.stderr,
            )
            return 1
        print(f"Training {lang} on {used:,} characters ({len(chunks)} extracts)...")
        model = train_char_lm(chunks)
        path = out_dir / f"{lang}.npz"
        model.save(path)
        print(f"  wrote {path.name} ({path.stat().st_size / 1024:.1f} KiB)")
        models[lang] = model
        time.sleep(2.0)  # cool-down between language editions

    print("Calibrating thresholds...")
    report = _fit_thresholds(models)
    thresholds_path = out_dir / "thresholds.json"
    payload = {
        "languages": list(LANGS),
        "order": 5,
        "source": "wikipedia",
        "thresholds": report["thresholds"],
        "global_threshold": report["global_threshold"],
        "calibration": {
            "positives": report["positives"],
            "negatives": {
                k: {
                    "best": _finite(v["best"]),
                    "by_lang": {lk: _finite(lv) for lk, lv in v["by_lang"].items()},
                }
                for k, v in report["negatives"].items()
            },
        },
    }
    thresholds_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {thresholds_path}")
    print(json.dumps(payload["thresholds"], indent=2))

    for lang, text in _calibration_positives().items():
        print(
            f"  pos {lang}: ppl={models[lang].perplexity(text):.2f} T={report['thresholds'][lang]:.2f}"
        )
    for name, text in _calibration_negatives():
        scores = {l: models[l].perplexity(text) for l in LANGS}
        print(f"  neg {name}: best={min(scores.values()):.2f} {scores}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
