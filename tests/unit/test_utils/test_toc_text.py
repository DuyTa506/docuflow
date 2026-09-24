"""TOC text detection / batching for translation."""

from utils.toc_text import looks_like_toc_text, split_toc_line_batches


def test_looks_like_toc_russian_leaders():
    text = "\n".join(
        [
            "Введение ..... 1",
            "Глава 1. Изображения и помехи ..... 5",
            "1.1. Объекты реального мира ..... 5",
            "1.2. Двумерная растровая модель ..... 6",
            "1.3. Векторная модель ..... 8",
            "Глава 2. Цвет ..... 40",
            "2.1. Колориметрия ..... 41",
            "2.2. XYZ ..... 58",
        ]
    )
    assert looks_like_toc_text(text)


def test_looks_like_toc_rejects_normal_paragraph():
    text = (
        "Digital image processing is an important field of study. "
        "This chapter introduces noise models and basic filtering.\n"
        "We then discuss color spaces and human vision briefly."
    )
    assert not looks_like_toc_text(text)


def test_split_toc_line_batches_preserves_order():
    lines = [f"Section {i} ..... {i}" for i in range(1, 25)]
    text = "\n".join(lines)
    batches = split_toc_line_batches(text, lines_per_batch=10)
    assert len(batches) == 3
    assert batches[0].startswith("Section 1")
    assert "Section 10" in batches[0]
    assert batches[1].startswith("Section 11")
    assert batches[2].endswith("Section 24 ..... 24")
    assert "\n".join(batches).count("\n") == 23
