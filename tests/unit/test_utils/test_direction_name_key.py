"""`research_directions` is a shared catalog that grew to 290 rows of near-copies.

`direction_name` is UNIQUE and looked up by exact string, so every rephrasing
the model produced became a permanent new row: "Giao thức PPP và HDLC",
"Giao thức HDLC và PPP", "Kiểm soát lỗi với CRC và Hamming" alongside
"... và mã Hamming".

`normalize_name` does not help here — measured on the real table it collapsed
**zero** of them, because the difference is word order, not spelling. Hence a
key that ignores order.
"""

from utils.ctdt_catalog import name_key, normalize_name


class TestOrderInsensitive:
    def test_reordered_conjunction_is_the_same_key(self):
        assert name_key("Giao thức PPP và HDLC") == name_key("Giao thức HDLC và PPP")

    def test_normalisation_alone_would_have_missed_it(self):
        """Documents why a second key exists at all."""
        assert normalize_name("Giao thức PPP và HDLC") != normalize_name("Giao thức HDLC và PPP")

    def test_spelling_variants_still_collapse(self):
        assert name_key("Kỹ thuật ra đa – dẫn đường") == name_key("KỸ THUẬT RA ĐA - DẪN ĐƯỜNG")

    def test_extra_words_keep_it_distinct(self):
        """A longer, more specific direction is not the same direction."""
        assert name_key("Giao thức PPP và HDLC") != name_key(
            "Giao thức PPP và HDLC trong tầng liên kết"
        )

    def test_different_topics_stay_distinct(self):
        assert name_key("Cơ học công trình") != name_key("Cơ học vật rắn")

    def test_empty_input_is_safe(self):
        assert name_key("") == ()
        assert name_key(None) == ()
