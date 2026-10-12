"""§2.3 of the 502-page Ballistics digest was chapter-title phrases the model rated 0.95
(«Ammunition Design Practice», 2 mentions in the book) while «Projectile» (1181) and
«Muzzle velocity» (38) were absent. The model's weight now shares the vote with how
often the book actually uses the term."""

from utils.keyword_validation import rank_by_frequency

TEXT = (
    "The projectile leaves the barrel. " * 300
    + "Muzzle velocity is measured near the muzzle. " * 20
    + "Ammunition Design Practice is a chapter title. "
)


def _kw(name, weight):
    return {"keyword": name, "weight": weight}


def test_a_term_the_book_uses_constantly_outranks_a_rarely_used_one_with_a_higher_model_weight():
    ranked = rank_by_frequency([_kw("Ammunition Design Practice", 0.95), _kw("projectile", 0.8)], TEXT)

    assert [k["keyword"] for k in ranked] == ["projectile", "Ammunition Design Practice"]


def test_the_result_is_sorted_by_the_blended_weight_and_stays_within_zero_and_one():
    ranked = rank_by_frequency(
        [_kw("Muzzle velocity", 0.7), _kw("projectile", 0.9), _kw("Ammunition Design Practice", 1.0)],
        TEXT,
    )

    weights = [k["weight"] for k in ranked]
    assert weights == sorted(weights, reverse=True)
    assert all(0.0 <= w <= 1.0 for w in weights)


def test_a_term_the_counter_cannot_find_is_not_penalised_to_the_bottom():
    """Hyphenation or width folding can make a validated term miss an exact count."""
    ranked = rank_by_frequency([_kw("unfindable", 0.9), _kw("Ammunition Design Practice", 0.9)], TEXT)

    assert ranked[0]["keyword"] == "unfindable"


def test_other_fields_of_each_item_are_kept():
    ranked = rank_by_frequency([{"keyword": "projectile", "weight": 0.5, "vi": "đầu đạn"}], TEXT)

    assert ranked[0]["vi"] == "đầu đạn"
