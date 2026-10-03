import pandas as pd
import pytest

from build_tables import add_derived
from number_gate import (extract_numbers, gate_level1, gate_level2, matches, parse_value_text,
                         regex_claims, summarize, verify_claim)


@pytest.fixture
def table():
    t = pd.DataFrame({
        "Entity": ["United Kingdom", "Japan", "Germany", "Netherlands", "France"],
        "Revenue_TW": [89759.37, 5735.24, 2840.80, 1156.28, 852.44],
        "Revenue_LW": [106749.18, 45.57, 3894.00, 303.93, 5710.94],
        "Revenue_LY": [74480.20, 0.0, 4144.93, 0.0, 4804.52],
        "Units_TW": [44149, 2962, 1025, 1116, 964],
        "Units_LW": [62225, 37, 1605, 227, 2992],
        "Orders_TW": [204, 1, 4, 1, 3],
        "Orders_LW": [263, 1, 9, 2, 6],
    })
    return add_derived(t)


@pytest.fixture
def product_table():
    t = pd.DataFrame({
        "Entity": ["PACK OF 72 RETRO SPOT CAKE CASES", "JUMBO BAG RED RETROSPOT", "SET OF 3 CAKE TINS PANTRY DESIGN"],
        "Revenue_TW": [1520.40, 1210.00, 640.25], "Revenue_LW": [1300.00, 1400.00, 600.00],
        "Revenue_LY": [None, None, None], "Units_TW": [2800, 620, 130], "Units_LW": [2400, 700, 120],
        "Orders_TW": [40, 31, 12], "Orders_LW": [35, 33, 11],
    })
    return add_derived(t)


def one(text, entities=()):
    recs = [r for r in extract_numbers(text, entities)]
    assert len(recs) == 1, recs
    return recs[0]


def labels(note, table):
    return [(r["raw_text"], r["label"]) for r in gate_level1(note, table)]

# ---------------------------------------------------------------- parsing


def test_currency_with_thousands_separator():
    r = one("Revenue was £12,345.67 this week")
    assert r["unit"] == "currency" and r["value"] == pytest.approx(12345.67) and r["decimals"] == 2


def test_negative_percent():
    r = one("WoW change of -15.9%")
    assert r["unit"] == "percent" and r["value"] == pytest.approx(-15.9) and r["signed"]


def test_k_multiplier():
    r = one("UK revenue £89.8k")
    assert r["scale"] == 1e3 and r["value"] == pytest.approx(89800) and r["decimals"] == 1


def test_m_and_b_multipliers():
    assert one("total of £1.2M")["value"] == pytest.approx(1.2e6)
    assert one("about $3bn in sales")["value"] == pytest.approx(3e9)


def test_percent_words_and_points():
    assert one("share rose 2.1 percentage points")["unit"] == "percent"
    assert one("a 4 percent gain")["unit"] == "percent"


def test_direction_word_down_sets_sign():
    r = one("Germany revenue fell 27.0% WoW")
    assert r["direction"] == "down" and r["value"] == pytest.approx(-27.0)


def test_direction_word_up_sets_sign():
    r = one("UK revenue grew 20.5% YoY")
    assert r["direction"] == "up" and r["value"] == pytest.approx(20.5)


def test_direction_after_number():
    assert one("France posted an 85.1% decline WoW")["value"] == pytest.approx(-85.1)


def test_level_after_direction_is_not_signed():
    r = one("Germany revenue fell to £2,840.80")
    assert r["direction"] == "none" and not r["signed"]


def test_count_unit():
    assert one("UK sold 44,149 units")["unit"] == "count"

# ---------------------------------------------------------------- rounding rule


def test_rounding_aware_match():
    rec = parse_value_text("£89.8k")
    assert matches([89759.37], rec)[0]
    assert matches([89700.00], rec)[0]       # rounds to 89.7k but is within 0.5 percent
    assert not matches([89000.00], rec)[0]   # 0.9 percent away


def test_half_percent_tolerance_for_millions():
    rec = parse_value_text("£1.23M")
    assert matches([1_236_000], rec)[0]       # rounds to 1.24M but is within 0.5 percent
    assert not matches([1_240_000], rec)[0]   # 0.8 percent away and rounds to 1.24M


def test_unsigned_citation_matches_absolute_value():
    assert matches([-15.9], parse_value_text("15.9%"))[0]
    assert not matches([-15.9], parse_value_text("+15.9%"))[0]

# ---------------------------------------------------------------- level 1 support


def test_exact_cell_match(table):
    assert labels("- United Kingdom revenue was £89,759.37 TW", table) == [("£89,759.37", "Supported_cell")]


def test_derived_share_of_units(table):
    # Germany units 1025 of 50216 total = 2.04 percent
    assert labels("- Germany took 2.0% of units TW", table) == [("2.0%", "Supported_derived")]


def test_derived_percent_change_units(table):
    # Germany units 1025 vs 1605 = -36.1 percent, not a stored column
    assert labels("- Germany units fell 36.1% WoW", table) == [("36.1%", "Supported_derived")]


def test_derived_column_sum_in_k(table):
    # total revenue 100,344.13 cited as £100.3k
    assert labels("- Total revenue was £100.3k TW", table) == [("£100.3k", "Supported_derived")]


def test_derived_percent_change_between_entities(table):
    # Germany 2840.80 vs France 852.44 = +233.3 percent
    out = labels("- Germany revenue was 233.3% higher than France", table)
    assert out == [("233.3%", "Supported_derived")]


def test_wrong_direction_is_unsupported(table):
    assert labels("- Germany revenue rose 27.0% WoW", table) == [("27.0%", "Unsupported")]


def test_ordinal_and_counting_numbers_are_skipped(table):
    note = "- The top 3 markets in week 42 of 2011 were led by the UK\n- 2 of the 5 countries grew WoW"
    out = labels(note, table)
    assert all(label == "Skipped" for _, label in out) and len(out) == 5


def test_rank_is_checked_not_skipped(table):
    assert labels("- Germany ranked 3rd by revenue TW", table) == [("3rd", "Supported_cell")]


def test_numbers_inside_entity_names_are_skipped(product_table):
    out = labels("- PACK OF 72 RETRO SPOT CAKE CASES led with £1,520.40 TW", product_table)
    assert out == [("72", "Skipped"), ("£1,520.40", "Supported_cell")]


def test_planted_errors_flag_exactly_three(table):
    note = "\n".join([
        "- United Kingdom revenue was £89.8k TW, down 15.9% WoW",
        "- United Kingdom revenue rose 20.5% YoY versus £74,480.20 LY",
        "- Japan revenue was £5,735.24 TW with a 5.7% share",
        "- Germany revenue fell 27.0% WoW to £2,840.80",
        "- France revenue fell 91.2% WoW",            # planted: true value is -85.1
        "- Netherlands revenue rose 280.4% WoW to £1,256.28",  # planted: true TW is 1156.28
        "- Germany units rose 36.1% WoW",             # planted: units fell
        "- Total revenue was £100.3k TW",
    ])
    flagged = [r for r in gate_level1(note, table) if r["label"] == "Unsupported"]
    assert [r["raw_text"] for r in flagged] == ["91.2%", "£1,256.28", "36.1%"]

# ---------------------------------------------------------------- level 2 claim binding


def test_claim_correct(table):
    claim = {"entity": "Germany", "metric": "revenue", "period": "WoW", "value_text": "-27.0%"}
    assert verify_claim(claim, table)[0] == "Correct"


def test_claim_wrong_entity(table):
    # £852.44 is France's revenue, attributed to Germany
    claim = {"entity": "Germany", "metric": "revenue", "period": "TW", "value_text": "£852.44"}
    label, evidence = verify_claim(claim, table)
    assert label == "Wrong_entity" and "France" in evidence


def test_claim_wrong_metric(table):
    # £3,894.00 is Germany's LW revenue, cited as TW
    claim = {"entity": "Germany", "metric": "revenue", "period": "TW", "value_text": "£3,894.00"}
    assert verify_claim(claim, table)[0] == "Wrong_metric"


def test_claim_unverifiable_entity(table):
    claim = {"entity": "Brazil", "metric": "revenue", "period": "TW", "value_text": "£500"}
    assert verify_claim(claim, table)[0] == "Unverifiable"


def test_regex_extractor_flags_planted_wrong_entity(table):
    note = "- UK revenue was £89.8k TW\n- Germany revenue was £852.44 TW"
    out = gate_level2(note, table, claims=regex_claims(note, table))
    assert [c["label"] for c in out] == ["Correct", "Wrong_entity"]
    summary = summarize(gate_level1(note, table), out)
    assert summary["claim_error_rate"] == pytest.approx(0.5)


def test_llm_extractor_output_is_parsed_and_verified(table):
    """The LLM path: JSON claims (with a direction word stripped from the value) are parsed and checked."""
    import json

    from number_gate import llm_claims

    reply = {"claims": [
        {"line": 0, "entity": "UK", "metric": "revenue", "period": "WoW", "value_text": "15.9%", "direction": "down"},
        {"line": 1, "entity": "Germany", "metric": "revenue", "period": "TW", "value_text": "£852.44", "direction": "none"},
    ]}

    def fake_complete(model, system, user, temperature=0, max_tokens=None):
        return {"text": "Here you go:\n" + json.dumps(reply), "cost_usd": 0.0}

    note = "- UK revenue was down 15.9% WoW\n- Germany revenue was £852.44 TW"
    claims, _ = llm_claims(note, table, "fake", fake_complete)
    assert [c["label"] for c in gate_level2(note, table, claims=claims)] == ["Correct", "Wrong_entity"]


# ---------------------------------------------------------------- regressions from real model notes


@pytest.fixture
def family_table():
    t = pd.DataFrame({
        "Entity": ["EDWARDIAN PARASOL BLACK", "EDWARDIAN PARASOL NATURAL", "EDWARDIAN PARASOL RED",
                   "WHITE HANGING HEART T-LIGHT HOLDER", "RED HANGING HEART T-LIGHT HOLDER", "DOORMAT KEEP CALM AND COME IN"],
        "Revenue_TW": [3448.60, 2935.10, 2755.55, 3740.98, 1807.91, 1356.79],
        "Revenue_LW": [3927.55, 2412.45, 3114.25, 3792.77, 822.00, 2068.84],
        "Revenue_LY": [None] * 6, "Units_TW": [520, 440, 410, 1382, 700, 175], "Units_LW": [600, 360, 470, 1400, 330, 272],
        "Orders_TW": [30, 25, 22, 79, 45, 20], "Orders_LW": [33, 20, 26, 66, 31, 28],
    })
    return add_derived(t)


def test_en_dash_is_a_minus_sign():
    assert one("Australia –60.6% WoW")["value"] == pytest.approx(-60.6)


def test_hedged_level_is_not_signed():
    r = one("France revenue dropped to just £515.12")
    assert r["direction"] == "none" and not r["signed"]
    assert one("France revenue fell to roughly \u00a3935")["qualifier"] == "approx"


def test_direction_word_after_comparison():
    assert one("revenue was 2.8% up on last week")["value"] == pytest.approx(2.8)
    assert one("units were 10.3% lower.")["value"] == pytest.approx(-10.3)


def test_plain_integer_cannot_match_a_rank(table):
    # 5 is only a Rank value in this table; before the fix any bare integer could match a Rank cell
    assert labels("- Japan orders were 5 LW", table) == [("5", "Unsupported")]


def test_ratio_matches_only_ratios(table):
    # Netherlands revenue TW / LW = 3.80x; 12.3x must not match a percent or count cell
    assert labels("- Netherlands revenue was 3.8x LW", table)[0][1] == "Supported_derived"
    assert labels("- Netherlands revenue was 16.2x LW", table)[0][1] == "Unsupported"


def test_family_total_is_supported(family_table):
    # 3448.60 + 2935.10 + 2755.55 = 9139.25; LW 9454.25
    out = labels("- The three EDWARDIAN PARASOL lines combined for £9,139.25 TW versus £9,454.25 LW", family_table)
    assert [lab for _, lab in out] == ["Supported_derived", "Supported_derived"]


def test_direction_is_ignored_for_levels(table):
    claim = {"entity": "Germany", "metric": "revenue", "period": "TW", "value_text": "£2,840.80", "direction": "down"}
    assert verify_claim(claim, table)[0] == "Correct"


def test_label_next_to_value_overrides_extractor_period(table):
    note = "- Germany fell to £2,840.80 TW from £3,894.00 LW (-27.0% WoW)"
    claims = [{"line_idx": 0, "entity": "Germany", "metric": "revenue", "period": "WoW", "value_text": "£3,894.00", "direction": "none"}]
    out = gate_level2(note, table, claims=claims)
    assert out[0]["label"] == "Correct" and "LW" in out[0]["adjusted"]


def test_units_noun_overrides_extractor_metric(table):
    note = "- Germany sold 1,025 units across 4 orders TW"
    claims = [{"line_idx": 0, "entity": "Germany", "metric": "units", "period": "TW", "value_text": "4", "direction": "none"}]
    assert gate_level2(note, table, claims=claims)[0]["label"] == "Correct"


def test_group_claim_is_unverifiable(family_table):
    claim = {"entity": "GROUP", "metric": "revenue", "period": "TW", "value_text": "£9,139.25"}
    assert verify_claim(claim, family_table)[0] == "Unverifiable"


def test_entity_with_and_in_its_name_is_not_a_group(family_table):
    claim = {"entity": "DOORMAT KEEP CALM AND COME IN", "metric": "revenue", "period": "TW", "value_text": "£1,356.79"}
    assert verify_claim(claim, family_table)[0] == "Correct"


def test_ambiguous_short_name_checks_every_sibling(family_table):
    # "Hanging heart T-light holder" fits WHITE and RED; the value is RED's, so the claim is correct
    claim = {"entity": "Hanging heart T-light holder", "metric": "revenue", "period": "TW", "value_text": "£1,807.91"}
    assert verify_claim(claim, family_table)[0] == "Correct"


def test_same_row_period_mixup_is_wrong_metric(table):
    claim = {"entity": "Germany", "metric": "revenue", "period": "WoW", "value_text": "-31.5%"}  # that is the YoY value
    assert verify_claim(claim, table)[0] == "Wrong_metric"


def test_rank_error_without_named_entity_is_wrong_value(table):
    claim = {"entity": "Germany", "metric": "rank", "period": "TW", "value_text": "2nd"}
    assert verify_claim(claim, table)[0] == "Wrong_value"


def test_binding_rate_excludes_wrong_values(table):
    note = "- Germany revenue was £852.44 TW\n- France revenue was £999.00 TW"
    out = gate_level2(note, table, claims=regex_claims(note, table))
    s = summarize(gate_level1(note, table), out)
    assert s["claim_error_rate"] == pytest.approx(1.0) and s["binding_error_rate"] == pytest.approx(0.5)


def test_table_totals_on_a_line_that_also_names_an_entity(table):
    # total TW 100,344.13; total LW 116,703.62 (the line also names France, so aggregates need an aggregate word)
    note = "- The 5 listed countries totalled £100,344.13 TW against £116,703.62 LW, with France the smallest"
    assert [lab for _, lab in labels(note, table)] == ["Skipped", "Supported_derived", "Supported_derived"]


# ---------------------------------------------------------------- regressions from the adjudicated Opus run


def test_half_up_rounding_does_not_match_the_lower_integer():
    rec = parse_value_text("6")
    assert not matches([6.5], rec)[0] and matches([6.49], rec)[0] and matches([5.5], rec)[0]


def test_value_is_found_only_where_it_stands_alone(table):
    # "4" appears inside "£2,840.80"; the standalone "4 orders" is Germany's Orders_TW
    note = "- Germany had £2,840.80 TW from 4 orders TW versus 9 LW"
    claims = [{"line_idx": 0, "entity": "Germany", "metric": "orders", "period": "TW", "value_text": "4"}]
    out = gate_level2(note, table, claims=claims)
    assert out[0]["label"] == "Correct" and out[0]["adjusted"] == ""


def test_change_wording_turns_a_level_into_a_change(table):
    # UK revenue 89,759.37 TW vs 106,749.18 LW: a decline of 16,989.81, written as £17.0k
    note = "- United Kingdom revenue was £89.8k TW, a decline of £17.0k versus LW"
    claims = [{"line_idx": 0, "entity": "United Kingdom", "metric": "revenue", "period": "LW", "value_text": "£17.0k"}]
    out = gate_level2(note, table, claims=claims)
    assert out[0]["label"] == "Correct" and "change wording" in out[0]["adjusted"]


def test_group_figure_is_checked_against_group_totals(family_table):
    note = "- The three EDWARDIAN PARASOL lines combined for £9,139.25 TW"
    ok = [{"line_idx": 0, "entity": "GROUP", "metric": "revenue", "period": "TW", "value_text": "£9,139.25"}]
    bad = [{"line_idx": 0, "entity": "GROUP", "metric": "revenue", "period": "TW", "value_text": "£12,139.25"}]
    assert gate_level2(note, family_table, claims=ok)[0]["label"] == "Correct"
    assert gate_level2(note, family_table, claims=bad)[0]["label"] == "Wrong_value"


def test_total_used_for_a_subset_is_a_group_figure(family_table):
    note = "- The three EDWARDIAN PARASOL lines totalled £9,139.25 TW"
    claims = [{"line_idx": 0, "entity": "TOTAL", "metric": "revenue", "period": "TW", "value_text": "£9,139.25"}]
    assert gate_level2(note, family_table, claims=claims)[0]["label"] == "Correct"


def test_rank_range_is_a_rank(table):
    out = labels("- The top markets hold ranks 1 to 5 TW", table)
    assert out == [("1", "Supported_cell"), ("5", "Supported_cell")]
    assert [r["is_rank"] for r in gate_level1("- The top markets hold ranks 1 to 5 TW", table)] == [True, True]


def test_versus_between_two_levels_is_not_a_change(table):
    note = "- Germany had TW revenue of £2,840.80 versus LW £3,894.00, a WoW change of -27.0%"
    claims = [{"line_idx": 0, "entity": "Germany", "metric": "revenue", "period": "TW", "value_text": "£2,840.80"},
              {"line_idx": 0, "entity": "Germany", "metric": "revenue", "period": "LW", "value_text": "£3,894.00"}]
    assert [c["label"] for c in gate_level2(note, table, claims=claims)] == ["Correct", "Correct"]
    note2 = "- Germany revenue was down 27.0 percent WoW from £3,894.00 on LW"
    claims2 = [{"line_idx": 0, "entity": "Germany", "metric": "revenue", "period": "LW", "value_text": "£3,894.00"}]
    assert gate_level2(note2, table, claims=claims2)[0]["label"] == "Correct"


def test_bare_rank_value_from_the_extractor_is_a_rank(table):
    claim = {"entity": "Germany", "metric": "rank", "period": "TW", "value_text": "3"}
    assert verify_claim(claim, table)[0] == "Correct"
