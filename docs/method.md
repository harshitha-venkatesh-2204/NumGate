# The NumGate number gate: method

This document states every rule the automatic gate applies, in plain language, with worked examples. The implementation is `src/number_gate.py`; the examples below are also unit tests in `tests/test_number_gate.py`.

All examples use this five-row country table (ISO week 2011-W06, revenue in GBP):

| Entity | Revenue_TW | Revenue_LW | Revenue_LY | Units_TW | Units_LW | Orders_TW | Orders_LW | WoW_Pct | YoY_Pct | Share_Pct | Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| United Kingdom | 89759.37 | 106749.18 | 74480.20 | 44149 | 62225 | 204 | 263 | -15.9 | 20.5 | 89.5 | 1 |
| Japan | 5735.24 | 45.57 | 0.00 | 2962 | 37 | 1 | 1 | 12485.6 | | 5.7 | 2 |
| Germany | 2840.80 | 3894.00 | 4144.93 | 1025 | 1605 | 4 | 9 | -27.0 | -31.5 | 2.8 | 3 |
| Netherlands | 1156.28 | 303.93 | 0.00 | 1116 | 227 | 1 | 2 | 280.4 | | 1.2 | 4 |
| France | 852.44 | 5710.94 | 4804.52 | 964 | 2992 | 3 | 6 | -85.1 | -82.3 | 0.8 | 5 |

## 1. Units of analysis

A note is a list of bullet lines. Each line is one insight. A **number** is any numeric mention in the note. A **claim** is a (entity, metric, period, value, direction) statement extracted from a line. The gate has two levels: Level 1 asks whether each number can be found in or derived from the table; Level 2 asks whether each claim attaches its number to the right entity, metric and period.

## 2. Level 1: number support

### 2.1 Extraction

Every numeric mention is found with one regular expression and turned into a record:

- **Value and precision.** Thousands separators are removed. The number of decimals written is the cited precision. "£89,759.37" has value 89759.37 and 2 decimals.
- **Multipliers.** k or thousand (x 1,000), M, m, mn or million (x 1,000,000), B, bn or billion (x 1,000,000,000). "£89.8k" has value 89,800, cited at 1 decimal on the thousands scale.
- **Unit.** `currency` if the number carries £, $, € or GBP/USD/EUR; `percent` if followed by %, percent, per cent, pct, pc, pp, pts, points or percentage points; `count` if followed by units, orders, invoices, customers, items or transactions, or if it is a rank; otherwise `plain`. A number followed by x, times or fold is a **ratio** (recorded with unit `plain` and flagged `is_ratio`).
- **Sign.** An explicit sign is kept: hyphen, minus sign, or an en dash written as a minus ("–60.6%"). A dash written directly after a percent sign or a closing bracket is a range ("1.7%–2.3%"), not a sign. Otherwise a direction word can set the sign:
  - A down word (fell, declined, down, dropped, decreased, lower, lost and inflections) within the three words before the number, in the same clause, makes it negative: "Germany revenue fell 27.0% WoW" gives -27.0. An up word (rose, grew, up, increased, gained, higher and inflections) makes it positive.
  - The direction lists include common verbs such as rebounded, advanced, spiked, slumped, sank, plummeted, eased and halved. "Gave up £2k" is a loss; "made up 40%" has no direction.
  - A level word (to, at, from, reaching, totaling, was, were, is) between the direction word and the number stops the search: in "fell to just £515.12" the level is unsigned, even with "just" in between.
  - A direction word right after the number also counts, skipping period labels: "an 85.1% decline", "a 39.2% year-on-year decline", "10.3% lower." (sentence end), "2.8% up on last week", "0.3% above LY".
  - A number with neither an explicit sign nor a direction word is **unsigned**.
- **Hedge.** "about", "around", "roughly", "nearly", "almost", "some", "circa", "c." or "~" just before the number make it **approximate** ("c.137%" is read as about 137%); "more than", "over", "above", "at least" make it a **lower bound**; "less than", "under", "below", "at most" make it an **upper bound**.
- **Rank.** A number with an ordinal suffix (3rd), or preceded by #, No., No.3, rank or ranked, is a rank claim. In a range such as "ranks 1 to 6", both numbers are ranks.
- **Period label.** A TW, LW or LY label directly after the number is recorded; when the number matches cells of its own row, cells of that period are tried first, so the evidence names the right cell ("£0.00 LY" is matched to Revenue_LY even when Revenue_LW is also 0).

### 2.2 Skipped numbers

Some numbers are not measurements. They are labelled **Skipped** with a reason and left out of every rate. The number of skipped numbers is recorded.

- **Entity names.** Digits inside an entity name that appears in the line ("PACK OF 72 RETRO SPOT CAKE CASES"), including a shortened name when the digit and its neighbouring word are consecutive words of some entity ("the 50'S CHRISTMAS kit"). Entity names are also found when written with different spacing or punctuation.
- **Dates and week labels.** 2011-W06, 2011-02-07, W06, week 6, wk 6.
- **Years.** A bare integer from 1990 to 2035 with no separator or unit.
- **Counting words.** A bare integer of at most 200 that follows top, bottom, first, last, next or past ("top 3"); that is followed by "of (the) (top) N" or is the N in "N of the M" ("2 of the 5 countries"); or that is followed, optionally after one adjective such as listed, shown or trading, by a counting noun (countries, products, markets, regions, entities, lines, SKUs, stores, weeks, days, months, categories, rows, colourways).

### 2.3 Matching rule

A cited number **matches** a true value when the true value rounds half-up to the cited value at the cited precision:

> -h <= |true| / scale - |cited| < h, where h = 0.5 x 10^(-decimals)

so 6.5 rounds to 7 and does not match a cited 6. A signed citation must also agree in sign with the true value.

For numbers written with k, M or B, a match within 0.5 percent relative error is also accepted, to allow for truncation instead of rounding. Unsigned citations are compared on absolute values; signed citations must match the sign. An approximate citation also matches within 5 percent. A bound is a near miss: "more than 900%" matches a true value from 900 to 1.5 times 900, and "less than" a value from two thirds of the bound up to it (never 0). On a line that names an entity, bounds are checked only against that entity's cells and derived values; on a line that names no entity, against table-level values and cells.

Examples against the table:

- "£89.8k" matches United Kingdom Revenue_TW 89,759.37 (89.759 rounds to 89.8).
- "£89.8k" also matches 89,700 (within 0.5 percent) but not 89,000.
- "£1.23M" would match 1,236,000 (0.49 percent away) but not 1,240,000.
- "15.9%" (unsigned) matches WoW_Pct -15.9; "+15.9%" does not.
- "£89,759" matches 89,759.37 at 0 decimals because |89,759.37 - 89,759| = 0.37 <= 0.5; "£89,758" does not.

### 2.4 Support search

Each non-skipped number is compared with candidate values of a compatible unit: equal units match; a `plain` number may match any unit except ranks and ratios; a rank matches only the Rank column; a ratio matches only ratio values. Candidates are tried in this order, and the first hit decides the label and the evidence:

1. **Cells of the entities named in the line** (Supported_cell).
2. **Values derived from those entities** (Supported_derived):
   - for each named entity: the difference, percent change, percent-of and ratio between TW and LW (and TW and LY) of each measure (Revenue, Units, Orders); its share of each TW and LW column total; the change in its revenue share in percentage points; the difference, percent change and ratio against the column mean; revenue per order and per unit; and the total of every other row ("non-UK markets"), with its week over week change;
   - for every pair of named entities: the difference of each column, and the percent change and ratio of each non-percent column;
   - for each named entity, units per order;
   - for the named entities together, and for each **product family** the line mentions (a run of one to three words, such as LANDMARK FRAME or hot water bottle, that appears in 2 to 20 entity names), plus the mean revenue of the family and the total and share of everything outside the group: the sums of each measure for TW, LW and LY, their difference and percent change, their share of each column total, and the sum of Share_Pct. When a line mentions several families, their union is also a group.
3. **Table-level values** (Supported_derived). Plain column sums are always available. The rest are used only when the line names no entity or uses an aggregate word (total, totalled, overall, top, all, across, combined, together, average, rest, others, listed, shown, and similar): for every non-percent column the sum, mean, median, max and min; for percent columns the mean, max and min; the row count; for each measure the difference, percent change, percent-of and ratio between the column totals for TW and LW, and TW and LY; the cumulative revenue and cumulative share of the top k rows for k = 2 to 10 and of the rows after the top k for k = 1 to 10; and whole-table revenue per order, revenue per unit and units per order for TW and LW, with their week over week changes.
4. **Any other cell of the table** (Supported_cell).
5. **Unsupported.** Nothing matched.

Keeping derived values anchored to the entities, groups and families named in the line keeps the candidate set small. Without that, a 150-row table offers hundreds of thousands of pairwise values and almost any number would match something by chance. Step 4 keeps the Level 1 question as "does this number exist in the table at all"; whether it is attached to the right row is the Level 2 question.

Worked examples:

| Line | Number | Label | Evidence |
|---|---|---|---|
| United Kingdom revenue was £89,759.37 TW | £89,759.37 | Supported_cell | cell(United Kingdom, Revenue_TW) |
| Germany took 2.0% of units TW | 2.0% | Supported_derived | share(Germany.Units_TW) = 1025 / 50216 = 2.04% |
| Germany units fell 36.1% WoW | 36.1% (signed -36.1) | Supported_derived | pct_change(Germany.Units_TW, Germany.Units_LW) = -36.1% |
| Total revenue was £100.3k TW | £100.3k | Supported_derived | sum(Revenue_TW) = 100,344.13 |
| Germany revenue was 233.3% higher than France | 233.3% | Supported_derived | pct_change(Germany.Revenue_TW, France.Revenue_TW) |
| The three EDWARDIAN PARASOL lines combined for £9,139.25 TW | £9,139.25 | Supported_derived | sum(family[EDWARDIAN PARASOL].Revenue_TW) |
| Netherlands revenue was 3.8x LW | 3.8x | Supported_derived | ratio(Netherlands.Revenue_TW, Netherlands.Revenue_LW) = 3.80 |
| Germany revenue rose 27.0% WoW | 27.0% (signed +27.0) | Unsupported | the table says -27.0: wrong direction |
| France revenue fell 91.2% WoW | 91.2% | Unsupported | true value -85.1 |
| The top 3 markets in week 42 of 2011 | 3, 42, 2011 | Skipped | counting, date_or_week, year |
| Germany ranked 3rd by revenue TW | 3rd | Supported_cell | cell(Germany, Rank) |

**Level 1 metric.** Number-level unsupported rate = Unsupported / (Supported_cell + Supported_derived + Unsupported), pooled over the notes in a group.

## 3. Level 2: claim binding

A number can exist in the table but belong to a different entity, metric or period. Level 2 checks the binding.

### 3.1 Claim extraction

Each line is turned into claims, one per number, with fields entity, metric (revenue, units, orders, share, rank), period (TW, LW, LY, WoW, YoY), value_text, direction (up, down, none) and qualifier (approx, over, under, none).

- **Real models:** one call to a cheap extractor model (`extractor_model` in `config.yaml`) with the fixed prompt in `prompts/extract_system.txt` and a JSON output schema. The prompt asks for the period of the value itself (a £ amount written next to LW is an LW level even if a WoW change follows), a direction only for changes, GROUP for figures about several entities together, and values in listed order for "respectively" sentences. The response is cached. The extractor only transcribes; the value is re-parsed from `value_text` by the Level 1 parser so precision and sign are handled identically. If the reply is not valid JSON, the extractor is asked once more; if it still fails, the note is left out of the claim rates and marked `extract_failed` (the rule-based extractor is never mixed into a real model's rates).
- **Mock model, or `extractor_model: regex`:** a rule-based extractor. The entity is the nearest entity mention before the number (Total and overall map to a TOTAL row). The metric is the nearest metric keyword before the number (revenue, sales, units, orders, share, rank), else the first keyword after it within the same clause, else the metric of the previous claim in the line, else revenue; currency is always revenue. The period is a period label within two words after the number, else the nearest one before it, else WoW for percentages and direction-bearing numbers, else TW.

**Labels written in the note win.** Before verification, the gate finds the value in its line, only where it stands alone (so "33" is not found inside "£680.33"). A period label directly after it ("£2,729.22 LW", "+17.5% WoW") or directly before it ("LW: £3,792.77") replaces the extractor's period when the label fits the value (a level label for money and counts, a change label for percentages). Change wording marks a money or count value as a change when the extractor called it a level: a change noun just before it ("a decline of £51.1k", "the largest swing at £1,492.90") or a comparison with no second value ("£16.8k versus LW"); "from £10,902.59 on LW" and "£3,740.98 versus LW £3,792.77" stay levels. A unit noun directly after the value ("8 orders", "140 units", "6.1% share") replaces the metric; a hedge word before it sets the qualifier. Every such change is recorded in the `adjusted` column of claims.csv. Ratios keep the extractor's period.

### 3.2 Verification

1. **Group claims.** A claim whose entity is GROUP (or "top five", "the rest", and similar) is checked against the values Level 1 builds for its line: group and product-family sums, changes and shares, top-k totals, totals of the rest of the table, and table-level values. It is **Correct** if one matches and **Wrong_value** otherwise. A claim the extractor gave to TOTAL that does not match the table total is treated the same way when the line talks about a subset (it names a product family or several entities, or says top, combined, together, lines or colourways): "hot water bottle lines totalled £12,111.00" is a family total, not the table total.
2. **Entity.** A name that joins two or more exact entity names ("UK and Germany") is a group. Otherwise find the row or rows: exact case-insensitive match; an alias or demonym list (UK and British to United Kingdom, Ireland and Irish to EIRE, German to Germany, and similar); phrases such as "across all 139 products" map to a virtual TOTAL row of column totals; after dropping a leading "the", a trailing "'s", a parenthesis and words such as "lines", every row whose name contains all the words of the claim's name; otherwise fuzzy matching (rapidfuzz WRatio at least 85, keeping every row within 2 points of the best). Words in brackets are kept first when they narrow the match ("Cutlery (Black)" fits only the black set). A name that fits more than five rows, or no row, is **Unverifiable**. A name found only by fuzzy matching never produces a confident error: if its check fails, the claim is **Unverifiable** (binding uncertain). A rank claim is checked as a rank even when the extractor writes it as a bare number.
3. **Ambiguous names.** When a name fits several rows (for example "Hanging heart T-light holder" fits the WHITE and the RED product), the claim is checked against each: it is **Correct** if it matches any of them and **Unverifiable** if it matches none, because the note never said which row it meant.
4. **Direction.** The extractor's direction signs the value only for changes (WoW, YoY). Levels (TW, LW, LY), shares at a level, and ranks are compared unsigned, so "fell to £1,632.90 TW" is checked as the level £1,632.90.
5. **Column.** Map (metric, period, unit) to a column. A percentage attached to a level is a share of that column (units TW with a percent value means the entity's share of units); a ratio ("3.8x LW") is the TW / LW ratio. Revenue WoW and YoY percentages are compared with both the rounded table cell and the unrounded change, and share changes with both the rounded and the exact share, so a correctly computed two-decimal figure is not a false alarm. When the value's type cannot fit the column at all (a money value for a count), the claim is Correct if it equals one of the row's derived values (as at Level 1) and Unverifiable otherwise. Levels map to stored columns (revenue TW to Revenue_TW). Changes map to percent, absolute or ratio change columns: a percentage WoW revenue claim maps to WoW_Pct; a currency WoW revenue claim maps to Revenue_TW minus Revenue_LW; "3.8x LW" maps to Revenue_TW / Revenue_LW; units and orders WoW changes are computed the same way. Share TW maps to Share_Pct, share LW to the LW share, a share WoW change to the change in percentage points, and rank to Rank.
6. **Label**, using the Level 1 matching rule, in this order:
   - **Correct:** the value matches the claimed cell.
   - **Wrong_value (direction wrong):** the value matches the claimed cell only when the sign is ignored.
   - **Wrong_metric:** the value matches another stored or change column of the same row with the same unit (a period or metric mix-up, such as the YoY figure cited as WoW). Derived shares are not searched here, so coincidences are not called mix-ups.
   - **Wrong_entity:** the value matches the same column in another row that is named in the same line, or in exactly one other row of the table. Ranks need the other entity to be named, because every rank from 1 to n exists somewhere.
   - **Wrong_value:** none of the above. The evidence notes when the value also matches several other rows by coincidence.
   - When the claimed cell is empty: **Wrong_metric** if the value matches another column of the row; **Wrong_value** if the claim is a percent change from a zero base (no such change exists); otherwise **Unverifiable** (for example units YoY, since there is no Units_LY).

Worked examples:

| Claim (entity, metric, period, value) | Label | Why |
|---|---|---|
| Germany, revenue, WoW, -27.0% | Correct | WoW_Pct for Germany is -27.0 |
| Germany, revenue, TW, £2,840.80, direction down | Correct | a level; the direction is ignored |
| Germany, revenue, TW, £852.44 | Wrong_entity | 852.44 is France's Revenue_TW, and only France's |
| Germany, revenue, TW, £3,894.00 | Wrong_metric | 3,894.00 is Germany's Revenue_LW |
| Germany, revenue, WoW, -31.5% | Wrong_metric | -31.5 is Germany's YoY change |
| Germany, rank, TW, 2nd | Wrong_value | Germany is 3rd, and the line names no other entity |
| GROUP, revenue, TW, £9,139.25 | Correct | the EDWARDIAN PARASOL family sums to £9,139.25 |
| GROUP, revenue, TW, £14,211.15 | Wrong_value | no group, family or top-k total in the line gives this value |
| Brazil, revenue, TW, £500 | Unverifiable | Brazil is not in the table |

**Level 2 metrics.**
- **Binding error rate** (research question 2) = (Wrong_entity + Wrong_metric) / (Correct + Wrong_entity + Wrong_metric + Wrong_value): the share of verifiable claims that attach a number to the wrong entity, metric or period.
- **Claim error rate** = (Wrong_entity + Wrong_metric + Wrong_value) / the same denominator: any wrong claim, including wrong values, which overlap with Level 1.
- Unverifiable claims are reported separately and are not in either denominator.

## 4. Planted-error check

The note below has three planted errors. The gate flags exactly those three (test `test_planted_errors_flag_exactly_three`):

```
- United Kingdom revenue was £89.8k TW, down 15.9% WoW
- United Kingdom revenue rose 20.5% YoY versus £74,480.20 LY
- Japan revenue was £5,735.24 TW with a 5.7% share
- Germany revenue fell 27.0% WoW to £2,840.80
- France revenue fell 91.2% WoW                          <- true value -85.1
- Netherlands revenue rose 280.4% WoW to £1,256.28       <- true TW is £1,156.28
- Germany units rose 36.1% WoW                           <- units fell 36.1%
- Total revenue was £100.3k TW
```

## 5. Statistics

Rates pool numbers (or claims) across the notes in a group. Confidence intervals are 95 percent percentile bootstrap intervals with 1,000 resamples of notes, so the within-note correlation of errors is respected. Differences between strategies use a paired bootstrap over conditions matched on table, model and repeat. S2 notes whose facts script failed twice are excluded from error rates and reported as a failure rate.

## 6. Validating the gate

`src/make_audit_sample.py` samples 200 numbers and 200 claims, stratified by strategy and gate label, into `results/audit_sample.csv`. A human fills `human_label` with the same vocabulary, looking up the table in `data/tables/<table_id>.csv`. `src/score_audit.py` reports precision and recall per label and for the binary decision "error versus fine". Report these alongside the headline rates.

## 7. Known failure modes

- Numbers written as words ("doubled", "a third", "two-fold") are not extracted.
- A legitimate derived value outside the operation library (for example a three-week average) is labelled Unsupported.
- In large tables a wrong number can coincide with an unrelated cell and be labelled Supported_cell at Level 1. Level 2 catches this when the claim names an entity.
- Truncated values cited without k or M ("£89,758" for 89,758.90) are counted as unsupported; the rounding rule is strict by design.
- Level 2 inherits extractor errors: a wrong entity or period from the extractor produces a false Wrong_* label. Labels written next to a value correct many of these (section 3.1), but not a value given to the wrong entity in a list.
- Ambiguous short names are Unverifiable at Level 2. A group figure for a group the library does not know (for example "the eight products that grew") is labelled Wrong_value at Level 2 and Unsupported at Level 1.
- Superlatives ("the largest swing", "led the table") are not checked.
- Group shares are accepted when they equal the sum of the table's rounded Share_Pct cells, even if the exact share rounds differently (37.2% against an exact 37.13%): the table as shown supports them. Counting these as errors is a policy choice.
- Level 1 cannot see binding errors: a real cell attached to the wrong row or period is Supported at Level 1 by design. Always read Level 1 together with Level 2.
- The 0.5 percent tolerance for k and M values accepts a wrong last digit at large scales (for example £209.5k for £208.7k).
