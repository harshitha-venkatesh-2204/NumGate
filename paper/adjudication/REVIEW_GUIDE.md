# Reviewer guide: is this number right?

Each item is one number in a weekly sales note, written about one table. Decide whether the number is correct
**for what the sentence says it is**, using only the table in `table_csv` (open it with Python and pandas).

## The table
- `Entity` is a country or a product. Rows shown are the whole universe for the note.
- `Revenue_TW`, `Revenue_LW`, `Revenue_LY`: revenue in GBP this week, last week, and the same ISO week last year (blank = not available).
- `Units_TW`, `Units_LW`, `Orders_TW`, `Orders_LW`: units and distinct invoices.
- `WoW_Pct`, `YoY_Pct`: revenue percent change TW vs LW and TW vs LY, one decimal (blank when the base is zero or missing).
- `Share_Pct`: the row's percent share of TW revenue across the rows shown. `Rank`: rank by TW revenue (1 is highest).
- TW = this week, LW = last week, LY = same week last year, WoW = week over week, YoY = year over year.

## Steps
1. Read the whole note for context, then the marked number: line `line`, text `number`, at `char_offset_in_line`.
2. Work out what the sentence says the number is: which entity or group (a named row, several named rows, "the top five", "the rest", or all rows), which measure (revenue, units, orders, share, rank, a percent change, a difference, a ratio, an average per order or per unit, a count of rows), and which period.
3. Compute that quantity from the table. Totals "overall" or for "TOTAL" are sums over all rows shown. A percent change is (new / old - 1) x 100. A change in share is in percentage points.
4. Compare at the precision written:
   - A rounded number is correct if the true value rounds to it at the written precision. `£58.3k` means 58,300 to the nearest 100. Accept either half-up or half-even rounding.
   - For a stated percent change, accept the table's `WoW_Pct` / `YoY_Pct` value or the value recomputed from the revenue cells.
   - Hedges: "about", "around", "roughly", "close to" are correct within 2 percent of the true value (or one unit in the last written digit, whichever is larger). "over" / "more than" / "above" need the true value to be greater; "under" / "less than" / "below" need it to be smaller; "nearly" / "almost" need it to be slightly smaller (within 5 percent).
   - A direction word attached to the number ("rose 5%", "down 12%", "a decline of") must match the sign.
5. Give a verdict:
   - `correct`: the number is right for what the sentence says.
   - `wrong`: it is not. That includes a wrong value, the right value for a different entity, metric or period, a group total that does not add up, or a wrong direction.
   - `unclear`: the sentence is genuinely ambiguous and one reasonable reading makes it right while another makes it wrong. Use this rarely and say why.
   - `not_a_measurement`: the number is part of a name or label (for example "SET OF 3"), a rank range, a week label, or similar, not a figure from the table.

Write `reading` (what the number is said to be, in a few words), `expected` (what the table gives for that reading, with
the computation for anything derived) and `reason` (one short sentence). Judge each item independently; do not assume
anything about how the note was produced.
