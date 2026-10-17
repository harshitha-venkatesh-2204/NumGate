# NumGate full study: analysis plan, fixed before data collection

Written on 2026-10-17, before any full-study note was generated. It is committed to the repository before the run, so the commit history shows that the plan predates the results. Anything not listed here is reported as exploratory.

## What is frozen

- **The gate.** `src/number_gate.py`, MD5 `bcdb434ba30ad334a702493668a0ce39` (gate v3, the version validated on held-out set 2). No gate rule changes until the analysis below is complete. If a later change is made, results are reported under both versions and the change is labelled exploratory.
- **Prompts, tables and code.** `prompts/`, `data/tables/` and `src/` as of this commit.

## Design

- **Writers:** `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5` (Anthropic API; temperature is not settable on Opus 5 and Sonnet 5, so their samples use the API default).
- **Strategies:** S1 Direct, S2 Compute-then-write, S3 Write-then-verify (up to 2 revision rounds), S4 Facts-template.
- **Tables:** 60 weekly tables, 20 per size (small 5 to 10 rows, medium 20 to 40, large 80 to 150), chosen by a seeded shuffle (seed 42).
- **Repeats:** 2 per cell. Total: 3 x 4 x 60 x 2 = 1,440 notes.
- **Claim extractor:** `claude-haiku-4-5` (the regex fallback is not used for real models).
- **Budget cap:** USD 60 for generation plus extraction.

## Primary and secondary sets

The gate's rules were revised after looking at the pilot's flags, and the pilot used 6 of these 60 tables. The **primary set is the 54 tables the pilot did not use**. All 60 tables are reported as a secondary analysis.

## Outcomes

1. **Gate unsupported rate:** numbers the gate marks Unsupported / numbers checked.
2. **Gate binding error rate:** claims marked Wrong entity or Wrong metric / verifiable claims.
3. **Adjudicated error rate (main outcome):** every gate flag in the primary set (each Unsupported number and each Wrong entity, Wrong metric or Wrong value claim) is checked against its source table by two independent reviewers who see the sentence, the number and the table, but not the model or the strategy. A third reviewer resolves disagreements. Adjudicated error rate = confirmed wrong numbers / numbers checked. **Gate precision** = confirmed flags / flags.
4. **Gate miss rate:** a seeded random sample of 300 numbers the gate supports and 200 claims it marks Correct, stratified by model, is reviewed the same way. Reported with exact (Clopper-Pearson) 95% intervals.

The reviewers are Claude agents with programmatic access to the table cells, not people. The paper says so, and a human spot check of a random subset is recommended before publication.

## Hypotheses (directional, from the pilot)

- **H1** S4 (Facts-template) has the lowest adjudicated error rate for every model.
- **H2** The gate's raw flag rate exceeds the adjudicated error rate (gate precision below 1), most of all for the strongest writer.
- **H3** For S1, error rates are higher on large tables than on small ones.
- **H4** Under S1, Haiku 4.5 has a higher adjudicated error rate than Opus 5.

## Statistics

- 95% intervals from a cluster bootstrap that resamples **tables** (2,000 resamples, seed 42), since the notes on one table are not independent.
- Strategy effects: paired differences against S1 on the same tables, cluster-bootstrapped, with a Holm correction across the 3 contrasts within each model. Model effects: the same against Opus 5 within each strategy.
- Reliability: agreement between repeat 1 and repeat 2 on whether a note contains any confirmed error (Cohen's kappa).

## Exclusions

- S2 notes whose facts script failed twice produce no note; they are reported as a failure rate and excluded from error rates.
- Notes whose claim extraction failed are reported and excluded from claim-level rates only.
- No other exclusions.
