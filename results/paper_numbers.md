# Headline numbers

Run smoke_opus: 48 notes from 6 tables, models claude-opus-5, 2 repeats per condition. Confidence intervals are 95 percent percentile bootstrap intervals over notes (1000 resamples).

Under direct generation (S1), 0.6 percent of cited numbers were unsupported by the table (95 percent CI 0.0 to 2.0; 514 numbers in 12 notes); compute-then-write (S2) gave 0.0 percent (95 percent CI 0.0 to 0.0), a change of -0.6 points (paired 95 percent CI -1.9 to +0.0); write-then-verify (S3) gave 0.0 percent (95 percent CI 0.0 to 0.0), a change of -0.6 points (paired 95 percent CI -2.0 to +0.0); facts-template (S4) gave 0.0 percent (95 percent CI 0.0 to 0.0), a change of -0.6 points (paired 95 percent CI -1.8 to +0.0).

At the claim level, 0.0 percent of verifiable S1 claims attached a number to the wrong entity, metric or period (95 percent CI 0.0 to 0.0; 508 claims), compared with S2 0.0 percent (CI 0.0 to 0.0), S3 0.0 percent (CI 0.0 to 0.0), S4 0.0 percent (CI 0.0 to 0.0).

Counting wrong values as well, 0.6 percent of verifiable S1 claims had an error (95 percent CI 0.0 to 2.2), compared with S2 0.0 percent (CI 0.0 to 0.0), S3 0.0 percent (CI 0.0 to 0.0), S4 0.0 percent (CI 0.0 to 0.0).

Of the S1 claim errors, 0% were wrong-entity, 0% wrong-metric, and 100% wrong-value bindings.

By table size under S1 (direct), the unsupported rate was small tables 0.0 percent (CI 0.0 to 0.0), medium tables 1.6 percent (CI 0.0 to 6.2), large tables 0.0 percent (CI 0.0 to 0.0).

By table size under S2 (compute-then-write), the unsupported rate was small tables 0.0 percent (CI 0.0 to 0.0), medium tables 0.0 percent (CI 0.0 to 0.0), large tables 0.0 percent (CI 0.0 to 0.0).

Write-then-verify (S3) moved the mean per-note unsupported rate from 0.7 percent on the draft, 0.0 percent after round 1. S3 revises against the same level 1 gate that scores it, so its level 2 claim error rate is the fairer comparison.

Compute-then-write (S2) facts scripts failed on both attempts in 0 of 12 notes (0.0%); failed notes are excluded from S2 error rates.
