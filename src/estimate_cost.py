"""Estimate the USD cost of a full run before spending anything.

Usage: python src/estimate_cost.py [--models claude-opus-5 gpt-4.1 llama-3.3-70b]
Prompt sizes come from the real tables and prompt templates; output sizes are assumptions listed below.
"""
import argparse

import pandas as pd

from config import TABLES_DIR, load_config, read_prompt
from strategies import facts_packet, system_prompt, user_fields

CHARS_PER_TOKEN = 3.0  # tables are digit-heavy, which tokenizes worse than prose (about 4 chars per token)
OUT_NOTE = 350  # output tokens for one 6 to 8 bullet note
OUT_CODE = 700  # output tokens for an S2 facts script
FACTS_JSON_CHARS = 2500  # size of the S2 facts JSON sent to the writing call
S3_REVISIONS = 1.5  # expected revision calls per S3 note (0 to 2)
S2_RETRIES = 0.2  # expected extra code calls per S2 note
THINKING = {"claude-opus-5": 1500, "claude-sonnet-5": 1500}  # extra output tokens per call for models that think by default
EXTRACT_IN, EXTRACT_OUT = 900, 700  # claim extractor tokens per note


def tokens(text):
    return len(text) / CHARS_PER_TOKEN


def per_note(table, meta):
    """Per strategy: (input tokens, visible output tokens, number of calls) for one note on this table."""
    s1_in = tokens(system_prompt("s1_system.txt") + read_prompt("s1_user.txt").format(**user_fields(table, meta)))
    code_in = tokens(read_prompt("s2_code_system.txt") + read_prompt("s2_code_user.txt")) + 300
    write_in = tokens(system_prompt("s2_write_system.txt")) + FACTS_JSON_CHARS / CHARS_PER_TOKEN
    revise_in = tokens(system_prompt("s3_revise_system.txt")) + s1_in + 400
    s4_in = tokens(system_prompt("s4_system.txt") + facts_packet(table)) + 100
    code_calls = 1 + S2_RETRIES
    return {
        "S1": (s1_in, OUT_NOTE, 1),
        "S2": (code_calls * code_in + write_in, code_calls * OUT_CODE + OUT_NOTE, code_calls + 1),
        "S3": (s1_in + S3_REVISIONS * revise_in, (1 + S3_REVISIONS) * OUT_NOTE, 1 + S3_REVISIONS),
        "S4": (s4_in, OUT_NOTE, 1),
    }


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=cfg["experiment_models"])
    args = p.parse_args()
    idx = pd.read_csv(TABLES_DIR / "index.csv")
    idx = pd.concat([g.head(cfg["tables_per_size"]) for _, g in idx.groupby("size")])
    repeats, strategies = cfg["repeats"], cfg["strategies"]
    extractor = cfg["models"][cfg["extractor_model"]]

    rows = []
    for model in args.models:
        spec = cfg["models"][model]
        tin = tout = 0.0
        for meta in idx.to_dict("records"):
            table = pd.read_csv(TABLES_DIR / f"{meta['table_id']}.csv")
            for strat, (t_in, t_out, calls) in per_note(table, meta).items():
                if strat in strategies:
                    tin += t_in * repeats
                    tout += (t_out + THINKING.get(model, 0) * calls) * repeats
        gen = tin * spec["price_in"] / 1e6 + tout * spec["price_out"] / 1e6
        n_notes = len(idx) * len(strategies) * repeats
        extract = n_notes * (EXTRACT_IN * extractor["price_in"] + EXTRACT_OUT * extractor["price_out"]) / 1e6
        rows.append({"model": model, "notes": n_notes, "input_Mtok": tin / 1e6, "output_Mtok": tout / 1e6,
                     "generation_usd": gen, "extractor_usd": extract, "total_usd": gen + extract})
    out = pd.DataFrame(rows)
    total = out["total_usd"].sum()
    print(f"Full run: {len(idx)} tables x {len(strategies)} strategies x {repeats} repeats per model "
          f"(claim extractor: {cfg['extractor_model']})")
    print(out.round(2).to_string(index=False))
    print(f"Estimated total: ${total:,.2f}. Budget cap in config: ${cfg['max_budget_usd']:,.2f}.")
    if total > cfg["max_budget_usd"]:
        print("The estimate is above the cap: raise MAX_BUDGET_USD for the full run, or the run stops at the cap "
              "(rerun with a higher cap to resume where it stopped).")


if __name__ == "__main__":
    main()
