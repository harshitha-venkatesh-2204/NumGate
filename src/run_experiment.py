"""Run every (table, strategy, model, repeat) condition, then gate every note.

Usage:
  python src/run_experiment.py --smoke            # 2 tables per size, mock model, run id "smoke"
  python src/run_experiment.py                    # full config, run id "full"
  python src/run_experiment.py --models claude-opus-5 --tables-per-size 2 --run-id smoke_opus
Resumable: notes already saved under runs/<run_id>/notes/ are not regenerated.
"""
import argparse
import itertools
import json
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import models
import number_gate
from config import ROOT, RUNS_DIR, TABLES_DIR, load_config
import strategies as strategies_mod
from strategies import STRATEGIES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="2 tables per size with the mock model only")
    p.add_argument("--run-id", default=None)
    p.add_argument("--models", nargs="+", default=None, help="override experiment_models from config.yaml")
    p.add_argument("--strategies", nargs="+", default=None)
    p.add_argument("--tables-per-size", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--gate-only", action="store_true", help="skip generation, only re-run the gate")
    p.add_argument("--workers", type=int, default=None, help="parallel API calls (default: config workers)")
    return p.parse_args()


def select_tables(cfg, per_size):
    """First per_size tables of each size after a seeded shuffle, so small runs still mix entity types."""
    idx = pd.read_csv(TABLES_DIR / "index.csv")
    picked = [g.sample(frac=1, random_state=cfg["seed"]).head(per_size) for _, g in idx.groupby("size")]
    idx = pd.concat(picked)
    return idx[idx["size"].isin(cfg["sizes"])].reset_index(drop=True)


def load_table(table_id):
    return pd.read_csv(TABLES_DIR / f"{table_id}.csv")


def write_json_atomic(path, record):
    """Write to a temp file, then rename, so a killed run never leaves a half-written note behind."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def generate_one(meta, strategy, model, repeat, path, stop):
    """Generate and save one note. Returns (status, trace or error text)."""
    if stop.is_set():
        return "cancelled", ""
    table = load_table(meta["table_id"])
    meta = dict(meta, csv_path=str(TABLES_DIR / f"{meta['table_id']}.csv"))
    try:
        note, trace = STRATEGIES[strategy](table, meta, model, repeat)
    except models.BudgetExceeded as err:
        stop.set()  # tell every other worker to stop starting new notes
        return "budget", str(err)
    except Exception as err:  # provider or network error: report and move on, a rerun retries it
        return "error", f"{type(err).__name__}: {err}"
    record = {"note_id": path.stem, **{k: meta[k] for k in ["table_id", "size", "entity_type", "week", "n_rows"]},
              "strategy": strategy, "model": model, "repeat": repeat, "note": note, "trace": trace}
    write_json_atomic(path, record)
    return "done", trace


def generate(run_dir, tables, strategies, model_names, repeats, workers):
    notes_dir = run_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(itertools.product(tables.to_dict("records"), strategies, model_names, range(1, repeats + 1)))
    todo = []  # conditions without a saved note
    for meta, strategy, model, repeat in conditions:
        path = notes_dir / f"{meta['table_id']}__{strategy}__{model}__r{repeat}.json"
        if not path.exists():
            todo.append((meta, strategy, model, repeat, path))
    print(f"{len(conditions)} conditions: {len(conditions) - len(todo)} already saved, {len(todo)} to generate "
          f"with {workers} parallel workers")
    stop = threading.Event()  # set on a budget stop
    counts = {"done": 0, "error": 0, "budget": 0, "cancelled": 0}
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(generate_one, *c, stop): c[4].stem for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            status, info = fut.result()
            counts[status] += 1
            name = futures[fut]
            if status == "done":
                tag = " FAILED (S2 script)" if info["failed"] else ""
                print(f"[{i}/{len(todo)}] {name}  {info['n_calls']} calls  ${info['cost_usd']:.3f}  "
                      f"{info['latency_s']:.0f}s{tag}   billed so far ${models.SPENT['usd']:.2f}", flush=True)
            elif status == "error":
                print(f"[{i}/{len(todo)}] {name}  ERROR {info}", flush=True)
            elif status == "budget":
                print(f"[{i}/{len(todo)}] {name}  BUDGET STOP: {info}", flush=True)
    except BaseException:  # Ctrl-C: stop starting new notes; saved notes are kept and a rerun resumes
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        print(f"Interrupted after {counts['done']} notes; rerun the same command to resume.")
        raise
    pool.shutdown(wait=True)
    if counts["budget"]:
        print("Budget cap reached. Saved notes are kept; raise MAX_BUDGET_USD and rerun to resume.")
    print(f"Generated {counts['done']} notes, {counts['error']} errors, {counts['cancelled']} not started. "
          f"Billed this session: ${models.SPENT['usd']:.4f}")


def missing_credentials(model_names, cfg):
    """Names of providers used by this run whose credentials are not set."""
    needs = {"openai": ["OPENAI_API_KEY"], "openrouter": ["OPENROUTER_API_KEY"],
             "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]}
    names = set(model_names)
    if cfg.get("extractor_model", "regex") != "regex" and any(m != "mock" for m in model_names):
        names.add(cfg["extractor_model"])
    missing = []
    for name in sorted(names):
        provider = cfg["models"][name]["provider"]
        if provider == "mock":
            continue
        has_profile = provider == "anthropic" and (Path.home() / ".config" / "anthropic").exists()
        if not has_profile and not any(os.environ.get(v) for v in needs[provider]):
            missing.append(f"{name} needs {' or '.join(needs[provider])}")
    return missing


def format_check(note):
    """Does the note follow the shared format: 6 to 8 bullets, each with a number, no em dash?"""
    lines = [l for l in note.splitlines() if l.strip()]
    bullets = [l for l in lines if l.startswith("- ")]
    with_number = sum(bool(re.search(r"\d", l)) for l in bullets)
    return {"n_lines": len(lines), "n_bullets": len(bullets),
            "format_ok": 6 <= len(bullets) <= 8 and with_number == len(bullets) == len(lines) and "\u2014" not in note}


def extract_claims(note, table, model, cfg):
    """LLM extractor for real models, regex extractor for the mock (or if extractor_model is 'regex')."""
    extractor = cfg.get("extractor_model", "regex")
    if model == "mock" or extractor == "regex":
        return number_gate.regex_claims(note, table), "regex", 0.0
    claims, resp = number_gate.llm_claims(note, table, extractor, models.complete)
    if claims is None:  # never mix the rule-based extractor into a real model's claim rates
        return None, "extract_failed", resp["cost_usd"] if resp else 0.0
    return claims, extractor, resp["cost_usd"]


def extract_safe(rec, cfg):
    """Claims for one note; never raises, so one bad note cannot stop the whole gate pass."""
    if rec["trace"]["failed"]:
        return [], "", 0.0
    try:
        return extract_claims(rec["note"], load_table(rec["table_id"]), rec["model"], cfg)
    except models.BudgetExceeded:
        return None, "budget_stop", 0.0
    except Exception as err:  # extractor API error: leave this note out of claim rates and say so
        print(f"Extractor error on {rec['note_id']}: {type(err).__name__}: {err}")
        return None, "extract_failed", 0.0


def gate_all(run_dir, cfg, workers=4):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "notes").glob("*.json"))]
    # Claim extraction calls a model for real runs, so run it in parallel; results are cached for reruns.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extracted = list(pool.map(lambda r: extract_safe(r, cfg), records))
    failed = sum(e[1] == "extract_failed" for e in extracted)
    if failed:
        print(f"Claim extraction failed for {failed} notes; they are excluded from claim rates (extractor=extract_failed).")
    stopped = sum(e[1] == "budget_stop" for e in extracted)
    if stopped:
        print(f"Budget cap reached during claim extraction: {stopped} notes have no claims yet. "
              "Raise MAX_BUDGET_USD and rerun with --gate-only.")
    number_rows, claim_rows, note_rows = [], [], []
    for rec, (raw_claims, extractor, extract_cost) in zip(records, extracted):
        table = load_table(rec["table_id"])
        trace = rec["trace"]
        keys = {k: rec[k] for k in ["note_id", "table_id", "size", "entity_type", "strategy", "model", "repeat"]}
        numbers, claims = [], []
        if not trace["failed"]:
            numbers = number_gate.gate_level1(rec["note"], table, rec["note_id"])
            if raw_claims is not None:
                claims = number_gate.gate_level2(rec["note"], table, rec["note_id"], raw_claims)
        number_rows += [{**keys, **{k: v for k, v in r.items() if k != "note_id"}} for r in numbers]
        claim_rows += [{**keys, **{k: v for k, v in r.items() if k != "note_id"}} for r in claims]
        summary = number_gate.summarize(numbers, claims)
        rounds = trace.get("rounds", [])
        round_rates = []  # re-gated with the current Level 1, so a gate change never leaves stale S3 rates
        for r in rounds:
            flagged, checked = strategies_mod.flagged_numbers(r["note"], table)
            round_rates.append(len(flagged) / checked if checked else None)
        note_rows.append({**keys, "week": rec["week"], "n_rows": rec["n_rows"], "failed": trace["failed"],
                          **format_check(rec["note"]), **summary,
                          "tokens_in": trace["tokens_in"], "tokens_out": trace["tokens_out"],
                          "cost_usd": trace["cost_usd"], "latency_s": trace["latency_s"], "n_calls": trace["n_calls"],
                          "extractor": extractor, "extractor_cost_usd": extract_cost,
                          "s3_rounds_used": max(len(rounds) - 1, 0) if rounds else None,
                          "s3_round_rates": json.dumps(round_rates) if rounds else "",
                          "s2_code_attempts": len(trace.get("code_runs", [])) or None})
    pd.DataFrame(number_rows).to_csv(run_dir / "numbers.csv", index=False)
    pd.DataFrame(claim_rows).to_csv(run_dir / "claims.csv", index=False)
    notes = pd.DataFrame(note_rows)
    notes.to_csv(run_dir / "notes.csv", index=False)
    return notes


def print_summary(notes):
    if notes.empty:
        print("No notes in this run yet.")
        return
    ok = notes[~notes["failed"]]
    g = ok.groupby("strategy")
    table = pd.DataFrame({
        "notes": notes.groupby("strategy").size(),
        "failed": notes.groupby("strategy")["failed"].sum(),
        "numbers_checked": g["n_numbers_checked"].sum(),
        "unsupported_rate": g["n_Unsupported"].sum() / g["n_numbers_checked"].sum(),
        "claims_verifiable": g["n_claims_verifiable"].sum(),
        "binding_error_rate": (g["n_Wrong_entity"].sum() + g["n_Wrong_metric"].sum()) / g["n_claims_verifiable"].sum(),
        "claim_error_rate": (g["n_Wrong_entity"].sum() + g["n_Wrong_metric"].sum() + g["n_Wrong_value"].sum())
        / g["n_claims_verifiable"].sum(),
        "format_ok": g["format_ok"].mean(),
        "cost_per_note": g["cost_usd"].mean(),
    })
    print(table.round(4).to_string())


def main():
    args = parse_args()
    cfg = load_config()
    model_names = args.models or (["mock"] if args.smoke else cfg["experiment_models"])
    per_size = args.tables_per_size or (2 if args.smoke else cfg["tables_per_size"])
    strategies = args.strategies or cfg["strategies"]
    repeats = args.repeats or cfg["repeats"]
    run_id = args.run_id or ("smoke" if args.smoke else "full")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = run_dir / "run_settings.json"
    if args.gate_only and saved.exists():  # re-scoring must not relabel the run with this command's defaults
        settings = json.loads(saved.read_text(encoding="utf-8"))
        print(f"Re-scoring run {run_id} ({', '.join(settings['models'])}) without generating notes")
        notes = gate_all(run_dir, cfg, args.workers or cfg.get("workers", 4))
        print(f"\nWrote {run_dir.relative_to(ROOT)}/notes.csv ({len(notes)} notes), numbers.csv, claims.csv\n")
        print_summary(notes)
        return
    shutil.copy(ROOT / "config.yaml", run_dir / "config_used.yaml")
    settings = {"run_id": run_id, "models": model_names, "strategies": strategies, "tables_per_size": per_size,
                "repeats": repeats, "budget_usd": models.budget_limit()}
    (run_dir / "run_settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    print(f"Run {run_id}: models={model_names} strategies={strategies} tables/size={per_size} repeats={repeats}")

    tables = select_tables(cfg, per_size)
    missing = missing_credentials(model_names, cfg)
    if missing and not args.gate_only:
        print("Missing credentials, nothing was called:\n  " + "\n  ".join(missing))
        sys.exit(1)
    if not args.gate_only:
        generate(run_dir, tables, strategies, model_names, repeats, args.workers or cfg.get("workers", 4))
    notes = gate_all(run_dir, cfg, args.workers or cfg.get("workers", 4))
    print(f"\nWrote {run_dir.relative_to(ROOT)}/notes.csv ({len(notes)} notes), numbers.csv, claims.csv\n")
    print_summary(notes)


if __name__ == "__main__":
    main()
