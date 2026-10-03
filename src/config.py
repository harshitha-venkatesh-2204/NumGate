"""Paths and config loading shared by every script."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # repo root, so scripts work from any cwd
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
TABLES_DIR = ROOT / "data" / "tables"
PROMPTS_DIR = ROOT / "prompts"
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"


def load_config(path=None):
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_prompt(name):
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")
