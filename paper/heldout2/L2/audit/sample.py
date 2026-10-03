from pathlib import Path
import json, numpy as np, pandas as pd
B = str(Path(__file__).resolve().parents[1]) + "/"
t = pd.read_csv(B + "truth.csv")
WRONG = {"Wrong_value", "Wrong_entity", "Wrong_metric", "Wrong_direction", "Group_wrong"}
t["stratum"] = np.where(t.truth.isin(WRONG), "wrong", "other")
rng = np.random.default_rng(20260928)
w = t.index[t.stratum == "wrong"].to_numpy()
o = t.index[t.stratum == "other"].to_numpy()
sw = rng.choice(w, 25, replace=False)
so = rng.choice(o, 45, replace=False)
s = t.loc[sorted(np.concatenate([sw, so]))].copy()
# blind copy: only identifying and claim fields
blind = s[["note_id", "table_id", "claim_idx", "line", "entity", "metric", "period", "value_text", "direction", "qualifier", "stratum"]]
blind.to_csv(B + "audit/sample_blind.csv")
s.to_csv(B + "audit/sample_with_truth.csv")
print(len(s), s.stratum.value_counts().to_dict())
