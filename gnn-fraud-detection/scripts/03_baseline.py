r"""Etape 4: XGBoost baseline (no graph structure).
Run from the repo root:  .venv\Scripts\python.exe scripts\03_baseline.py
Paste the printed table back to Claude.

Four variants:
  - all features vs local-only features
  - no class weighting vs scale_pos_weight

Why local-only matters: the dataset's aggregated features already summarize each
node's one-hop neighborhood, so "all features" is a baseline that already sees
some graph information. Local-only is the cleanest "no graph at all" baseline.
"""
import json
import os
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import ILLICIT, LICIT, get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate, per_step_f1

SEED = 42
N_LOCAL = 93  # first 93 columns of x = local features (the rest are aggregated)

data = load_elliptic()
t = load_time_steps().numpy()
tr, va, te = [i.numpy() for i in get_split(data)]
X, y = data.x.numpy(), data.y.numpy()

results = {}


def run(name, cols, weighted):
    n_neg, n_pos = (y[tr] == LICIT).sum(), (y[tr] == ILLICIT).sum()
    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=float(n_neg / n_pos) if weighted else 1.0,
        eval_metric="aucpr", early_stopping_rounds=50,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X[tr][:, cols], y[tr], eval_set=[(X[va][:, cols], y[va])], verbose=False)

    p_val = model.predict_proba(X[va][:, cols])[:, 1]
    thr = best_threshold(y[va], p_val)  # threshold chosen on val only
    p_test = model.predict_proba(X[te][:, cols])[:, 1]

    res = evaluate(y[te], p_test, thr)
    res["best_iteration"] = int(model.best_iteration)
    res["val_f1_at_thr"] = evaluate(y[va], p_val, thr)["f1"]
    early = t[te] <= 42
    res["f1_steps_35_42"] = evaluate(y[te][early], p_test[early], thr)["f1"]  # pooled, same as GNN scripts
    res["per_step_f1"] = per_step_f1(y[te], p_test, t[te], thr)
    results[name] = res


run("xgb_all_noweight", slice(None), False)
run("xgb_all_weighted", slice(None), True)
run("xgb_local_noweight", slice(0, N_LOCAL), False)
run("xgb_local_weighted", slice(0, N_LOCAL), True)

print("=== XGBoost baselines: TEST metrics (illicit class) ===")
print(f"{'model':<22}{'prec':>7}{'recall':>8}{'F1':>7}{'PR-AUC':>8}{'thr':>6}{'val F1':>8}{'iters':>7}{'F1 35-42':>10}")
for name, r in results.items():
    print(f"{name:<22}{r['precision']:>7.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}"
          f"{r['pr_auc']:>8.3f}{r['threshold']:>6.2f}{r['val_f1_at_thr']:>8.3f}{r['best_iteration']:>7d}{r['f1_steps_35_42']:>10.3f}")

print("\n=== Per-time-step test F1 (variant with best VAL F1) ===")
best = max(results, key=lambda k: results[k]["val_f1_at_thr"])  # chosen on VAL, not test
print(f"variant: {best}")
print({s: (round(v, 2) if v is not None else None) for s, v in results[best]["per_step_f1"].items()})

os.makedirs("results", exist_ok=True)
with open("results/baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/baseline_metrics.json")
