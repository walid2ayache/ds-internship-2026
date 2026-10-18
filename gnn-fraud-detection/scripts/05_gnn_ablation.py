r"""Etape 5b: ablation to find out whether our GNN setup handicaps the GNNs.
Run from the repo root:  .venv\Scripts\python.exe scripts\05_gnn_ablation.py
Paste the printed table back to Claude.

Varies, with plain (unweighted) loss, 3 seeds each:
  - model:         sage, gcn
  - features:      all 165  vs  local-only (first 93)   [double-smoothing test]
  - input dropout: 0.5     vs  0.0                     [too-aggressive-dropout test]

Selection rule: the config with the best mean VAL PR-AUC is "the winner".
Test numbers are shown for reporting, not for choosing.
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate
from src.models import GNN
from src.train import train_gnn

SEEDS = [0, 1, 2]
N_LOCAL = 93
PRE_SHUTDOWN_LAST_STEP = 42  # steps 35-42 vs 43-49 reported separately
device = "cuda" if torch.cuda.is_available() else "cpu"

data = load_elliptic()
t = load_time_steps().numpy()
y = data.y.numpy()
tr, va, te = get_split(data)
tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
early = t[te_np] <= PRE_SHUTDOWN_LAST_STEP

rows = {}
for kind, feats, in_drop in itertools.product(["sage", "gcn"], ["all", "local"], [0.5, 0.0]):
    name = f"{kind}|{feats}|indrop={in_drop}"
    d = data.clone()
    if feats == "local":
        d.x = data.x[:, :N_LOCAL]
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GNN(d.num_node_features, kind=kind, in_dropout=in_drop)
        probs, val_pr = train_gnn(model, d, tr, va, device=device)
        thr = best_threshold(y[va_np], probs[va_np])
        r = evaluate(y[te_np], probs[te_np], thr)
        r["val_pr_auc"] = val_pr
        r["f1_steps_35_42"] = evaluate(y[te_np][early], probs[te_np][early], thr)["f1"]
        runs.append(r)
    rows[name] = runs
    print(f"done {name}")

keys = ["val_pr_auc", "pr_auc", "f1", "f1_steps_35_42"]
print("\n=== Ablation: mean over seeds (test metrics except val PR-AUC) ===")
print(f"{'config':<26}{'val PR-AUC':>11}{'test PR-AUC':>12}{'test F1':>9}{'F1 35-42':>10}")
summary = {}
for name, runs in rows.items():
    m = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    summary[name] = m
    print(f"{name:<26}{m['val_pr_auc']:>11.3f}{m['pr_auc']:>12.3f}{m['f1']:>9.3f}{m['f1_steps_35_42']:>10.3f}")

winner = max(summary, key=lambda k: summary[k]["val_pr_auc"])
print(f"\nWinner by VAL PR-AUC: {winner}")

os.makedirs("results", exist_ok=True)
with open("results/gnn_ablation.json", "w") as f:
    json.dump({"runs": rows, "summary": summary, "winner": winner}, f, indent=2)
print("Saved results/gnn_ablation.json")
