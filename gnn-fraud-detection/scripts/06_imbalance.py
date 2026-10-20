r"""Etape 6: imbalance handling for GraphSAGE (the best GNN so far).
Run from the repo root:  .venv\Scripts\python.exe scripts\06_imbalance.py
Paste the printed table back to Claude.

Fixed: model = GraphSAGE, input dropout 0.5, 5 seeds.
Varied:
  - features: all 165  vs  local-only (first 93)
  - loss:     plain CE | full class weights | sqrt class weights | focal (gamma=2)

Selection rule (decided in advance): best mean VAL PR-AUC. Test is for reporting.
Val is nearly saturated, so treat small val differences as noise.
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import class_weights, get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate
from src.models import GNN
from src.train import train_gnn

SEEDS = [0, 1, 2, 3, 4]
N_LOCAL = 93
device = "cuda" if torch.cuda.is_available() else "cpu"

data = load_elliptic()
t = load_time_steps().numpy()
y = data.y.numpy()
tr, va, te = get_split(data)
tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
early = t[te_np] <= 42
w_full = class_weights(data.y[tr])
losses = {
    "plain": dict(),
    "weighted": dict(weight=w_full),
    "sqrt_weighted": dict(weight=w_full.sqrt()),
    "focal_g2": dict(focal_gamma=2.0),
}

rows = {}
for feats, loss_name in itertools.product(["all", "local"], losses):
    name = f"sage|{feats}|{loss_name}"
    d = data.clone()
    if feats == "local":
        d.x = data.x[:, :N_LOCAL]
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GNN(d.num_node_features, kind="sage", in_dropout=0.5)
        probs, val_pr = train_gnn(model, d, tr, va, device=device, **losses[loss_name])
        thr = best_threshold(y[va_np], probs[va_np])
        r = evaluate(y[te_np], probs[te_np], thr)
        r["val_pr_auc"] = val_pr
        r["f1_steps_35_42"] = evaluate(y[te_np][early], probs[te_np][early], thr)["f1"]
        runs.append(r)
    rows[name] = runs
    print(f"done {name}")

keys = ["val_pr_auc", "pr_auc", "f1", "precision", "recall", "f1_steps_35_42"]
print("\n=== Imbalance methods: mean +- std over 5 seeds (test metrics except val PR-AUC) ===")
print(f"{'config':<24}{'val PR':>8}{'test PR-AUC':>16}{'test F1':>16}{'prec':>7}{'recall':>8}{'F1 35-42':>10}")
summary = {}
for name, runs in rows.items():
    m = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}
    summary[name] = m
    print(f"{name:<24}{m['val_pr_auc'][0]:>8.3f}"
          f"{m['pr_auc'][0]:>9.3f}+-{m['pr_auc'][1]:.3f}"
          f"{m['f1'][0]:>9.3f}+-{m['f1'][1]:.3f}"
          f"{m['precision'][0]:>7.3f}{m['recall'][0]:>8.3f}{m['f1_steps_35_42'][0]:>10.3f}")

winner = max(summary, key=lambda k: summary[k]["val_pr_auc"][0])
print(f"\nWinner by VAL PR-AUC: {winner}")

os.makedirs("results", exist_ok=True)
with open("results/imbalance_metrics.json", "w") as f:
    json.dump({"runs": rows, "summary": summary, "winner": winner}, f, indent=2)
print("Saved results/imbalance_metrics.json")
