r"""Etape 7b: figures for the report.
Run from the repo root:  .venv\Scripts\python.exe scripts\08_plots.py

Produces:
  results/per_step_f1.png   illicit F1 per test step, step 43 marked
  results/pr_curves.png     test precision-recall curves with average precision

No probabilities were saved by the training scripts, so the three plotted models
are retrained here:
  - XGBoost: same hyperparameters as 03_baseline.py, random_state=42 (its seed);
    seed 0 would no longer match baseline_metrics.json.
  - GNNs: torch/numpy seed 0, trained with src/train.py, in_dropout=0.5.

Threshold for the F1 panel is picked on VAL only, exactly as in the other scripts.
The SAGE-local curve is included because it is the best GNN on test, but it is
labelled as NOT the VAL-selected config: selection stays VAL-only.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate, per_step_f1
from src.models import GNN
from src.train import train_gnn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_LOCAL = 93
GNN_SEED = 0
XGB_SEED = 42


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def step_delta(new, saved):
    """Largest |new - saved| over steps where both have a value."""
    diffs = []
    for k, v in saved.items():
        nv = new.get(int(k))
        if v is not None and nv is not None:
            diffs.append(abs(nv - v))
    return max(diffs) if diffs else float("nan")


def main():
    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    steps = np.unique(t[te_np])
    X = data.x.numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}; test steps {steps[0]}-{steps[-1]}")

    def train_xgb():
        n_neg = int((y[tr_np] == 0).sum())
        n_pos = int((y[tr_np] == 1).sum())
        model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float(n_neg / n_pos),
            eval_metric="aucpr", early_stopping_rounds=50,
            random_state=XGB_SEED, n_jobs=-1, device=device, tree_method="hist",
        )
        model.fit(X[tr_np], y[tr_np],
                  eval_set=[(X[va_np], y[va_np])], verbose=False)
        return (model.predict_proba(X[va_np])[:, 1],
                model.predict_proba(X[te_np])[:, 1])

    def train_sage(feats):
        d = data.clone()
        if feats == "local":
            d.x = data.x[:, :N_LOCAL]
        torch.manual_seed(GNN_SEED)
        np.random.seed(GNN_SEED)
        model = GNN(d.num_node_features, kind="sage", in_dropout=0.5)
        probs, val_pr = train_gnn(model, d, tr, va, device=device)
        return probs[va_np], probs[te_np], val_pr

    def summarize(p_val, p_test, val_pr, color, marker):
        thr = best_threshold(y[va_np], p_val)  # VAL only
        r = evaluate(y[te_np], p_test, thr)
        return {
            "thr": thr, "p_test": p_test, "pr_auc": r["pr_auc"],
            "f1": r["f1"], "val_pr": float(val_pr),
            "per_step": per_step_f1(y[te_np], p_test, t[te_np], thr),
            "color": color, "marker": marker,
        }

    p_val, p_test = train_xgb()
    xgb_model = summarize(p_val, p_test,
                          average_precision_score(y[va_np], p_val),
                          "#444444", "s")
    xgb_model["label"] = "XGBoost"

    p_val, p_test, val_pr = train_sage("all")
    sage_all = summarize(p_val, p_test, val_pr, "#1f77b4", "o")
    sage_all["label"] = "SAGE all"

    p_val, p_test, val_pr = train_sage("local")
    sage_local = summarize(p_val, p_test, val_pr, "#2ca02c", "^")
    sage_local["label"] = "SAGE local†"

    models = [xgb_model, sage_all, sage_local]

    # Cross-check the retrained models against the saved JSONs.
    base = load("baseline_metrics.json")["xgb_all_weighted"]
    gnn0 = load("gnn_metrics.json")["runs"]["sage_plain"][0]
    abl0 = load("gnn_ablation.json")["runs"]["sage|local|indrop=0.5"][0]
    saved = [("XGBoost all/weighted", base),
             ("SAGE all/plain seed0", gnn0),
             ("SAGE local/plain seed0", abl0)]
    print("\n== retrained vs saved ==")
    for (name, ref), m in zip(saved, models):
        print(f"{name:<22} PR-AUC {m['pr_auc']:.3f} vs {ref['pr_auc']:.3f} "
              f"({m['pr_auc'] - ref['pr_auc']:+.3f})   "
              f"F1 {m['f1']:.3f} vs {ref['f1']:.3f} "
              f"({m['f1'] - ref['f1']:+.3f})")

    d_xgb = step_delta(xgb_model["per_step"], base["per_step_f1"])
    d_all = step_delta(sage_all["per_step"], gnn0["per_step_f1"])
    print(f"per-step F1 max |delta|: xgb {d_xgb:.4f}, sage_all {d_all:.4f}")
    print(f"thresholds: xgb {xgb_model['thr']:.2f} "
          f"(saved {base['threshold']:.2f}), "
          f"sage_all {sage_all['thr']:.2f} (saved {gnn0['threshold']:.2f})")
    if max(d_xgb, d_all) > 0.02:
        print("WARNING: retrained per-step F1 differs from saved by >0.02; "
              "check randomness before trusting the figure.")
    if any(abs(m["pr_auc"] - ref["pr_auc"]) > 0.01
           for (_, ref), m in zip(saved, models)):
        print("WARNING: a retrained PR-AUC differs from the saved value by >0.01.")

    # --- Figure 1: per-time-step test F1 -------------------------------
    note = (f"† SAGE local: best GNN on test, not the VAL pick "
            f"(VAL PR-AUC {sage_local['val_pr']:.3f} vs "
            f"{sage_all['val_pr']:.3f} for SAGE all).")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvspan(42.5, 49.5, color="0.9", zorder=0)
    for m in models:
        xs = [int(s) for s in steps if m["per_step"].get(int(s)) is not None]
        ys = [m["per_step"][int(s)] for s in xs]
        ax.plot(xs, ys, color=m["color"], marker=m["marker"], ms=3.5, lw=1.7,
                label=f"{m['label']} (AP {m['pr_auc']:.3f})")
    ax.axvline(43, color="crimson", ls="--", lw=1.2, label="step 43")
    ax.text(46, 0.95, "shutdown\n(43–49)", ha="center", va="top",
            fontsize=8, color="dimgray")
    ax.set_xlabel("test time step")
    ax.set_ylabel("F1 (illicit class)")
    ax.set_title("Per-time-step test F1: XGBoost vs SAGE (threshold picked on VAL)")
    ax.set_xlim(34.5, 49.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(steps)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "per_step_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: test precision-recall ------------------------------
    prev = float(y[te_np].mean())
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for m in models:
        prec, rec, _ = precision_recall_curve(y[te_np], m["p_test"])
        ax.plot(rec, prec, color=m["color"], lw=1.9,
                label=f"{m['label']} (AP {m['pr_auc']:.3f})")
    ax.axhline(prev, color="grey", ls=":", lw=1.2,
               label=f"chance (test prevalence {prev:.1%})")
    ax.set_xlabel("recall (illicit)")
    ax.set_ylabel("precision (illicit)")
    ax.set_title("Test precision-recall curves (steps 35–49)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {RESULTS / 'per_step_f1.png'}")
    print(f"Saved {RESULTS / 'pr_curves.png'}")


if __name__ == "__main__":
    main()
