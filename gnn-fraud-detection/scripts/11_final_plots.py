r"""Final figures for the report (supersedes 08_plots.py).
Run from the repo root:  .venv\Scripts\python.exe scripts\11_final_plots.py

Plots:
  results/per_step_f1.png   illicit F1 per test step 35-49, step 43 marked
  results/pr_curves.png     test precision-recall curves

Models:
  - XGBoost new best: local 93 + degree + neighbour-contrast, unweighted
  - XGBoost previous: all 165, class weights
  - SAGE local 93, plain loss, 3 seeds averaged

Probabilities are not saved by the training scripts, so all three are retrained
here (XGB seed 42, GNN seeds 0-2). Thresholds are picked on VAL only.
"""
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
XGB_SEED = 42
XGB_SEEDS = [42, 43, 44, 45, 46]
SEEDS = [0, 1, 2]


def features(data):
    X = data.x.numpy()
    src, dst = data.edge_index.numpy()
    n = data.num_nodes
    xl = X[:, :N_LOCAL]
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)
    deg = np.column_stack([np.log1p(out_deg), np.log1p(in_deg)]).astype(np.float32)
    neigh_sum = np.zeros_like(xl)
    np.add.at(neigh_sum, dst, xl[src])
    np.add.at(neigh_sum, src, xl[dst])
    counts = np.bincount(dst, minlength=n) + np.bincount(src, minlength=n)
    neigh_mean = neigh_sum / np.maximum(counts, 1)[:, None]
    contrast = (xl - neigh_mean).astype(np.float32)
    return X, np.hstack([xl, deg, contrast]).astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    X, X_new = features(data)
    yv = y[va_np]
    trva = np.concatenate([tr_np, va_np])

    def xgb_ref():
        model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", early_stopping_rounds=50,
            random_state=XGB_SEED, n_jobs=-1, device=device, tree_method="hist")
        model.fit(X_new[tr_np], y[tr_np],
                  eval_set=[(X_new[va_np], y[va_np])], verbose=False)
        return (model.predict_proba(X_new[va_np])[:, 1],
                model.predict_proba(X_new[te_np])[:, 1], model)

    def xgb_final(n_trees):
        ps = []
        for seed in XGB_SEEDS:
            m = xgb.XGBClassifier(
                n_estimators=n_trees, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=seed,
                n_jobs=-1, device=device, tree_method="hist",
                eval_metric="aucpr")
            m.fit(X_new[trva], y[trva], verbose=False)
            ps.append(m.predict_proba(X_new[te_np])[:, 1])
        return np.mean(ps, axis=0)

    def sage_local():
        d = data.clone()
        d.x = torch.tensor(X[:, :N_LOCAL], dtype=torch.float32)
        pv, pt = [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GNN(d.num_node_features, kind="sage", in_dropout=0.5)
            probs, _ = train_gnn(model, d, tr, va, device=device)
            pv.append(probs[va_np])
            pt.append(probs[te_np])
        return np.mean(pv, axis=0), np.mean(pt, axis=0)

    p_ref_val, p_ref_test, ref_model = xgb_ref()
    thr_ref = best_threshold(yv, p_ref_val)
    n_final = int(ref_model.best_iteration) + 1
    p_final = xgb_final(n_final)
    p_sage_val, p_sage_test = sage_local()

    specs = [
        ("XGBoost final fit (1-34, 5 seeds)", None, p_final, "#1f77b4", "-", thr_ref),
        ("XGBoost train-only (1-29)", p_ref_val, p_ref_test, "#444444", "--", thr_ref),
        ("SAGE local 93 (3 seeds)", p_sage_val, p_sage_test, "#2ca02c", "-", None),
    ]

    models = []
    for label, pv, pt, color, ls, fixed_thr in specs:
        thr = fixed_thr if fixed_thr is not None else best_threshold(yv, pv)
        r = evaluate(y[te_np], pt, thr)
        models.append({"label": label, "color": color, "ls": ls,
                       "p_test": pt, "pr_auc": r["pr_auc"], "f1": r["f1"],
                       "thr": thr,
                       "per_step": per_step_f1(y[te_np], pt, t[te_np], thr)})
        print(f"{label:<42} val threshold {thr:.2f} | "
              f"test PR {r['pr_auc']:.4f} | F1 {r['f1']:.4f}", flush=True)

    steps = np.unique(t[te_np])
    note = ("Threshold from VAL. XGB final fit: steps 1-34, 332 trees, 5 seeds, "
            "threshold from the train-only model. XGB train-only: 1-29. "
            "SAGE local seed-averaged.")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvspan(42.5, 49.5, color="0.9", zorder=0)
    for m in models:
        xs = [int(s) for s in steps if m["per_step"].get(int(s)) is not None]
        ys = [m["per_step"][int(s)] for s in xs]
        ax.plot(xs, ys, color=m["color"], ls=m["ls"], marker="o", ms=3.5,
                lw=1.7, label=f"{m['label']} (AP {m['pr_auc']:.3f})")
    ax.axvline(43, color="crimson", ls="--", lw=1.2, label="step 43")
    ax.text(46, 0.95, "shutdown\n(43–49)", ha="center", va="top",
            fontsize=8, color="dimgray")
    ax.set_xlabel("test time step")
    ax.set_ylabel("F1 (illicit class)")
    ax.set_title("Per-time-step test F1 (threshold picked on VAL)")
    ax.set_xlim(34.5, 49.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(steps)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "per_step_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    prev = float(y[te_np].mean())
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for m in models:
        prec, rec, _ = precision_recall_curve(y[te_np], m["p_test"])
        ax.plot(rec, prec, color=m["color"], ls=m["ls"], lw=1.9,
                label=f"{m['label']} (AP {m['pr_auc']:.3f})")
    ax.axhline(prev, color="grey", ls=":", lw=1.2,
               label=f"chance (test prevalence {prev:.1%})")
    ax.set_xlabel("recall (illicit)")
    ax.set_ylabel("precision (illicit)")
    ax.set_title("Test precision-recall curves (steps 35–49)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved {RESULTS / 'per_step_f1.png'}")
    print(f"Saved {RESULTS / 'pr_curves.png'}")


if __name__ == "__main__":
    main()
