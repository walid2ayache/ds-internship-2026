r"""Compare the CV-tuned XGB against the current best, with paired bootstrap CIs.
Run from the repo root:  .venv\Scripts\python.exe scripts\13_compare.py

Both models are retrained here so the comparison is on identical data:
  - tuned: base 188 + top-32 aggregated features, depth 7, 300 trees, 5 seeds averaged
  - current: base 188, depth 6, early stopping on val, seed 42 (as in 10_boost.py)

Paired node bootstrap over the test set gives CIs for delta PR-AUC and delta F1 at each
model's own VAL threshold. Node bootstrap ignores graph/time dependence: indicative only.
"""
import json
import os
import sys

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import get_split, load_elliptic, load_time_steps
from src.metrics import best_threshold, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
N_LOCAL = 93
SEEDS = [42, 43, 44, 45, 46]


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
    return X[:, :N_LOCAL], deg, contrast, X[:, N_LOCAL:]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    tr, va, te = get_split(data)
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    xl, deg, contrast, agg = features(data)
    base = np.hstack([xl, deg, contrast]).astype(np.float32)

    ranker = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, device=device, tree_method="hist", eval_metric="aucpr")
    ranker.fit(np.hstack([base, agg])[tr_np], y[tr_np], verbose=False)
    order = np.argsort(ranker.feature_importances_[-agg.shape[1]:])[::-1]
    tuned_X = np.hstack([base, agg[:, order[:32]]]).astype(np.float32)

    def tuned_model(seed):
        m = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7,
            min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, device=device,
            tree_method="hist", eval_metric="aucpr")
        m.fit(tuned_X[tr_np], y[tr_np], verbose=False)
        return m

    pv, pt = [], []
    for seed in SEEDS:
        m = tuned_model(seed)
        pv.append(m.predict_proba(tuned_X[va_np])[:, 1])
        pt.append(m.predict_proba(tuned_X[te_np])[:, 1])
    p_new_val, p_new = np.mean(pv, axis=0), np.mean(pt, axis=0)

    prev = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", early_stopping_rounds=50, random_state=42,
        n_jobs=-1, device=device, tree_method="hist")
    prev.fit(base[tr_np], y[tr_np],
             eval_set=[(base[va_np], y[va_np])], verbose=False)
    p_old_val = prev.predict_proba(base[va_np])[:, 1]
    p_old = prev.predict_proba(base[te_np])[:, 1]

    def report(name, p_val, p_test):
        thr = best_threshold(y[va_np], p_val)
        r = evaluate(y[te_np], p_test, thr)
        print(f"{name:<10} thr {thr:.2f} | val AP {average_precision_score(y[va_np], p_val):.4f} "
              f"| test PR {r['pr_auc']:.4f} | F1 {r['f1']:.4f} "
              f"| P {r['precision']:.3f} R {r['recall']:.3f}", flush=True)
        return thr, r

    thr_new, r_new = report("tuned", p_new_val, p_new)
    thr_old, r_old = report("current", p_old_val, p_old)

    def f1_at(y_true, p, thr):
        pred = p >= thr
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    def boot(y_true, pa, pb, ta, tb, n=1000):
        rng = np.random.default_rng(0)
        dpr, df1 = [], []
        for _ in range(n):
            idx = rng.integers(0, len(y_true), len(y_true))
            yy = y_true[idx]
            if yy.sum() < 2 or yy.sum() == len(yy):
                continue
            dpr.append(average_precision_score(yy, pa[idx]) -
                       average_precision_score(yy, pb[idx]))
            df1.append(f1_at(yy, pa[idx], ta) - f1_at(yy, pb[idx], tb))
        dpr, df1 = np.array(dpr), np.array(df1)
        return (dpr.mean(), np.percentile(dpr, 2.5), np.percentile(dpr, 97.5),
                df1.mean(), np.percentile(df1, 2.5), np.percentile(df1, 97.5))

    pr_m, pr_lo, pr_hi, f1_m, f1_lo, f1_hi = boot(
        y[te_np], p_new, p_old, thr_new, thr_old)
    print(f"\ndelta PR-AUC (tuned - current): {pr_m:+.4f} "
          f"[95% {pr_lo:+.4f}, {pr_hi:+.4f}]", flush=True)
    print(f"delta F1 at each VAL threshold: {f1_m:+.4f} "
          f"[95% {f1_lo:+.4f}, {f1_hi:+.4f}]", flush=True)

    payload = {"tuned": {"threshold": thr_new, **{k: r_new[k] for k in
                                                    ("pr_auc", "f1", "precision", "recall")}},
               "current": {"threshold": thr_old, **{k: r_old[k] for k in
                                                     ("pr_auc", "f1", "precision", "recall")}},
               "delta_pr": [pr_m, pr_lo, pr_hi],
               "delta_f1": [f1_m, f1_lo, f1_hi],
               "note": "Node bootstrap; no test-based selection."}
    with open(os.path.join(RESULTS, "xgb_compare.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {os.path.join(RESULTS, 'xgb_compare.json')}", flush=True)


if __name__ == "__main__":
    main()
