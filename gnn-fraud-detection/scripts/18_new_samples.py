r"""Create and score a fresh batch of held-out transactions for the demo.
Run from the repo root:  .venv\Scripts\python.exe scripts\18_new_samples.py

Selects 12 transactions from test steps 35-42 and 43-49, excluding every node
already in app/data/demo.npz, then scores them with the five deployed boosters.
True labels are kept only for checking; they never enter the model.

Writes results/new_samples.json, including the 188-feature vector for each row,
so the samples can be replayed through POST /api/predict.
Use --via-api to score them through the running server instead of directly.
"""
import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_elliptic, load_time_steps

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
RESULTS = ROOT / "results"
N_LOCAL = 93


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
    return np.hstack([xl, deg, contrast]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="new_samples.json")
    parser.add_argument("--via-api", action="store_true")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    data = load_elliptic()
    t = load_time_steps().numpy()
    y = data.y.numpy()
    X = features(data)
    n = data.num_nodes

    demo = np.load(APP / "data" / "demo.npz")
    demo_nodes = set(int(v) for v in demo["node"])
    is_demo = np.zeros(n, dtype=bool)
    is_demo[list(demo_nodes)] = True

    rng = np.random.default_rng(7)
    selected = []
    for lo, hi in [(35, 42), (43, 49)]:
        for lab in (1, 0):
            idx = np.where((t >= lo) & (t <= hi) & (y == lab) & ~is_demo)[0]
            take = min(3, len(idx))
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    selected = sorted(set(selected), key=lambda i: (int(t[i]), int(i)))
    Xs = X[selected]

    models = []
    for i in range(5):
        b = xgb.Booster()
        b.load_model(str(APP / "model" / f"xgb_{i}.ubj"))
        models.append(b)
    dmat = xgb.DMatrix(Xs)
    probs = np.mean([b.predict(dmat) for b in models], axis=0)
    contribs = np.mean([b.predict(dmat, pred_contribs=True)
                        for b in models], axis=0)[:, :-1]

    meta = json.loads((APP / "data" / "meta.json").read_text(encoding="utf-8"))
    thr = float(meta["threshold"])
    names = meta["features"]

    samples = []
    for row, i in enumerate(selected):
        c = contribs[row]
        j = int(np.argmax(c))
        samples.append({
            "node": int(i),
            "step": int(t[i]),
            "true_label": "Illicit" if y[i] == 1 else "Licit",
            "y": int(y[i]),
            "prob": float(probs[row]),
            "flagged": bool(probs[row] >= thr),
            "threshold": thr,
            "top_positive": {"name": names[j], "value": float(Xs[row, j]),
                             "contrib": float(c[j])},
            "features": [float(v) for v in Xs[row]],
        })

    print(f"{'node':<10}{'step':>5}{'score':>8}{'decision':>10}{'true':>9}  top signal")
    print("-" * 78)
    for s in samples:
        print(f"{s['node']:<10}{s['step']:>5}{s['prob']:>8.3f}"
              f"{'Flagged' if s['flagged'] else 'Clear':>10}"
              f"{s['true_label']:>9}  {s['top_positive']['name']} "
              f"({s['top_positive']['contrib']:+.2f})")

    payload = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "model": meta["model"], "threshold": thr,
               "selection": "steps 35-42 and 43-49, excluding demo nodes, 3 illicit + 3 licit per range",
               "samples": samples}
    with open(RESULTS / args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {RESULTS / args.out} ({len(samples)} samples, 188 features each)")

    if args.via_api:
        print("\nScoring through the running API:")
        for s in samples:
            req = urllib.request.Request(
                args.api_url.rstrip("/") + "/api/predict",
                data=json.dumps({"features": s["features"]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                p = json.loads(r.read())
            print(f"  node {s['node']}: api {p['prob']:.6f} "
                  f"{'Flagged' if p['flagged'] else 'Clear'} "
                  f"(direct {s['prob']:.6f}, "
                  f"match={abs(p['prob'] - s['prob']) < 1e-6})")


if __name__ == "__main__":
    main()
