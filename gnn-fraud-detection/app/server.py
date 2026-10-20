r"""Zero-dependency demo server for the Elliptic fraud portfolio app.
Run from the repo root:  .venv\Scripts\python.exe app\server.py
Then open http://127.0.0.1:8000

Endpoints:
  GET  /api/meta                 model metrics, threshold, feature names
  GET  /api/transactions         demo rows with live probabilities
  GET  /api/transaction/<id>     one row + XGBoost contributions
  POST /api/predict              {"features": [188 floats]}
  GET  /health

Only numpy and xgboost are needed at serving time; torch is not imported.
"""
import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import xgboost as xgb

APP = Path(__file__).resolve().parent
STATIC = APP / "static"
DATA = APP / "data"
MODEL_DIR = APP / "model"


class Demo:
    def __init__(self):
        self.meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
        z = np.load(DATA / "demo.npz")
        self.X = z["X"]
        self.y = z["y"]
        self.step = z["step"]
        self.node = z["node"]
        self.prob = z["prob"]
        self.features = self.meta["features"]
        self.models = []
        for i in range(5):
            b = xgb.Booster()
            b.load_model(str(MODEL_DIR / f"xgb_{i}.ubj"))
            self.models.append(b)

    def transactions(self):
        return [{"id": i, "step": int(self.step[i]), "y": int(self.y[i]),
                 "prob": float(self.prob[i]), "node": int(self.node[i])}
                for i in range(len(self.prob))]

    def _contrib(self, rows):
        dmat = xgb.DMatrix(rows)
        cs = [m.predict(dmat, pred_contribs=True) for m in self.models]
        return np.mean(cs, axis=0)[:, :-1], np.mean(
            [c[:, -1] for c in cs], axis=0)

    def _items(self, vals, contrib):
        items = [{"name": self.features[j], "value": float(vals[j]),
                  "contrib": float(contrib[j])}
                 for j in range(len(self.features))]
        pos = sorted([x for x in items if x["contrib"] > 0],
                     key=lambda x: -x["contrib"])[:6]
        neg = sorted([x for x in items if x["contrib"] < 0],
                     key=lambda x: x["contrib"])[:6]
        return pos, neg

    def detail(self, i):
        contrib, bias = self._contrib(self.X[i:i + 1])
        pos, neg = self._items(self.X[i], contrib[0])
        p = float(self.prob[i])
        return {"id": i, "node": int(self.node[i]), "step": int(self.step[i]),
                "y": int(self.y[i]), "prob": p,
                "threshold": float(self.meta["threshold"]),
                "flagged": p >= float(self.meta["threshold"]),
                "bias": float(bias[0]),
                "top_positive": pos, "top_negative": neg}

    def predict(self, feats):
        x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        contrib, bias = self._contrib(x)
        p = float(np.mean([m.predict(xgb.DMatrix(x))[0]
                           for m in self.models]))
        pos, neg = self._items(x[0], contrib[0])
        return {"prob": p, "threshold": float(self.meta["threshold"]),
                "flagged": p >= float(self.meta["threshold"]),
                "bias": float(bias[0]),
                "top_positive": pos, "top_negative": neg}


class Handler(BaseHTTPRequestHandler):
    server_version = "EllipticDemo/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, status=200, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode("utf-8"), status,
                   "application/json; charset=utf-8")

    def _static(self, rel):
        base = STATIC.resolve()
        target = (base / rel).resolve()
        if target != base and base not in target.parents:
            return self._json({"error": "forbidden"}, 403)
        if not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send(target.read_bytes(), 200, ctype)

    def do_GET(self):
        demo = self.server.demo
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path == "/health":
                return self._json({"ok": True, "rows": len(demo.prob)})
            if path == "/api/meta":
                return self._json(demo.meta)
            if path == "/api/transactions":
                return self._json(demo.transactions())
            if path.startswith("/api/transaction/"):
                try:
                    i = int(path.rsplit("/", 1)[1])
                except ValueError:
                    return self._json({"error": "bad id"}, 400)
                if not 0 <= i < len(demo.prob):
                    return self._json({"error": "not found"}, 404)
                return self._json(demo.detail(i))
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            return self._json({"error": "not found"}, 404)
        except Exception as exc:  # keep the demo alive on a bad request
            return self._json({"error": str(exc)}, 500)

    def do_POST(self):
        demo = self.server.demo
        if urlparse(self.path).path != "/api/predict":
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 1_000_000:
                return self._json({"error": "bad body size"}, 400)
            payload = json.loads(self.rfile.read(length))
            feats = payload.get("features")
            if not isinstance(feats, list) or len(feats) != len(demo.features):
                return self._json(
                    {"error": f"expected {len(demo.features)} features"}, 400)
            arr = np.asarray(feats, dtype=np.float32)
            if not np.isfinite(arr).all():
                return self._json({"error": "non-finite feature"}, 400)
            return self._json(demo.predict(feats))
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()

    demo = Demo()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.demo = demo
    print(f"Elliptic demo on http://{args.host}:{args.port} "
          f"({len(demo.prob)} demo rows, threshold {demo.meta['threshold']:.2f})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
