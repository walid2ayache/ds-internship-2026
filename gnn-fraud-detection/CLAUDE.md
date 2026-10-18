# CLAUDE.md: GNN Fraud Detection

## Goal
Detect illicit transactions in the Elliptic Bitcoin graph with GNNs (GCN, GraphSAGE, GAT),
compare against an XGBoost baseline, handle class imbalance, and explain predictions.
This is a portfolio project: clean code, honest results, a strong README.

## Stack
Python, PyTorch, PyTorch Geometric, scikit-learn, XGBoost, matplotlib.
Windows machine, run commands from the repo root.

## Layout
- src/data.py       dataset loading and splits
- src/models.py     GCN / SAGE / GAT
- src/train.py      training and evaluation loop
- src/explain.py    GNNExplainer
- scripts/          runnable entry points (01_explore.py, 02_baseline.py, ...)
- results/          figures and metrics tables (commit these)
- data/             downloaded data (never commit)

## Hard rules
- Split by TIME (train on early time steps, test on later). Never random splits.
- Report F1 on the illicit class, precision, recall, PR-AUC. Never plain accuracy.
- Every model uses the same split and metrics as the XGBoost baseline.
- Unknown-label nodes are excluded from loss and metrics.
- Set random seeds and save metrics to results/.
- Do not add dependencies without asking.

## Working style
- Do one étape at a time. Propose a short plan before writing code, then wait for approval.
- Keep scripts small and runnable. Print results clearly.
- Do not refactor files that were not part of the task.
- If something is uncertain (label encoding, mask meaning), print and verify it, do not assume.

## Étapes
1 setup (done) | 2 explore data | 3 graph + split | 4 XGBoost baseline | 5 GCN/SAGE/GAT
6 imbalance (weighted loss, focal) | 7 evaluation table | 8 GNNExplainer | 9 README polish
