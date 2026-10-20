import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def train_gnn(model, data, train_idx, val_idx, weight=None, focal_gamma=None,
              epochs=300, lr=0.005, wd=5e-4, patience=50, device="cuda"):
    """Full-batch training. Early stopping on VALIDATION PR-AUC.

    Loss uses only labeled TRAIN nodes.
      weight:      optional per-class weights for cross-entropy.
      focal_gamma: if set, use focal loss (1 - p_t)^gamma * CE, which down-weights
                   easy examples. Combined with `weight` if both are given.
    Returns (illicit probability for every node, best validation PR-AUC).
    Does not modify the `data` passed in.
    """
    model = model.to(device)
    data = data.clone().to(device)
    tr, va = train_idx.to(device), val_idx.to(device)
    w = weight.to(device) if weight is not None else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    y_val = data.y[va].cpu().numpy()

    def loss_fn(logits, target):
        if focal_gamma is None:
            return F.cross_entropy(logits, target, weight=w)
        ce_plain = F.cross_entropy(logits, target, reduction="none")
        ce = F.cross_entropy(logits, target, weight=w, reduction="none")
        p_t = torch.exp(-ce_plain)
        return ((1 - p_t) ** focal_gamma * ce).mean()

    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[tr], data.y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
        score = average_precision_score(y_val, p[va].cpu().numpy())
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
    return probs, float(best)
