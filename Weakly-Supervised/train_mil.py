import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUM_IDX = CLASSES.index("TUM")
HARD_NEG_CLASSES = ["NORM", "STR", "DEB", "MUC"]
TEST_WITNESSES = [1, 2, 3, 4, 8]
TEST_BAG_SIZES = [128, 256, 512, 1024]
TEST_BAG_SEED = 2024
N_TEST_BAGS = 1000
HP = dict(batch_size=32, lr=2e-4, weight_decay=1e-4, hidden=256, attn_dim=128, dropout=0.25, clam_bag_weight=0.7)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings(emb_dir):
    data = {}
    for split in ["train", "val", "test"]:
        X = np.load(os.path.join(emb_dir, f"feats_{split}.npy")).astype(np.float32)
        y = np.load(os.path.join(emb_dir, f"labels_{split}.npy")).astype(np.int64)
        data[split] = (X, y)
    mu = data["train"][0].mean(0, keepdims=True)
    sd = data["train"][0].std(0, keepdims=True) + 1e-6
    return {s: (torch.as_tensor((X - mu) / sd, device=DEVICE), y) for s, (X, y) in data.items()}


def make_bags(y, n_bags, bag_size, ntum, seed, neg_classes=None):
    rng = np.random.RandomState(seed)
    tum = np.where(y == TUM_IDX)[0]
    non_mask = y != TUM_IDX
    if neg_classes is not None:
        non_mask &= np.isin(y, [CLASSES.index(c) for c in neg_classes])
    non = np.where(non_mask)[0]
    bag_y = np.array([1] * (n_bags // 2) + [0] * (n_bags - n_bags // 2))
    rng.shuffle(bag_y)
    idx = np.zeros((n_bags, bag_size), int)
    is_tum = np.zeros((n_bags, bag_size), bool)
    for b, label in enumerate(bag_y):
        k = 0
        if label == 1:
            k = ntum if isinstance(ntum, int) else rng.randint(ntum[0], ntum[1] + 1)
        members = np.concatenate([rng.choice(tum, k, replace=False), rng.choice(non, bag_size - k, replace=False)])
        mask = np.array([True] * k + [False] * (bag_size - k))
        perm = rng.permutation(bag_size)
        idx[b], is_tum[b] = members[perm], mask[perm]
    return idx, bag_y, is_tum


def make_test_sets(y):
    tests = {}
    for k in TEST_WITNESSES:
        tests[f"w{k}_n64"] = make_bags(y, N_TEST_BAGS, 64, k, TEST_BAG_SEED + k)
    for n in TEST_BAG_SIZES:
        tests[f"w1_n{n}"] = make_bags(y, N_TEST_BAGS, n, 1, TEST_BAG_SEED + n)
    for k in [1, 4]:
        tests[f"hard_w{k}_n64"] = make_bags(y, N_TEST_BAGS, 64, k, TEST_BAG_SEED + 7000 + k,
                                            neg_classes=HARD_NEG_CLASSES)
    return tests


class InstanceMIL(nn.Module):
    def __init__(self, in_dim, pool):
        super().__init__()
        self.pool = pool
        self.scorer = nn.Sequential(nn.Linear(in_dim, HP["hidden"]), nn.ReLU(),
                                    nn.Dropout(HP["dropout"]), nn.Linear(HP["hidden"], 1))

    def forward(self, x):
        s = self.scorer(x).squeeze(-1)
        bag = s.mean(1) if self.pool == "mean" else s.max(1).values
        return bag, s, None


class GatedAttention(nn.Module):
    def __init__(self, hidden, attn_dim):
        super().__init__()
        self.V = nn.Linear(hidden, attn_dim)
        self.U = nn.Linear(hidden, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h):
        return self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)


class ABMIL(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, HP["hidden"]), nn.ReLU(), nn.Dropout(HP["dropout"]))
        self.attn = GatedAttention(HP["hidden"], HP["attn_dim"])
        self.classifier = nn.Linear(HP["hidden"], 1)

    def forward(self, x):
        h = self.embed(x)
        a = torch.softmax(self.attn(h), dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), a, None


class CLAMSB(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, HP["hidden"]), nn.ReLU(), nn.Dropout(HP["dropout"]))
        self.attn = GatedAttention(HP["hidden"], HP["attn_dim"])
        self.classifier = nn.Linear(HP["hidden"], 2)
        self.inst = nn.ModuleList([nn.Linear(HP["hidden"], 2) for _ in range(2)])

    def forward(self, x):
        h = self.embed(x)
        a = torch.softmax(self.attn(h), dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        logits = self.classifier(z)
        return logits[:, 1] - logits[:, 0], a, (logits, h)

    def instance_loss(self, a, h, bag_y, k):
        top = a.topk(k, dim=1).indices
        bot = (-a).topk(k, dim=1).indices
        sel = torch.cat([top, bot], 1)
        hs = torch.gather(h, 1, sel.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        target = torch.cat([torch.ones_like(top), torch.zeros_like(bot)], 1)
        out = torch.where(bag_y.view(-1, 1, 1).bool(), self.inst[1](hs), self.inst[0](hs))
        return F.cross_entropy(out.reshape(-1, 2), target.reshape(-1))


def make_model(method, in_dim):
    if method in ("mean", "max"):
        return InstanceMIL(in_dim, method)
    return {"abmil": ABMIL, "clam_sb": CLAMSB}[method](in_dim)


def train_mil(method, X, idx, bag_y, seed, steps, clam_k):
    set_seed(seed)
    model = make_model(method, X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    idx_t = torch.as_tensor(idx, device=DEVICE)
    y_t = torch.as_tensor(bag_y, device=DEVICE)
    bs = min(HP["batch_size"], len(idx))
    rng = np.random.RandomState(seed)
    model.train()
    step = 0
    while step < steps:
        order = rng.permutation(len(idx))
        for i in range(0, len(order) - bs + 1, bs):
            b = torch.as_tensor(order[i:i + bs], device=DEVICE)
            bag_logit, a, extra = model(X[idx_t[b]])
            if method == "clam_sb":
                logits, h = extra
                w = HP["clam_bag_weight"]
                loss = w * F.cross_entropy(logits, y_t[b]) + (1 - w) * model.instance_loss(a, h, y_t[b], clam_k)
            else:
                loss = F.binary_cross_entropy_with_logits(bag_logit, y_t[b].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= steps:
                break
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, idx):
    bs = max(1, 65536 // idx.shape[1])
    probs, loc = [], []
    for i in range(0, len(idx), bs):
        bag_logit, s, _ = model(X[torch.as_tensor(idx[i:i + bs], device=DEVICE)])
        probs.append(torch.sigmoid(bag_logit).cpu())
        loc.append(s.cpu())
    return torch.cat(probs).numpy(), torch.cat(loc).numpy()


def bag_metrics(bag_y, probs, is_tum, loc):
    fpr, tpr, _ = roc_curve(bag_y, probs)
    loc_auc, hit1, prec_k = [], [], []
    for b in np.where(bag_y == 1)[0]:
        k = int(is_tum[b].sum())
        order = np.argsort(-loc[b], kind="stable")
        loc_auc.append(roc_auc_score(is_tum[b], loc[b]))
        hit1.append(float(is_tum[b][order[0]]))
        prec_k.append(float(is_tum[b][order[:k]].mean()))
    return {
        "bag_auc": float(roc_auc_score(bag_y, probs)),
        "bag_accuracy": float(accuracy_score(bag_y, probs >= 0.5)),
        "sens_at_95spec": float(tpr[fpr <= 0.05].max()),
        "loc_auc": float(np.mean(loc_auc)),
        "hit_at_1": float(np.mean(hit1)),
        "precision_at_k": float(np.mean(prec_k)),
    }


def evaluate(model, X, tests, out_dir, name, seed):
    results, preds = {}, {}
    for test, (idx, bag_y, is_tum) in tests.items():
        probs, loc = predict(model, X, idx)
        results[test] = bag_metrics(bag_y, probs, is_tum, loc)
        preds.update({f"{test}_idx": idx, f"{test}_bag_y": bag_y, f"{test}_is_tum": is_tum,
                      f"{test}_probs": probs, f"{test}_loc": loc})
    np.savez_compressed(os.path.join(out_dir, f"preds_{name}_seed{seed}.npz"), **preds)
    return results


def summarize(per_seed):
    seeds = list(per_seed)
    summary = {}
    for test in per_seed[seeds[0]]:
        summary[test] = {}
        for metric in per_seed[seeds[0]][test]:
            v = np.array([per_seed[s][test][metric] for s in seeds])
            summary[test][metric] = {"mean": float(v.mean()), "std": float(v.std())}
    return summary


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["mean", "max", "abmil", "clam_sb"],
                        choices=["mean", "max", "abmil", "clam_sb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bag-size", type=int, default=64)
    parser.add_argument("--n-train-bags", type=int, default=2000)
    parser.add_argument("--min-tumour", type=int, default=1)
    parser.add_argument("--max-tumour", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--clam-k", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_embeddings(args.emb_dir)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    tests = make_test_sets(y_test)

    out = {"settings": vars(args), "hyperparameters": HP, "results": {}}
    for method in args.methods:
        per_seed = {}
        for seed in args.seeds:
            idx, bag_y, _ = make_bags(y_train, args.n_train_bags, args.bag_size,
                                      (args.min_tumour, args.max_tumour), seed)
            model = train_mil(method, X_train, idx, bag_y, seed, args.steps, args.clam_k)
            per_seed[seed] = evaluate(model, X_test, tests, args.out_dir, method, seed)
            print(f"{method} seed {seed}: w1_n64 bag AUC {per_seed[seed]['w1_n64']['bag_auc']:.4f}", flush=True)
        out["results"][method] = {"summary": summarize(per_seed), "per_seed": per_seed}
        save_json(out, os.path.join(args.out_dir, "mil_results.json"))


if __name__ == "__main__":
    main()
