import argparse
import os

import torch
import torch.nn.functional as F

from train_mil import (DEVICE, HP, TUM_IDX, InstanceMIL, evaluate, load_embeddings, make_test_sets,
                       save_json, set_seed, summarize)

BASELINE_HP = dict(lr=1e-3, batch_size=256)


def train_patch_classifier(X, y, seed, steps):
    set_seed(seed)
    model = InstanceMIL(X.shape[1], "max").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=BASELINE_HP["lr"], weight_decay=HP["weight_decay"])
    target = torch.as_tensor(y == TUM_IDX, dtype=torch.float32, device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    model.train()
    for _ in range(steps):
        i = torch.randint(0, len(X), (BASELINE_HP["batch_size"],), generator=g, device=DEVICE)
        loss = F.binary_cross_entropy_with_logits(model.scorer(X[i]).squeeze(-1), target[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_embeddings(args.emb_dir)
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    tests = make_test_sets(y_test)

    per_seed = {}
    for seed in args.seeds:
        model = train_patch_classifier(X_train, y_train, seed, args.steps)
        per_seed[seed] = evaluate(model, X_test, tests, args.out_dir, "baseline", seed)
        print(f"baseline seed {seed}: w1_n64 bag AUC {per_seed[seed]['w1_n64']['bag_auc']:.4f}", flush=True)

    out = {"settings": vars(args), "hyperparameters": {**HP, **BASELINE_HP},
           "results": {"baseline": {"summary": summarize(per_seed), "per_seed": per_seed}}}
    save_json(out, os.path.join(args.out_dir, "baseline_results.json"))


if __name__ == "__main__":
    main()
