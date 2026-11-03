"""Iterative confidence-thresholded pseudo-labeling in feature space (one run).

Round 0 trains on labeled data only. In each later round the previous model
predicts on all unlabeled features (clean), predictions with confidence
>= tau become pseudo-labels (re-selected from scratch every round), and a new
model is trained from scratch on labeled + pseudo-labeled features with
Gaussian feature noise (FixMatch-style: confident prediction on the clean
view, training on the perturbed view).

Hidden true labels of the unlabeled pool are used ONLY to log pseudo-label
accuracy (diagnostic). Inlined into the Stage 3 kernel by tools/build_kernels.py.
"""
from common import *  # noqa: F401,F403  (inlined on Kaggle)

TAUS = [0.9, 0.95]
N_ROUNDS = 5  # pseudo-labeling rounds after the labeled-only round 0


def run_one(data, kind, frac, tau, seed):
    """One full pseudo-labeling run.

    Returns (per-round log, final model, n_labeled, n_unlabeled)."""
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    lab = stratified_subset(ytr, frac, seed)
    unl = np.setdiff1d(np.arange(len(ytr)), lab)
    X_unl, y_unl_hidden = Xtr[unl], ytr[unl]

    rounds = []
    model = train_classifier(Xtr[lab], ytr[lab], kind, seed)
    for r in range(N_ROUNDS + 1):
        if r > 0:
            probs = predict_proba(model, X_unl)
            conf, pseudo = probs.max(1), probs.argmax(1)
            mask = conf >= tau
            n_added = int(mask.sum())
            # Diagnostic only: hidden labels never enter training.
            pl_acc = float((pseudo[mask] == y_unl_hidden[mask]).mean()) if n_added else float("nan")
            model = train_classifier(Xtr[lab], ytr[lab], kind, seed,
                                     X_pl=X_unl[mask], y_pl=pseudo[mask])
        else:
            n_added, pl_acc = 0, float("nan")
        p_val = predict_proba(model, data["val"]["X"]).argmax(1)
        p_test = predict_proba(model, data["test"]["X"]).argmax(1)
        rounds.append({
            "round": r, "n_pseudo": n_added, "pseudo_acc": pl_acc,
            "val_macro_f1": float(f1_score(data["val"]["y"], p_val, average="macro")),
            "test_macro_f1": float(f1_score(data["test"]["y"], p_test, average="macro")),
        })
        log(f"{kind}/{frac}/tau={tau}/seed={seed} round {r}: n_pseudo={n_added} "
            f"pl_acc={pl_acc:.4f} val_f1={rounds[-1]['val_macro_f1']:.4f}")
    return rounds, model, len(lab), len(unl)
