"""t1_baseline.py — the no-learning baseline every T1 model must beat, plus a smoke
test that the dataset + loader are wired correctly.

Two numbers per corpus:
  LOO-kNN identification: leave-one-out cosine nearest-neighbor over ALL rows on the
    headline feature blocks — measures raw feature identifiability; ceiling-adjusted
    accuracy cap printed alongside (DATACARD ceilings).
  size-only control (H0): same kNN using ONLY [depth, n_cone] — the null hypothesis
    baseline; a model that doesn't beat THIS has learned nothing about proof identity.

Run: python3 t1_baseline.py            (numpy required; ~seconds)
"""
import numpy as np
from loader import load, feature_matrix, ceiling


def loo_knn_top1(X, k_norm=True):
    Xn = X - X.mean(0)
    n = np.linalg.norm(Xn, axis=1, keepdims=True); n[n == 0] = 1
    Xn = Xn / n
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)
    return nn


def run_corpus(corpus):
    rows = load(corpus)
    if len(rows) < 5:
        return
    merk = [r["targets"]["merkle"] for r in rows]
    # identification: NN row shares the merkle? (distinct rows, so success = the NN is
    # the unique feature-twin; with all-distinct merkles top1 'identity' is only
    # meaningful via duplicate-feature groups -> report twin-consistency instead:
    # the fraction whose NN has IDENTICAL feature vector (ceiling mass) vs distinct)
    X, ids = feature_matrix(rows, ("struct", "content_anon", "compression"))
    Xs, _ = feature_matrix(rows, ("dag",))
    size_only = Xs[:, :2]                      # depth, n_cone
    nn_full = loo_knn_top1(np.asarray(X))
    nn_size = loo_knn_top1(np.asarray(size_only))
    # proxy retrieval task for the unlearned baseline: does the NN share the same
    # DIRECT-DEP MULTISET? (a structure-identity surrogate that generalizes)
    deps = [frozenset(r["targets"]["dep_labels"]) for r in rows]
    acc_full = np.mean([deps[i] == deps[nn_full[i]] for i in range(len(rows))])
    acc_size = np.mean([deps[i] == deps[nn_size[i]] for i in range(len(rows))])
    ceil = ceiling(corpus).get("struct+content_anon", {}).get("rate", float("nan"))
    print(f"{corpus:8s} rows={len(rows):5d}  NN-depset-match: full-features {acc_full:.3f}"
          f" | size-only(H0) {acc_size:.3f} | collision ceiling {ceil:.3f}")


if __name__ == "__main__":
    for c in ("hol", "ql"):
        run_corpus(c)
    print("\n(any trained T1 model must beat the size-only column by a clear margin;")
    print(" exact-identification accuracy is capped at 1 - ceiling)")
