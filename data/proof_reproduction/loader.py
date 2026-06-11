"""loader.py — zero-dependency dataloader for the proof_reproduction dataset.
Drop-in for the training pipeline: pure stdlib + optional numpy.

    from loader import load, feature_matrix, vocab
    rows = load("ql", split="train", split_kind="level")     # list of dicts
    X, ids = feature_matrix(rows, blocks=("struct","content_anon","compression"))
    tok2id = vocab("ql")                                     # registry codes

Conventions (DATACARD.md is the contract):
  * headline T1 uses blocks struct+content_anon (+compression); content_named is the
    fingerprint upper bound (leakage rule D4);
  * T2 targets: targets["steps_flat"] / ["steps_gram"]; only train on rows with
    targets["steps_certified"] == True (currently all);
  * exact-match metric: merkle (order-free); classic Gödel via
    atlas.core.godel.classic_godel(dep_codes);
  * never pool kernel19 metrics with .mm corpora.
"""
from __future__ import annotations
import json, pathlib
from typing import Dict, Iterable, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
CORPORA = ("hol", "ql", "kernel19")


def load(corpus: str, split: str | None = None, split_kind: str = "level") -> List[Dict]:
    rows = [json.loads(l) for l in open(HERE / f"{corpus}.jsonl")]
    if split is not None:
        rows = [r for r in rows if r["split"][split_kind] == split]
    return rows


def vocab(corpus: str) -> Dict[str, int]:
    return json.loads((HERE / f"{corpus}.registry.json").read_text())["code_of_label"]


def ceiling(corpus: str) -> Dict:
    p = HERE / f"{corpus}.ceiling.json"
    return json.loads(p.read_text()) if p.exists() else {}


def feature_matrix(rows: Iterable[Dict],
                   blocks: Tuple[str, ...] = ("struct", "content_anon", "compression"),
                   ) -> Tuple["object", List[str]]:
    """Returns (X, ids). Uses numpy if available, else nested lists."""
    feats, ids = [], []
    for r in rows:
        v: List[float] = []
        for b in blocks:
            v.extend(r["features"].get(b, []))
        feats.append(v); ids.append(r["id"])
    try:
        import numpy as np
        return np.asarray(feats, dtype=float), ids
    except ImportError:
        return feats, ids


def step_targets(rows: Iterable[Dict], kind: str = "flat",
                 require_certified: bool = True) -> List[Tuple[str, List[str]]]:
    key = {"flat": "steps_flat", "gram": "steps_gram"}[kind]
    out = []
    for r in rows:
        t = r["targets"]
        if require_certified and not t.get("steps_certified"):
            continue
        if t.get(key):
            out.append((r["id"], t[key]))
    return out
