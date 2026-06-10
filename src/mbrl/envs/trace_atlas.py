"""trace_atlas — the proof-spectral reconstruction environment (Gymnasium).

Connects the trace-atlas theorem dataset (math/trace-atlas, ~57k Metamath rows) to the
curvature-MBRL trainer. The agent OBSERVES a theorem's spectral fingerprint (top-k
Laplacian eigenvalues of its proof-dependency DAG) and must OUTPUT an embedding that
reconstructs the theorem's target code — a continuous-control task whose reward is
reconstruction accuracy.

This is the **v0 warm-up**: a 1-step env over a FIXED per-theorem code vector — a
stand-in for the discrete proof symbol/index. trace-atlas carries `id` / `godel_code` /
`merkle` as reconstruction targets and an invertible Gödel expander (`atlas/core/godel.py`
`classic_decode`) for full ORDERED-PROOF targets later. The agent's policy is the
continuous-control learner; the map from its latent to this embedding space is the
"output embedding" being trained, and the curvature penalty regularizes the reward
surface over it.

Formulation (one decision worth the owner's call — see DESIGN NOTE at the bottom):
  observation = spectral vector            Box(k)
  action      = a unit embedding vector    Box(embed_dim)   [continuous control]
  reward      = cosine(action, target_code)  +  correctness bonus when the nearest
                codebook entry is this theorem (lookup-style decode is allowed)
  episode     = 1 step (one code per theorem)

Row schema (math/trace-atlas/*.jsonl): {id, tier, merkle, godel_code, spectral[k]}.
Stdlib JSONL load — no hard dependency on the trace-atlas package. Deterministic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Default corpus: the trace-atlas validation split, a sibling of the mbrl repo
# (math/trace-atlas/val.jsonl). Pass an explicit path for train.jsonl / a custom file.
DEFAULT_CORPUS = Path(__file__).resolve().parents[4] / "trace-atlas" / "val.jsonl"


def load_corpus(path) -> list[dict]:
    """Read a trace-atlas JSONL corpus -> list of rows (skips blank / torn lines)."""
    rows: list[dict] = []
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise FileNotFoundError(f"corpus not found: {path}") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn line — skip
        if isinstance(obj, dict) and isinstance(obj.get("spectral"), list):
            rows.append(obj)
    return rows


def code_vector(key: str, dim: int) -> np.ndarray:
    """A deterministic UNIT codebook vector for a theorem id (seeded by its hash).

    Stand-in for the theorem's discrete target embedding: stable across runs, distinct
    per id, unit-norm so the cosine reward is well-scaled. Swap for a learned/owned
    codebook by passing `codebook=` to the env.
    """
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class TraceAtlasReconstructionEnv(gym.Env):
    """1-step proof-spectral reconstruction (see module docstring)."""

    metadata: dict = {"render_modes": []}

    def __init__(self, corpus_path=None, k: int | None = None, embed_dim: int = 32,
                 codebook: dict | None = None, correctness_bonus: float = 1.0,
                 seed: int | None = None):
        super().__init__()
        self.rows = load_corpus(corpus_path if corpus_path is not None else DEFAULT_CORPUS)
        if not self.rows:
            raise ValueError(f"no usable rows in corpus: {corpus_path}")
        self.k = int(k if k is not None else len(self.rows[0]["spectral"]))
        self.embed_dim = int(embed_dim)
        self.correctness_bonus = float(correctness_bonus)
        self.ids = [str(r["id"]) for r in self.rows]
        # codebook: id -> unit target vector (deterministic stand-in by default)
        self.codebook = (codebook if codebook is not None
                         else {i: code_vector(i, self.embed_dim) for i in self.ids})
        self._cb = np.stack([self.codebook[i] for i in self.ids]).astype(np.float32)  # (N, dim)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.k,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.embed_dim,), np.float32)
        self._rng = np.random.default_rng(seed)
        self._cur = 0

    def _obs(self) -> np.ndarray:
        return np.asarray(self.rows[self._cur]["spectral"][: self.k], dtype=np.float32)

    # ---- gymnasium API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cur = int(self._rng.integers(0, len(self.rows)))
        row = self.rows[self._cur]
        info = {"id": row["id"], "tier": row.get("tier"),
                "godel_code": row.get("godel_code"), "merkle": row.get("merkle")}
        return self._obs(), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)[: self.embed_dim]
        na = float(np.linalg.norm(a))
        a_unit = a / na if na > 0 else a
        cosine = float(np.dot(a_unit, self.codebook[self.ids[self._cur]]))
        nn = int(np.argmax(self._cb @ a_unit))        # nearest codebook entry (decode)
        correct = nn == self._cur
        reward = cosine + (self.correctness_bonus if correct else 0.0)
        info = {"id": self.ids[self._cur], "correct": bool(correct),
                "nn_id": self.ids[nn], "cosine": cosine}
        return self._obs(), float(reward), True, False, info   # 1-step episode


def make_trace_atlas_env(corpus_path=None, **kw) -> TraceAtlasReconstructionEnv:
    """Factory mirroring make_task_env in tasks.py. corpus_path=None -> DEFAULT_CORPUS."""
    return TraceAtlasReconstructionEnv(corpus_path, **kw)


# ----------------------------------------------------------------------------------
# DESIGN NOTE — the one decision the formulation hangs on (for the owner):
#
# 1. ACTION INTERFACE  [CHOSEN 2026-06-09: continuous embedding + codebook].
#    v0 uses a continuous embedding action scored by cosine against a
#    per-theorem codebook vector, with a nearest-neighbour "decode" for the correctness
#    bonus. This keeps the env continuous-control-native (matches the existing trainer)
#    and puts the learned latent->embedding map as the thing optimized. The alternative
#    is a discrete action (the decoded symbol/index) with the decoder living in the
#    agent — simpler reward, but not continuous control.
#
# 2. CODEBOOK SOURCE. Default is a deterministic random unit vector per id (a fixed
#    target the policy learns to hit — the "useful output embedding" warm-up). Pass
#    `codebook=` to use a learned/owned embedding (e.g. an embedding of the godel_code
#    expansion or the Lean token sequence) when that exists.
#
# 3. HORIZON. v0 is 1 step (one code per theorem). Full proof reconstruction is a
#    SEQUENCE — decode godel_code -> ordered line-codes via trace-atlas
#    classic_decode/godel_sentence, then a multi-step env emitting one line per step.
#    That is the next env once the 1-step output-embedding warm-up trains cleanly.
# ----------------------------------------------------------------------------------
