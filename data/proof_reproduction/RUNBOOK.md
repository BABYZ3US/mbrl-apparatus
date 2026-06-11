# RUNBOOK — proof_reproduction experiment (ready state, 2026-06-10)

Everything below is in place; the experiment is one command away once the training slot
frees. Contract: `DATACARD.md`. Design: `math/proof_reproduction_ml_design_2026-06-10.md`.

## Ready now
- **Data:** `hol.jsonl` (149), `ql.jsonl` (1140), `kernel19.jsonl` (19) — four feature
  channels + compression, three target granularities (merkle / dep_codes / certified
  steps_flat + steps_gram), leak-safe splits, ceilings, registries.
- **Loader:** `loader.py` — `load(corpus, split=..., split_kind="level")`,
  `feature_matrix(rows, blocks)`, `step_targets(rows, kind="flat"|"gram")`, `vocab`,
  `ceiling`. Pure stdlib (+numpy if present).
- **Baseline (must-beat):** `python3 t1_baseline.py` — LOO-kNN dep-set match with
  size-only H0 control. Current: hol 0.027/0.027, ql 0.035/0.017 (vs ceilings
  0.047/0.027) → essentially all headroom is open for learning.
- **Verifier oracle:** `trace-atlas/atlas/research/mm_replay.py` + vendored
  `mmverify.py` (MIT). For T2 eval, verify a decoded flat sequence via
  `MM.verify(f_hyps, e_hyps, conclusion, decoded_labels)` against the loaded DB.
- **Replay caches (M3 scale, DONE):** `trace-atlas/ingest/set.replay.json`
  (47,319 thms, 100% roundtrip-certified, ratio mean 2.68 max 488) and
  `iset.replay.json` (16,011, 100%, mean 2.06). Step targets for the full 63k corpus
  exist; only the FEATURE export remains (below).

## Launch sequence (when the slot frees)
1. **T1 retrieval** on ql (1140 rows): embed `struct+content_anon+compression`,
   metric-learn, top-1/top-5 vs ceiling 0.027; report level + rand splits; ablate blocks.
2. **T2 generation** on ql: decoder over registry vocab; targets A/B = steps_flat vs
   steps_gram (pre-registered: length, decode speed, verifier-pass, merkle-match).
   Train only on `steps_certified` rows (currently all).
3. **kernel19**: eval-only qualitative rows (report the 3 eval items individually).

## M3 scale-up (one engineering item left)
The M1 exporter's per-node cone eigendecomposition is fine at 10³ rows but naive at
set.mm scale. Before exporting set/iset features, port the optimized corpus path
(`atlas/corpus.py`, numpy, 57k thms in ~70 s with `--max-upstream`) into
`proof_repro_export.py`, then add "set", "iset" to its corpus list. Step targets are
already cached — only features/splits/ceilings need the run.

## Invariants the suite enforces (don't silently regress)
MMX (dep purity/acyclicity), MMV (verifier replay == extracted deps), L3AG-style log
certificates. If you add a database: run `scripts/fetch_databases.sh` (manifest +
license segregation), then mm_replay, then export — and MMV picks it up by adding the
db name to its list.
