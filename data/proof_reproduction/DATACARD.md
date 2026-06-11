# proof_reproduction dataset — M1 export (2026-06-10)

**Design:** `math/proof_reproduction_ml_design_2026-06-10.md`. **Builders:**
`trace-atlas/atlas/research/proof_repro_export.py`, `kernel19_export.py` (re-run to
regenerate; deterministic).

## Files
Per corpus: `<corpus>.jsonl` (one row per *provable* item — axioms excluded from rows,
kept in `<corpus>.registry.json` with the code table), `<corpus>.ceiling.json`
(cospectrality collision ceilings — the information-theoretic accuracy cap per feature
block), `summary.json`.

## Row schema
```
id, corpus, depth, n_cone,
split: {level: train|test,   # top-20%-depth band held out (anti-leakage)
        rand:  train|test},  # merkle-hash 1-in-5 (sanity split)
targets: {merkle,            # order-free Gödel index (content hash) — exact-match target
          dep_codes,         # registry code sequence (canonical first-appearance order)
          dep_labels},
features: {struct[16],       # cone Laplacian spectrum (atlas.core.feature_vector)
           dag[5],           # depth, |cone|, leaves, in-deg, out-deg
           content_named[16],# step-symbol bipartite singular values
           content_anon[16]} # same, symbols -> 8 log-frequency buckets (D4 rule:
                             # T1 headline numbers use THIS variant)
```
`classic_godel(dep_codes)` is computable downstream via `atlas.core.godel`. Per spec
(2026-06-10): any decoded sequence that **verifies** counts as a win; exact-match is
scored on the order-free merkle index.

## Counts and ceilings (struct+content_anon combined block; clean refetch 2026-06-10)
| corpus | task rows | ceiling (collision rate) |
|---|---|---|
| hol | 149 | 0.047 |
| ql | 1140 | 0.027 |
| kernel19 | 19 (16 train / 3 eval) | n/a (report individually; `kernel:` namespace vocab=71) |

(Earlier export was built from TRUNCATED web_fetch artifacts — hol 60→97 KB, ql 54→591 KB
after clean refetch; caught by the MMV verifier replay. Old counts 116/171 are obsolete.)

*Removed 2026-06-10 to avoid confusion (regenerable from the builders):* `demo0.*` (1-row
toy), `peano.*` (0 rows), and the stale v0.6 `trace-atlas/training_corpus.jsonl`
(57k-theorem corpus WITHOUT the content channel or the new splits — regenerate via
`atlas corpus` only when M3 re-export adds both).
**Correction (same day):** the peano "parser gap" was misdiagnosed — the local peano.mm
was a truncated download containing zero `$p` statements (caught by the new fetch script's
sanity check). The broken file is deleted; a good copy is refetchable via
`bash scripts/fetch_databases.sh all` (note: GPL → lands in `ingest/gpl/`, excluded from
license-clean corpora).

Headline implication: with both channels, ≤ 9.5% of hol rows are feature-indistinguishable
→ top-1 retrieval caps at ≈ 90.5% there (ql ≈ 96%). Struct-alone caps at ~80/91% —
the content channel is doing real work, as intended.

## Public .mm databases for training pulls (researched 2026-06-10)

All official databases live in one repo: **github.com/metamath/set.mm** (raw files at
`raw.githubusercontent.com/metamath/set.mm/develop/<name>.mm`). License: CC0/public domain
for all EXCEPT peano.mm (GPL — segregate or drop if license-clean training data matters).

| db | domain | ~scale | status here |
|---|---|---|---|
| set.mm | ZFC, classical (MPE; PNT, Basel, etc.) | ~47k thms, 50 MB | **local** (ingest/) |
| **iset.mm** | intuitionistic set theory (ILE) | ~10k+ thms | **NOT local — the high-value pull** (domain shift: same statements, no LEM) |
| nf.mm | Quine's New Foundations | ~6k thms | local |
| ql.mm | quantum logic / ortholattices | ~1k | local (M1 corpus) |
| hol.mm | higher-order logic bridge | ~200 | local (M1 corpus) |
| peano.mm | Peano arithmetic | small | local; **GPL license**; parser gap (M2) |
| miu.mm / demo0.mm / big-unifier.mm | toys / verifier stress tests | tiny | local / skip (not training material) |

Additional source worth flagging: **set.mm's git history** — thousands of proof revisions
(shortenings, refactors) = *multiple valid proofs of the same statement*, a unique
supervision signal for exactly the "relationship between proofs + content" structure this
experiment targets. Snapshot-diff extraction would be an M3+ pipeline addition.

## Alignment guarantees (added 2026-06-10 after the $f-pollution find)

How we know the .mm extraction and the proof DAG actually align — three layers:
1. **Machine-checked (suite claim `MMX`, pass, 107↔107):** dependency purity (every dep
   is an $a/$p assertion — the original extractor leaked $e/$f hypothesis labels: 46 $f
   refs + 6 dangling in hol.mm, now fixed in `providers/metamath.py` and re-exported;
   current data: 0 polluted rows), acyclicity (toposort), global label uniqueness,
   axiom-leaf census, distinct merkle ids.
2. **By construction:** merkle indices are computed FROM the extracted DAG, so
   index↔DAG consistency is definitional; the kernel19 corpus aligns by schema analogy
   (coarse human-level steps in a `kernel:` namespace), not formal identity — do not pool
   it with .mm metrics.
3. **Gold standard (suite claim `MMV`, pass, 108↔108):** every proof in hol+ql replays
   correctly through the official `mmverify.py` (vendored, MIT), and the per-theorem
   captured usage sets **exactly equal** our extracted dependency sets — zero soundness
   violations, zero listed-but-unused labels. The replay also exposed that the previous
   hol/ql ingests were truncated web_fetch artifacts (refetched clean; dataset rebuilt).
   Machinery: `atlas/research/mm_replay.py` (recording-dict instrumentation, vendored
   verifier untouched; requires py≥3.10 → runs under the mbrl venv).

## M4 step-level targets (added 2026-06-10; BOTH granularities kept by design)
The schema now carries theorem-level AND step-level targets side by side:
- `targets.steps_flat` — the fully expanded label sequence (normal-format-equivalent
  RPN proof). **Verifier-certified**: every flat sequence was re-verified through
  mmverify.py as a normal-format proof during extraction (`steps_certified`;
  currently 149/149 hol, 1140/1140 ql).
- `targets.steps_gram` — the native grammar-compressed stream (labels + `<SAVE>`/`<REFk>`
  specials; Metamath's Z-tag mechanism ≈ straight-line-program grammar). Shorter,
  compositional; decodes deterministically to `steps_flat`.
- `features.compression = [ratio, len_flat, len_gram]` — ratio = flat/gram length
  (proof self-similarity; ql mean 1.93, max 9.2; 1.0 = no internal sharing).
Pre-registered M4 A/B: train decoders on flat vs gram targets; compare target length,
decode speed, verifier-pass and exact-merkle rates. Note: step streams legitimately
contain $e/$f hypothesis labels (they are real proof steps); the $a/$p-purity rule (MMX)
applies to DEPENDENCY sets only — do not confuse the two.
Regeneration: `../mbrl/.venv/bin/python3 -m atlas.research.mm_replay ingest/<db>.mm`
(writes `<db>.replay.json`), then the exporter.

## Remaining scope limits
1. Content channel uses own+direct-dep statements (not full cone) — cheap, ablatable.
2. set.mm deferred to M3 (50 MB local, pipeline-ready; replay cache will take ~minutes).
3. Eigenvector-aware features (heat traces, Fiedler entropy) designed but not yet in
   `atlas.core.feature_vector` — M2 item; expected to lower ceilings further.
4. kernel19 rows have no step-level targets (kernel proofs are already step-granular;
   their dep_codes ARE the step sequence).
