# VERDICT — Verified Evidence, Rubric-Driven Inference & Calibrated Triage

Candidate-ranking system for the Redrob **Intelligent Candidate Discovery & Ranking Challenge**: rank the top 100 of 100,000 candidates for the *Senior AI Engineer — Founding Team* JD.

**Core formula — judgment, not similarity:**

```
Score(c) = J(c)^α × C(c)^β × A(c)^γ × e^(−flags)
```

- **J — Judgment**: the JD is compiled offline into an executable rubric ([artifacts/rubric_program.json](artifacts/rubric_program.json)) — hard gates, weighted crisp rules, fuzzy evidence predicates, and the JD's explicit disqualifier dampeners. Rules read *evidence*, never keyword lists.
- **C — Credibility**: every profile claim is trust-weighted by cost-of-faking; internal contradictions (tenure math, expert-skills-with-zero-use, narrative-vs-skills divergence) collapse C. Honeypots eliminate themselves — no special-casing.
- **A — Availability**: the 23 `redrob_signals` + logistics → probability the candidate can actually be engaged.

Multiplicative because recruiter instinct is multiplicative: brilliant-but-fake = 0, brilliant-but-unreachable ≈ 0. Full design + diagrams: [ARCHITECTURE.md](ARCHITECTURE.md).

## Reproduce the submission

```bash
# 1) install (Python 3.11+)
uv venv .venv && uv pip install -p .venv numpy orjson fastembed tqdm pyyaml

# 2) offline precompute (one-time; may exceed 5 min - allowed by spec §10.3)
python precompute.py --candidates ./candidates.jsonl --all-candidates
#    optional GPU speed-up:  pip install fastembed-gpu && add --cuda

# 3) ranking step (the constrained one: <5 min, CPU-only, no network)
python rank.py --candidates ./candidates.jsonl --out submission.csv

# 4) validate
python validate_submission.py submission.csv
```

Measured on a Ryzen 9 5900HX / 16 GB laptop: rank step **~12 s** end-to-end (budget: 300 s). The rank step loads no model and makes no network calls — it is pure numpy over precomputed vectors.

## Pipeline (two phases)

```
OFFLINE  (unconstrained)          RANK  (≤5 min · CPU · no network)
──────────────────────            ─────────────────────────────────
JD ─→ rubric_program.json         gates 100K → ~76K
       (LLM-compiled, reviewed)   ABM + BM25 + dense → RRF → 2.5K
candidates ─→ claim ledger        J×C×A scoring on 2.5K
sentence embeddings (bge-small)   fusion + Bradley-Terry tournament (top 300)
probe/predicate embeddings        evidence-grounded reasoning → CSV
```

## Repo map

| Path | What it is |
|---|---|
| `rank.py` | The ranking step (Stage-3 reproduced command) |
| `precompute.py` | Offline artifact builder (embeddings, ledger, probes) |
| `search.py` | Recruiter-style multi-role search surface over the same artifacts |
| `src/verdict/normalizer.py` | Entity standardization: titles, skills ontology, companies, locations |
| `src/verdict/evidence.py` | L1 claim ledger: timelines, corroboration, evidence sentences |
| `src/verdict/credibility.py` | L2: contradiction detection, stuffing signature → C |
| `src/verdict/recall.py` | Gates + ABM + BM25 + dense recall + RRF fusion |
| `src/verdict/judgment.py` | L3: rubric execution, fuzzy predicates → J |
| `src/verdict/availability.py` | L4: 23 behavioral signals → A |
| `src/verdict/fusion.py` | L5: J×C×A fusion + finalist tournament |
| `src/verdict/reasoning.py` | L6: hallucination-proof reasoning assembly |
| `src/verdict/feel_compiler.py` | L0: Gemini-powered JD→rubric draft compiler (offline only) |
| `artifacts/rubric_program.json` | The compiled, human-reviewed rubric (the auditable core) |
| `scripts/` | Calibration + audit tooling (predicate calibration, top-100 honeypot audit, evidence-cap audit) |

## Key design facts (for review)

- **Embeddings are evidence detectors, not scores.** Career narratives are embedded sentence-level; the skills list is *never* embedded — self-declared skills enter only as low-trust claims needing corroboration.
- **Predicate scoring is contrastive.** BGE cosines have a ~0.6 floor between any two texts; predicates score `max(cos_pos − cos_nearest_negative)`, calibrated on observed distributions (`scripts/calibrate_predicates.py`).
- **Top-100 honeypot audit: 0 leaked** (480 internally-contradictory profiles detected pool-wide, all excluded) — `scripts/audit_top100.py`.
- **Reasoning cannot hallucinate**: it is assembled from ledger fields and fired-rule evidence pointers only, tone tied to rank band, concerns always voiced.
- AI usage declaration: LLMs (Claude, Codex, Gemini) used for architecture research, code drafting, and offline rubric compilation. All ranking logic is deterministic local code; no LLM runs at rank time.
