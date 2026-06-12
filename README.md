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

## Sandbox / demo app

The portal sandbox is a small-sample demo, not the full 100K Stage-3 reproduction.
It accepts a JSONL or JSON array with up to 100 candidates, embeds the sample
on CPU at runtime, and produces a ranked CSV using the same claim ledger,
rubric predicates, credibility, availability, fusion, and reasoning modules as
the submission code.

Run locally:

```bash
pip install -r requirements-sandbox.txt
streamlit run sandbox_app.py
```

Docker alternative:

```bash
docker build -f Dockerfile.sandbox -t verdict-redrob-sandbox .
docker run --rm -p 8501:8501 verdict-redrob-sandbox
```

Hosted options:

- Streamlit Cloud: app file `sandbox_app.py`, requirements file `requirements-sandbox.txt`.
- HuggingFace Spaces: create a Streamlit Space and use `sandbox_app.py` as the app entrypoint.

Important: the sandbox embeds only the small uploaded/bundled sample. The
official full-pool ranking command remains:

```bash
python rank.py --candidates ./candidates.jsonl --out submission.csv
```

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
| `rank.py` | The constrained full-pool ranking step |
| `precompute.py` | Offline artifact builder for candidate evidence vectors, ledgers, and probes |
| `sandbox_app.py` | Small-sample Streamlit demo for the portal sandbox requirement |
| `scripts/build_probes.py` | Rebuild JD/probe embeddings without re-embedding candidate evidence |
| `scripts/bench_cpu_embeddings.py` | CPU embedding benchmark harness |
| `scripts/audit_quality_gates.py` | Submission-level quality gate for keyword stuffing, honeypots, availability, and hidden gems |
| `scripts/audit_honeypots.py` | Full-pool trap inventory plus top-100 leakage audit |
| `src/verdict/data.py` | Candidate JSON/JSONL/GZIP streaming loader |
| `src/verdict/normalizer.py` | Entity standardization: titles, skills ontology, companies, locations |
| `src/verdict/evidence.py` | Claim ledger: timelines, corroboration, evidence sentences |
| `src/verdict/embedder.py` | Offline FastEmbed wrapper used by `precompute.py` |
| `src/verdict/recall.py` | Gates + ABM + BM25 + dense recall + RRF fusion |
| `src/verdict/judgment.py` | Rubric execution and fuzzy predicates |
| `src/verdict/credibility.py` | Contradiction detection and stuffing penalties |
| `src/verdict/availability.py` | Behavioral/logistics availability scoring |
| `src/verdict/fusion.py` | J x C x A fusion and finalist tournament |
| `src/verdict/reasoning.py` | Evidence-grounded reasoning assembly |
| `artifacts/rubric_program.json` | Compiled, reviewed rubric |
| `requirements-sandbox.txt` | Dependencies for the sandbox app |
| `Dockerfile.sandbox` | Optional Docker wrapper for the sandbox app |

## Key design facts (for review)

- **Embeddings are evidence detectors, not scores.** Career narratives are embedded sentence-level; the skills list is *never* embedded — self-declared skills enter only as low-trust claims needing corroboration.
- **Predicate scoring is contrastive.** BGE cosines have a ~0.6 floor between any two texts; predicates score `max(cos_pos - cos_nearest_negative)`.
- **Top-100 honeypot audit: 0 leaked** under the current observable-trap audit - `scripts/audit_honeypots.py`.
- **Reasoning cannot hallucinate**: it is assembled from ledger fields and fired-rule evidence pointers only, tone tied to rank band, concerns always voiced.
- AI usage declaration: LLMs (Claude, Codex, Gemini) used for architecture research, code drafting, and offline rubric compilation. All ranking logic is deterministic local code; no LLM runs at rank time.
