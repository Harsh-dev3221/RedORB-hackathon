# CPU-Only Embedding Optimization Research

Scope: Redrob challenge job-description path, CPU-only, no network during `rank.py`.

## Executive Finding

The ranking architecture is already correct for the Stage-3 constraint: `rank.py`
does not embed text or load an embedding model. It consumes precomputed vectors
from `artifacts/` and finishes in roughly 13 seconds on the current machine.

The optimization target is therefore not "make rank.py embed faster"; it is:

1. Make offline precompute faster and more reproducible on CPU.
2. Make job-description iteration cheap by rebuilding only JD/probe vectors.
3. Reduce artifact IO and memory conversion in rank-time scoring.
4. Keep embedding quality high enough that hidden plain-language candidates are
   still discovered.

## Current Repo Baseline

Current code:

- `src/verdict/embedder.py`
  - FastEmbed `TextEmbedding`
  - model: `BAAI/bge-small-en-v1.5`
  - 384 dimensions
  - ONNX Runtime backend
  - passage text clipped to `MAX_PASSAGE_WORDS = 80`
- `precompute.py`
  - builds evidence sentence embeddings
  - saves `evidence_vectors.npy` as `float16`
  - saves candidate mean vectors as `float16`
  - embeds JD/hypothetical/predicate probes once into `probes.npz`
- `rank.py`
  - loads artifacts
  - no model load
  - no external network
  - runs gates, recall, scoring, fusion

Current artifact stats:

- candidates: `100000`
- evidence sentences: `1106477`
- evidence sentences per candidate: mean `11.06`, min `6`, max `12`
- `evidence_vectors.npy`: `810.4 MB`
- `mean_vecs.npy`: `73.2 MB`
- `records.jsonl.gz`: `57.1 MB`
- `probes.npz`: `0.07 MB`

Current runtime/package environment:

- Python `3.11.15`
- FastEmbed `0.8.0`
- ONNX Runtime `1.26.0`
- CPU count: `16`
- ONNX providers available: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`

Small local CPU benchmark on real evidence sentences:

| Threads | Batch | Throughput |
|---:|---:|---:|
| 2 | 8 | `54.1 sent/s` |
| 2 | 16 | `42.8 sent/s` |
| 2 | 32 | `34.1 sent/s` |
| 8 | 8 | `116.9 sent/s` on 120-sentence probe |
| 8 | 4 | `86.7 sent/s` on 240-sentence probe |
| 8 | 8 | `85.5 sent/s` on 240-sentence probe |
| 8 | 16 | `61.7 sent/s` on 240-sentence probe |
| 12 | 4 | `77.4 sent/s` |
| 16 | 4 | `41.0 sent/s` |

Important: larger batches were worse on this CPU. `threads=16` was worse than
`threads=8`, likely due to oversubscription/cache pressure.

## Source-Backed Optimization Notes

### 1. Keep Embeddings Out Of `rank.py`

The submission spec permits precomputation outside the 5-minute rank step. It
expects the final ranker to be a small CPU-only ranker over precomputed features.
This repo already follows that shape.

Do not move candidate embedding into `rank.py`.

### 2. Tune ONNX Runtime Threads Explicitly

ONNX Runtime exposes intra-op and inter-op thread settings. Intra-op threads
parallelize inside operators; inter-op threads parallelize across graph nodes.
ORT docs also note that graph optimization defaults to `ORT_ENABLE_ALL`.

Implication for this repo:

- Keep `TextEmbedding(..., threads=N)` explicit.
- Benchmark `N` on the target CPU.
- On this Windows 16-logical-core machine, `threads=8` beat `threads=16`.
- Avoid nested oversubscription: `workers * threads` should not exceed physical
  cores by much.

Primary source:

- ONNX Runtime threading/performance docs:
  https://onnxruntime.ai/docs/performance/tune-performance/threading.html

### 3. Batch Size Must Be CPU-Benchmarked, Not Guessed

FastEmbed docs recommend using `batch_size` and `parallel` for throughput. That
is true generally, but the local benchmark shows this model + CPU prefers small
batches (`4` or `8`) over the current default `32`.

Implication for this repo:

- Change precompute defaults from `--batch-size 32` to `8` after validating on a
  larger sample.
- Keep a benchmark script that tests `threads x batch_size x workers` and writes
  results to `output/`.
- Do not use GPU-style huge batches on CPU.

Primary sources:

- FastEmbed throughput docs:
  https://qdrant.tech/documentation/fastembed/fastembed-optimize/
- FastEmbed uses ONNX Runtime and quantized weights:
  https://github.com/qdrant/fastembed

### 4. Avoid Process Parallelism Until It Beats Single-Session ORT

`precompute.py` has a multiprocessing branch. Each worker initializes its own
model with `threads=2`, and chunks are passed through IPC as fp16 bytes.

This can help on Linux/high-core machines, but on Windows it can also hurt due to
process spawn cost, model duplication, IPC overhead, and thread oversubscription.

Recommended policy:

- Default to one process with tuned ORT threads.
- Enable `--workers` only after benchmark proof.
- If using workers:
  - use `threads = max(1, floor(cpu_count / workers))`
  - use smaller chunk sizes for progress visibility
  - avoid `workers=8, threads=8` style oversubscription

### 5. Candidate Vectors Are JD-Agnostic: Exploit That Hard

The candidate evidence vectors do not depend on the job description. For a new
JD, only the following should change:

- rubric JSON
- hypothetical ideal profiles
- predicate positives/negatives
- `probes.npz`

That means a new JD should not trigger a 1.1M sentence re-embedding. The repo
already has `scripts/build_probes.py` for this.

Recommended workflow:

```powershell
python scripts/build_probes.py --rubric artifacts/generated/rubric_new_role.json --profiles artifacts/generated/profiles_new_role.json --out artifacts/probes_new_role.npz
python rank.py --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --out submission.csv
```

### 6. Reduce Embedding Count Before Optimizing Inference

Precompute currently embeds about `1.1M` evidence sentences. That is only feasible
because the evidence selector caps to 12 sentences per candidate.

Optimization options, in quality-risk order:

1. Safe: deduplicate identical evidence sentences globally before embedding, then
   scatter vectors back to candidate offsets.
2. Safe: content-hash cache for incremental ingest/precompute reruns.
3. Medium risk: cap to `8-10` evidence sentences instead of `12`, but only after
   running `scripts/audit_evidence_cap.py`.
4. Medium risk: role/JD-aware evidence retention for the fixed JD.
5. High risk: candidate-level single blob embedding. Do not do this; it reopens
   the keyword-stuffing trap.

### 7. Use Token/Word Clipping, But Audit Hidden-Gem Recall

`MAX_PASSAGE_WORDS = 80` is a good start. The local sample average is only about
15 words per retained sentence, so clipping is not the main current bottleneck.

Recommended:

- Keep word clipping.
- Add token-length telemetry: p50/p90/p99 token count before embedding.
- Consider reducing to `48-64` words if audit shows no predicate loss.
- Never clip by taking only profile summary; keep career-history evidence.

### 8. Quantize Model Weights And Artifacts Separately

FastEmbed already uses ONNX Runtime and quantized model weights for supported
models. That optimizes model inference.

The repo's output vectors are still stored as `float16`, which is sensible for
quality and simple dot products. For rank-time speed and disk size, artifact
quantization is a separate decision:

- `float16` evidence vectors: current, simple, high fidelity.
- scalar `int8` evidence vectors: smaller/faster IO; requires per-dimension scale
  calibration and quality audit.
- binary vectors: much smaller but likely too lossy for fuzzy predicate scoring.

Recommended:

- Keep `float16` as the canonical artifact for now.
- Experiment with `int8` copies only behind a flag and compare top-100 stability,
  predicate firing stability, honeypot leakage, and hidden-gem recall.

Primary sources:

- ONNX Runtime quantization docs:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Sentence Transformers embedding quantization docs:
  https://www.sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html

### 9. Consider OpenVINO Only If Target CPU Is Intel

Sentence Transformers supports ONNX and OpenVINO backends. OpenVINO is mainly an
Intel optimization path. The current machine appears to be Windows with a
non-Intel Ryzen-class CPU from earlier repo notes, so OpenVINO is not the first
optimization to chase here.

Primary sources:

- Sentence Transformers inference efficiency:
  https://sbert.net/docs/sentence_transformer/usage/efficiency.html
- ONNX Runtime OpenVINO Execution Provider:
  https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html

### 10. Static Embeddings Are Interesting, But Risky For This Challenge

Static embedding models can be much faster on CPU. Hugging Face's Sentence
Transformers post reports 100x-400x CPU speedups while retaining much of the
quality.

For this repo, that is a research branch, not a default change. The challenge
hinges on subtle semantic evidence like "built recommendation-style features" and
"ranking evaluation"; replacing BGE with static embeddings could lose the exact
plain-language hidden gems the JD warns about.

Use static embeddings only for first-stage recall experiments, not predicate
judgment, unless audits prove quality.

Primary source:

- Hugging Face static embeddings post:
  https://huggingface.co/blog/static-embeddings

## Recommended Implementation Plan

### Phase 1: Benchmark Harness

Add `scripts/bench_cpu_embeddings.py`:

- loads real evidence sentences from the candidate file
- supports `--sample-sentences`
- benchmarks:
  - `threads`: `1,2,4,8,12,16`
  - `batch_size`: `4,8,16,32,64`
  - optional `workers`: `1,2,4`
- prints:
  - model load time
  - warmup time
  - sentences/sec
  - p50/p90 text length
  - peak RSS if available
- writes CSV to `output/embedding_bench.csv`

Success criterion:

- Pick defaults from data, not intuition.
- On current machine, initial evidence points to `threads=8`, `batch_size=4-8`.

### Phase 2: Precompute Defaults

After benchmark:

- set `precompute.py --batch-size` default to the measured winner
- keep `--threads` default near the measured winner
- set `--workers 1` as the default unless multiprocessing wins
- make worker thread count configurable instead of hardcoded `threads=2`

Likely starting config:

```powershell
python precompute.py --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --all-candidates --threads 8 --batch-size 8 --workers 1
```

### Phase 3: Dedup And Cache

Add a global evidence sentence hash cache:

- normalize whitespace
- hash sentence text
- embed unique sentences only
- reconstruct `evidence_vectors.npy` in original offset order

This is safe because identical sentence text should have identical embeddings.

Expected gain depends on dataset duplication. Measure first:

```powershell
python scripts/count_sentence_dupes.py
```

### Phase 4: Rank-Time IO Cleanup

Rank-time is already fast, but artifact loading can be cleaner:

- use `np.load(..., mmap_mode="r")` for `evidence_vectors.npy`
- avoid converting large arrays to float32 globally
- convert only recall/scoring slices
- consider storing `mean_vecs` in both fp16 and optional int8/normalized layout

This is secondary because current rank is already about 13 seconds.

### Phase 5: Quality Guardrail Suite

Every optimization must run these audits:

1. `rank.py` top-100 runtime under 5 minutes.
2. no non-technical AI-keyword stuffers in top 100.
3. honeypot/impossible profile leakage check.
4. hidden-gem check: recommendation/search/ranking profiles without shiny
   keywords still surface.
5. top-100 overlap versus current baseline.
6. top-10 manual inspection.

Minimum acceptance:

- top-10 does not degrade qualitatively
- no new keyword-stuffer leak
- no obvious hidden-gem loss
- final rank remains reproducible CPU-only/no-network

Current implementation:

- `scripts/audit_quality_gates.py`: submission-level pass/fail gate.
- `scripts/audit_honeypots.py`: pool inventory plus top-100 leakage audit.

Current full-pool trap inventory on `100000` candidates:

| Trap class | Count |
|---|---:|
| impossible skill duration vs career length | `343` |
| company before founding date | `93` |
| stated YOE impossible vs reconstructed timeline | `25` |
| zero-month advanced/expert skill cluster | `21` |
| non-technical AI keyword stuffer candidates | `3992` |
| stale 6mo plus low response | `1420` |
| stale 8mo | `1643` |
| passive 5 percent response | `368` |
| notice period over 90 days | `30551` |
| plain-language hidden-gem candidates | `666` |

Current guarded top-100 result:

- impossible profiles: `0`
- severe behavioral traps: `0`
- non-technical keyword stuffers: `0`
- plain-language hidden gems: `58`
- notice period over 90 days: `21`

Interpretation:

- Long notice period is a logistics penalty, not a honeypot label. It should
  down-rank close calls but not automatically disqualify a candidate whose job
  evidence is otherwise much stronger.
- Stale activity plus very low recruiter response is a hard availability trap.
  That now caps availability directly in `src/verdict/availability.py`.
- Keyword-stuffer detection must look at career family and corroborated evidence,
  not just whether the skills list contains AI terms.

Run before accepting faster embedding changes:

```powershell
.\.venv\Scripts\python.exe rank.py --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --out output\submission_guardrail_test.csv
.\.venv\Scripts\python.exe "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py" output\submission_guardrail_test.csv
.\.venv\Scripts\python.exe scripts\audit_quality_gates.py --submission output\submission_guardrail_test.csv --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
.\.venv\Scripts\python.exe scripts\audit_honeypots.py --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --submission output\submission_guardrail_test.csv
```

Research basis:

- LinkedIn's job-matching retrieval paper argues that job matching is not only
  semantic similarity: it needs qualification constraints, explainability, and
  adjustable matching rules.
- LinkedIn's activity-feature work supports using member/job-seeking behavior as
  relevance evidence rather than relying only on static profile text.

## Concrete Next Patch Candidates

1. Add benchmark harness.
2. Change precompute defaults only after benchmark CSV confirms.
3. Add sentence dedupe measurement script.
4. Add optional dedupe cache in precompute.
5. Add `mmap_mode="r"` in `rank.py` for `evidence_vectors.npy`.
6. Add a `scripts/build_probes.py` workflow section to README for "new JD, no
   candidate re-embedding."

## Do Not Do

- Do not embed skills-list text into scoring vectors.
- Do not move embeddings into rank-time.
- Do not switch to one big profile embedding.
- Do not optimize by removing behavioral/credibility scoring.
- Do not choose a faster model without hidden-gem and honeypot audits.
