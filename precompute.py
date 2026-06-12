"""PHASE OFFLINE: build embedding artifacts (may exceed the 5-minute window).

Produces, under artifacts/:
  evidence_vectors.npy   float16 [total_sentences, 384]  sentence embeddings
  sent_counts.npy        int32   [n_candidates]          sentences per candidate
  candidate_ids.txt      one id per line (artifact order; gate survivors by default)
  mean_vecs.npy          float16 [n_artifact_candidates, 384] mean narrative vector
  records.jsonl.gz       normalized LedgerRecord rows for fast rank startup
  probes.npz             ideal personas + predicate/negative query vectors

Usage:
  python precompute.py --candidates path/to/candidates.jsonl [--threads 8]

Default speed path:
  hard gates -> ABM union BM25-top-20000 -> dense embeddings.
  Use --no-prefilter to embed every gate survivor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tqdm import tqdm

from verdict.data import iter_candidates
from verdict.embedder import DIM, embed_passages, embed_queries, get_model
from verdict.evidence import build_record, record_to_dict
from verdict.recall import abm_score, bm25_rank, passes_gates

ART = Path(__file__).parent / "artifacts"

_worker_model = None
_worker_batch_size = 8


def _init_worker(threads: int, batch_size: int):
    global _worker_model
    global _worker_batch_size
    _worker_model = get_model(threads=threads)
    _worker_batch_size = batch_size


def _embed_chunk(chunk: list[str]) -> bytes:
    # fp16 bytes keep IPC payloads small
    return embed_passages(_worker_model, chunk, batch_size=_worker_batch_size).astype(np.float16).tobytes()


def _sentence_hash(text: str) -> bytes:
    normalized = " ".join(text.split()).encode("utf-8")
    return hashlib.blake2b(normalized, digest_size=16).digest()


def _dedupe_sentences(sentences: list[str]) -> tuple[list[str], np.ndarray]:
    """Return unique first-seen sentences and inverse indices into that list."""
    seen: dict[bytes, int] = {}
    unique: list[str] = []
    inverse = np.empty(len(sentences), dtype=np.int64)
    for i, sent in enumerate(sentences):
        key = _sentence_hash(sent)
        idx = seen.get(key)
        if idx is None:
            idx = len(unique)
            seen[key] = idx
            unique.append(sent)
        inverse[i] = idx
    return unique, inverse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-threads", type=int, default=2)
    ap.add_argument("--no-dedupe", action="store_true", help="disable sentence hash dedupe before embedding")
    ap.add_argument("--cuda", action="store_true", help="use fastembed-gpu CUDA path for offline embedding")
    ap.add_argument(
        "--all-candidates",
        action="store_true",
        help="embed every candidate; default embeds only hard-gate survivors used by rank.py",
    )
    ap.add_argument(
        "--prefilter-top",
        type=int,
        default=20000,
        help="BM25 top-K gate survivors to embed, unioned with all ABM-positive survivors",
    )
    ap.add_argument(
        "--no-prefilter",
        action="store_true",
        help="embed all gate survivors instead of the faster ABM+BM25 prefilter",
    )
    args = ap.parse_args()

    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    hypo = orjson.loads((ART / "hypothetical_profiles.json").read_bytes())

    # ---- pass 1: extract evidence sentences (deterministic, same code as rank) ----
    t0 = time.time()
    ids: list[str] = []
    counts: list[int] = []
    sentences: list[str] = []
    records = []
    n_seen = 0
    for c in tqdm(iter_candidates(args.candidates), desc="extracting evidence"):
        n_seen += 1
        rec = build_record(c)
        if not args.all_candidates and not passes_gates(rec, rubric["gates"]):
            continue
        records.append(rec)
    if not args.all_candidates and not args.no_prefilter and records:
        abm = np.asarray([abm_score(r, rubric["abm"]) for r in records], dtype=np.float32)
        bm25 = bm25_rank([r.narrative_text for r in records], rubric["bm25_query"])
        k = min(args.prefilter_top, len(records))
        bm25_top = set(np.argsort(-bm25, kind="stable")[:k].tolist())
        keep = bm25_top | {i for i, score in enumerate(abm) if score > 0}
        records = [r for i, r in enumerate(records) if i in keep]
        print(
            f"prefilter kept {len(records)} candidates "
            f"(BM25 top {k} union ABM-positive {int((abm > 0).sum())})"
        )
    for rec in records:
        ids.append(rec.candidate_id)
        counts.append(len(rec.sentences))
        sentences.extend(rec.sentences)
    n = len(ids)
    scope = "all candidates" if args.all_candidates else f"gate survivors from {n_seen}"
    print(f"{n} {scope}, {len(sentences)} evidence sentences ({time.time()-t0:.0f}s)")

    # ---- embed (token-bound on CPU -> shard across worker processes) ----
    t0 = time.time()
    if args.no_dedupe:
        embed_sentences = sentences
        inverse = None
    else:
        embed_sentences, inverse = _dedupe_sentences(sentences)
        saved = len(sentences) - len(embed_sentences)
        pct = (100.0 * saved / len(sentences)) if sentences else 0.0
        print(f"sentence hash dedupe: {len(embed_sentences)} unique, saved {saved} embeds ({pct:.1f}%)")

    unique_vecs = np.empty((len(embed_sentences), DIM), dtype=np.float16)
    cs = 1024
    chunks = [embed_sentences[i : i + cs] for i in range(0, len(embed_sentences), cs)]
    if args.workers <= 1 or len(embed_sentences) < 4096:
        model = get_model(threads=args.threads, cuda=args.cuda)
        off = 0
        for ch in tqdm(chunks, desc="embedding"):
            out = embed_passages(model, ch, batch_size=args.batch_size).astype(np.float16)
            unique_vecs[off : off + len(out)] = out
            off += len(out)
    else:
        import multiprocessing as mp

        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.worker_threads, args.batch_size),
        ) as pool:
            off = 0
            for raw in tqdm(
                pool.imap(_embed_chunk, chunks), total=len(chunks), desc="embedding"
            ):
                out = np.frombuffer(raw, dtype=np.float16).reshape(-1, DIM)
                unique_vecs[off : off + len(out)] = out
                off += len(out)
    if inverse is None:
        vecs = unique_vecs
    else:
        vecs = unique_vecs[inverse]
    print(f"embedded {len(embed_sentences)} unique / {len(sentences)} total sentences in {time.time()-t0:.0f}s")

    model = get_model(threads=args.threads, cuda=args.cuda)  # for probe/query embedding below

    # ---- mean narrative vector per candidate ----
    mean_vecs = np.zeros((n, DIM), dtype=np.float16)
    off = 0
    for i, k in enumerate(counts):
        if k:
            m = vecs[off : off + k].astype(np.float32).mean(axis=0)
            norm = np.linalg.norm(m)
            if norm > 1e-9:
                mean_vecs[i] = (m / norm).astype(np.float16)
        off += k

    # ---- probes: ideals + predicate positives + negatives (query-side) ----
    ideal_vecs = embed_queries(model, hypo["ideal"])
    neg_vecs = embed_queries(model, rubric["predicate_negatives"])
    probes: dict[str, np.ndarray] = {"ideal": ideal_vecs, "neg": neg_vecs}
    for pid, cfg in rubric["fuzzy_predicates"].items():
        probes[f"pred_{pid}"] = embed_queries(model, cfg["positives"])

    ART.mkdir(exist_ok=True)
    np.save(ART / "evidence_vectors.npy", vecs)
    np.save(ART / "sent_counts.npy", np.asarray(counts, dtype=np.int32))
    np.save(ART / "mean_vecs.npy", mean_vecs)
    (ART / "candidate_ids.txt").write_text("\n".join(ids), encoding="utf-8")
    with gzip.open(ART / "records.jsonl.gz", "wb") as f:
        for rec in records:
            f.write(orjson.dumps(record_to_dict(rec)))
            f.write(b"\n")
    np.savez(ART / "probes.npz", **probes)
    print("artifacts written to", ART)


if __name__ == "__main__":
    main()
