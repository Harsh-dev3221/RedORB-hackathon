"""Benchmark CPU embedding throughput on real evidence sentences.

This is intentionally repo-local and dataset-shaped: synthetic fixed strings can
hide tokenization and sentence-length costs. Use it before changing
`precompute.py` defaults.

Example:
  python scripts/bench_cpu_embeddings.py --candidates ./candidates.jsonl.gz \
    --sample-sentences 800 --threads 4,8,12 --batch-sizes 4,8,16
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.embedder import embed_passages, get_model
from verdict.evidence import build_record


def _parse_ints(raw: str) -> list[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _load_sentences(path: str, limit: int) -> list[str]:
    texts: list[str] = []
    for cand in iter_candidates(path):
        texts.extend(build_record(cand).sentences)
        if len(texts) >= limit:
            break
    return texts[:limit]


def _pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--sample-sentences", type=int, default=800)
    ap.add_argument("--threads", type=_parse_ints, default="2,4,8,12")
    ap.add_argument("--batch-sizes", type=_parse_ints, default="4,8,16,32")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--out", default="output/embedding_bench.csv")
    args = ap.parse_args()

    texts = _load_sentences(args.candidates, args.sample_sentences)
    if not texts:
        raise SystemExit("no evidence sentences found")
    word_lens = [len(t.split()) for t in texts]
    print(
        f"loaded {len(texts)} sentences | words "
        f"mean={statistics.mean(word_lens):.1f} p50={_pct(word_lens, 0.50):.0f} "
        f"p90={_pct(word_lens, 0.90):.0f} p99={_pct(word_lens, 0.99):.0f}",
        flush=True,
    )

    rows: list[dict[str, str]] = []
    for threads in args.threads:
        t0 = time.time()
        model = get_model(threads=threads, cuda=False)
        model_init_s = time.time() - t0
        t0 = time.time()
        embed_passages(model, texts[: min(args.warmup, len(texts))], batch_size=min(args.warmup, len(texts)))
        warmup_s = time.time() - t0
        for batch_size in args.batch_sizes:
            t0 = time.time()
            embed_passages(model, texts, batch_size=batch_size)
            elapsed = time.time() - t0
            sent_s = len(texts) / max(elapsed, 1e-9)
            row = {
                "sample_sentences": str(len(texts)),
                "mean_words": f"{statistics.mean(word_lens):.2f}",
                "p90_words": f"{_pct(word_lens, 0.90):.0f}",
                "threads": str(threads),
                "batch_size": str(batch_size),
                "model_init_s": f"{model_init_s:.3f}",
                "warmup_s": f"{warmup_s:.3f}",
                "elapsed_s": f"{elapsed:.3f}",
                "sentences_per_s": f"{sent_s:.2f}",
            }
            rows.append(row)
            print(
                f"threads={threads:<2} batch={batch_size:<3} "
                f"{sent_s:8.2f} sent/s ({elapsed:.2f}s)",
                flush=True,
            )
        del model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
