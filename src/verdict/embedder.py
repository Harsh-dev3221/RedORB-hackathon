"""Offline embedding via fastembed (ONNX, CPU). Model: BAAI/bge-small-en-v1.5.

Chosen as the best CPU-supported model for this machine (Ryzen 9 5900HX, 16GB):
384-d, int8 ONNX, ~no install weight (no torch). Embeddings are computed ONLY
offline (precompute.py); the rank step consumes saved vectors - it never loads
a model.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
MAX_PASSAGE_WORDS = 80


def get_model(threads: int | None = None, cuda: bool = False):
    from pathlib import Path

    from fastembed import TextEmbedding
    if cuda:
        import onnxruntime as ort

        ort.preload_dlls(directory="")

    cache = Path(__file__).resolve().parents[2] / ".cache" / "fastembed"
    cache.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=MODEL_NAME,
        threads=threads,
        cache_dir=str(cache),
        cuda=cuda,
        lazy_load=True,
    )


def _clip_passage(text: str) -> str:
    """Bound tokenization/padding cost while preserving the concrete evidence."""
    words = text.split()
    if len(words) <= MAX_PASSAGE_WORDS:
        return text
    return " ".join(words[:MAX_PASSAGE_WORDS])


def embed_passages(
    model, texts: list[str], batch_size: int = 256, parallel: int | None = None
) -> np.ndarray:
    """L2-normalized float32 [n, 384]. parallel>1 = data-parallel worker processes."""
    texts = [_clip_passage(t) for t in texts]
    out = np.empty((len(texts), DIM), dtype=np.float32)
    i = 0
    for vec in model.embed(texts, batch_size=batch_size, parallel=parallel):
        out[i] = vec
        i += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, np.maximum(norms, 1e-9), out=out)
    return out


def embed_queries(model, texts: list[str]) -> np.ndarray:
    """BGE query-side embedding (instruction prefix handled by fastembed)."""
    vecs = list(model.query_embed(texts))
    out = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)
