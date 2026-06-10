"""Recall: hard gates -> {ABM, BM25, dense} -> RRF fusion -> recall set.

Three channels (JUDE pattern): structured attribute matches, lexical BM25,
dense similarity to hypothetical ideal profiles. Fused with reciprocal rank
fusion - no score normalization needed.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .evidence import LedgerRecord

_TOKEN = re.compile(r"[a-z0-9+#\-]{2,}")
_STOP = frozenset(
    "the a an and or of to in for with on at by from as is are was were be been "
    "this that it its we our i my you your they their he she his her them us".split()
)


def passes_gates(rec: LedgerRecord, gates: dict) -> bool:
    if rec.n_jobs < gates["min_career_entries"]:
        return False
    yoe = max(rec.yoe_stated, rec.yoe_timeline)
    if not (gates["yoe_min"] <= yoe <= gates["yoe_max"]):
        return False
    if (
        gates["reject_abroad_without_relocation"]
        and rec.location_bucket == "abroad"
        and not rec.willing_to_relocate
    ):
        return False
    return True


def abm_score(rec: LedgerRecord, abm: dict) -> float:
    """Structured attribute match strength (used to rank the ABM channel)."""
    score = 0.0
    fams = set(abm["families"])
    if rec.current_family in fams:
        score += 2.0
    elif any(f in fams for f in rec.families):
        score += 1.0
    core = rec.corroborated_categories & set(abm["core_categories"])
    score += min(len(core), 5) * 0.6
    return score


def bm25_rank(texts: list[str], query: str, k1: float = 1.4, b: float = 0.75) -> np.ndarray:
    """Return BM25 scores for each doc in texts against the query."""
    q_terms = [t for t in _TOKEN.findall(query.lower()) if t not in _STOP]
    n = len(texts)
    doc_tfs: list[Counter] = []
    doc_lens = np.empty(n, dtype=np.float32)
    df: Counter = Counter()
    qset = set(q_terms)
    for i, txt in enumerate(texts):
        toks = [t for t in _TOKEN.findall(txt) if t not in _STOP]
        doc_lens[i] = len(toks) or 1
        tf = Counter(t for t in toks if t in qset)  # only query terms needed
        doc_tfs.append(tf)
        for t in tf:
            df[t] += 1
    avgdl = float(doc_lens.mean()) if n else 1.0
    scores = np.zeros(n, dtype=np.float32)
    for t in q_terms:
        d = df.get(t, 0)
        if d == 0:
            continue
        idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
        for i, tf in enumerate(doc_tfs):
            f = tf.get(t, 0)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * doc_lens[i] / avgdl))
    return scores


def rrf_fuse(rankings: list[np.ndarray], n: int, k: int = 60) -> np.ndarray:
    """rankings: list of index arrays (best first). Returns fused scores [n]."""
    fused = np.zeros(n, dtype=np.float32)
    for order in rankings:
        ranks = np.empty(len(order), dtype=np.float32)
        ranks[:] = np.arange(1, len(order) + 1)
        fused[order] += 1.0 / (k + ranks)
    return fused


def run_recall(
    records: list[LedgerRecord],
    mean_vecs: np.ndarray,           # [n, 384] candidate mean narrative vectors (gate-survivor order)
    ideal_vecs: np.ndarray,          # [n_ideal, 384]
    rubric: dict,
) -> np.ndarray:
    """Return indices (into records) of the recall set, best-first."""
    n = len(records)
    # channel 1: ABM
    abm = np.array([abm_score(r, rubric["abm"]) for r in records], dtype=np.float32)
    abm_order = np.argsort(-abm, kind="stable")
    abm_order = abm_order[abm[abm_order] > 0]
    # channel 2: BM25
    bm = bm25_rank([r.narrative_text for r in records], rubric["bm25_query"])
    bm_order = np.argsort(-bm, kind="stable")[: max(n // 4, 4000)]
    # channel 3: dense vs ideal personas (max-sim over ideals)
    sims = mean_vecs @ ideal_vecs.T  # [n, n_ideal]
    dense = sims.max(axis=1)
    dense_order = np.argsort(-dense, kind="stable")[: max(n // 4, 4000)]

    fused = rrf_fuse([abm_order, bm_order, dense_order], n, k=rubric["recall"]["rrf_k"])
    top = int(rubric["recall"]["recall_top"])
    order = np.argsort(-fused, kind="stable")
    return order[: min(top, n)]
