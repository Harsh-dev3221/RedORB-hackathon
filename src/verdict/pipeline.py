"""Shared scoring pipeline over the artifact index.

One audited code path for every tool that needs full J x C x A scoring
(rubric_diff, drift monitor, dossier). Mirrors rank.py's flow; rank.py itself
stays frozen as the challenge-verified entry point.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson

from .availability import score_availability
from .credibility import score_credibility
from .evidence import LedgerRecord, record_from_dict
from .fusion import Scored, finalize
from .judgment import judge, predicate_scores
from .recall import passes_gates, run_recall

ART = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass
class Index:
    ids: list[str]
    counts: np.ndarray
    offsets: np.ndarray
    sent_vecs: np.ndarray      # fp16 memmap-able
    mean_vecs: np.ndarray      # fp16
    records: list[LedgerRecord]
    id_to_idx: dict[str, int]


def load_index(art: Path = ART, mmap: bool = True) -> Index:
    ids = (art / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    counts = np.load(art / "sent_counts.npy")
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    sent_vecs = np.load(art / "evidence_vectors.npy", mmap_mode="r" if mmap else None)
    mean_vecs = np.load(art / "mean_vecs.npy")
    records: list[LedgerRecord] = []
    with gzip.open(art / "records.jsonl.gz", "rb") as f:
        for line in f:
            if line.strip():
                records.append(record_from_dict(orjson.loads(line)))
    assert [r.candidate_id for r in records] == ids, "records/ids misaligned"
    return Index(ids, counts, offsets, sent_vecs, mean_vecs, records,
                 {cid: i for i, cid in enumerate(ids)})


def load_probes(path: Path, rubric: dict) -> dict[str, np.ndarray]:
    z = np.load(path)
    out = {"ideal": z["ideal"].astype(np.float32), "neg": z["neg"].astype(np.float32)}
    for pid in rubric["fuzzy_predicates"]:
        out[f"pred_{pid}"] = z[f"pred_{pid}"].astype(np.float32)
    return out


def score_candidate(idx: Index, i: int, rubric: dict, probes: dict) -> Scored:
    rec = idx.records[i]
    sv = np.asarray(idx.sent_vecs[idx.offsets[i] : idx.offsets[i + 1]], dtype=np.float32)
    pred_vecs = {pid: probes[f"pred_{pid}"] for pid in rubric["fuzzy_predicates"]}
    preds = predicate_scores(sv, pred_vecs, probes["neg"], rubric["predicate_scoring"])
    j, rules, notes, dnotes = judge(rec, preds, rubric)
    c, cflags = score_credibility(rec, rubric.get("credibility"))
    a, aflags = score_availability(rec, rubric)
    return Scored(idx=i, candidate_id=rec.candidate_id, j=j, c=c, a=a,
                  rule_scores=rules, evidence_notes=notes, dampener_notes=dnotes,
                  flags=cflags + aflags)


def rank_pipeline(idx: Index, rubric: dict, probes: dict) -> list[Scored]:
    """Full gates -> recall -> score -> fuse+tournament. Returns final top-N."""
    survivors, surv_idx = [], []
    for i, rec in enumerate(idx.records):
        if passes_gates(rec, rubric["gates"]):
            survivors.append(rec)
            surv_idx.append(i)
    surv_mean = idx.mean_vecs[np.asarray(surv_idx)].astype(np.float32)
    recall_idx = run_recall(survivors, surv_mean, probes["ideal"], rubric)
    scored = [score_candidate(idx, surv_idx[int(si)], rubric, probes) for si in recall_idx]
    return finalize(scored, rubric)
