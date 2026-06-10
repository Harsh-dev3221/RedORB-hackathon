"""PHASE RANK: candidates.jsonl + precomputed artifacts -> submission CSV.

Constraint-compliant by construction: CPU-only, no network, no model loading -
pure numpy over precomputed vectors. Target < 3 min on a 16 GB machine.

Usage:
  python rank.py --candidates path/to/candidates.jsonl --out submission.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict.availability import score_availability
from verdict.credibility import score_credibility
from verdict.data import iter_candidates
from verdict.evidence import build_record, record_from_dict
from verdict.fusion import Scored, finalize
from verdict.judgment import judge, predicate_scores
from verdict.reasoning import synthesize
from verdict.recall import passes_gates, run_recall

ART = Path(__file__).parent / "artifacts"


def _load_precomputed_records(pre_ids: list[str]) -> list | None:
    path = ART / "records.jsonl.gz"
    if not path.exists():
        return None
    records = []
    with gzip.open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(record_from_dict(orjson.loads(line)))
    if [r.candidate_id for r in records] != pre_ids:
        return None
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top", type=int, default=None, help="override final list size (sandbox/sample runs)")
    args = ap.parse_args()
    t_start = time.time()

    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    sent_vecs = np.load(ART / "evidence_vectors.npy")           # fp16 [S, 384]
    counts = np.load(ART / "sent_counts.npy")                   # [N]
    mean_vecs_all = np.load(ART / "mean_vecs.npy")              # fp16 [N, 384]
    pre_ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    probes = np.load(ART / "probes.npz")
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"artifacts loaded ({time.time()-t_start:.1f}s)")

    # ---- records: prefer precomputed ledger, fall back to raw candidate parsing ----
    t0 = time.time()
    precomputed = _load_precomputed_records(pre_ids)
    if precomputed is not None:
        # records.jsonl.gz may contain the full pool (--all-candidates builds);
        # hard gates must still apply on the rank path.
        survivors, surv_pool_idx = [], []
        for i, rec in enumerate(precomputed):
            if passes_gates(rec, rubric["gates"]):
                survivors.append(rec)
                surv_pool_idx.append(i)
        print(
            f"loaded {len(precomputed)} precomputed records, gates passed "
            f"{len(survivors)} ({time.time()-t0:.1f}s)"
        )
    else:
        survivors = []          # LedgerRecord
        surv_pool_idx = []      # index into precompute order
        id_to_pre = {cid: i for i, cid in enumerate(pre_ids)}
        n_seen = 0
        for c in iter_candidates(args.candidates):
            n_seen += 1
            rec = build_record(c)
            if passes_gates(rec, rubric["gates"]):
                pi = id_to_pre.get(rec.candidate_id)
                if pi is None:
                    continue  # candidate not in precompute (shouldn't happen)
                survivors.append(rec)
                surv_pool_idx.append(pi)
        print(f"parsed {n_seen}, gates passed {len(survivors)} ({time.time()-t0:.0f}s)")

    # ---- recall: ABM + BM25 + dense -> RRF ----
    t0 = time.time()
    surv_mean = mean_vecs_all[np.asarray(surv_pool_idx)].astype(np.float32)
    ideal = probes["ideal"].astype(np.float32)
    recall_idx = run_recall(survivors, surv_mean, ideal, rubric)
    print(f"recall set {len(recall_idx)} ({time.time()-t0:.0f}s)")

    # ---- full J x C x A scoring on the recall set ----
    t0 = time.time()
    neg = probes["neg"].astype(np.float32)
    pred_vecs = {
        pid: probes[f"pred_{pid}"].astype(np.float32)
        for pid in rubric["fuzzy_predicates"]
    }
    pcfg = rubric["predicate_scoring"]
    scored: list[Scored] = []
    for si in recall_idx:
        rec = survivors[int(si)]
        pi = surv_pool_idx[int(si)]
        sv = sent_vecs[offsets[pi] : offsets[pi + 1]].astype(np.float32)
        preds = predicate_scores(sv, pred_vecs, neg, pcfg)
        j, rules, notes, dnotes = judge(rec, preds, rubric)
        c_score, c_flags = score_credibility(rec, rubric.get("credibility"))
        a_score, a_flags = score_availability(rec, rubric)
        scored.append(
            Scored(
                idx=int(si), candidate_id=rec.candidate_id, j=j, c=c_score, a=a_score,
                rule_scores=rules, evidence_notes=notes, dampener_notes=dnotes,
                flags=c_flags + a_flags,
            )
        )
    print(f"scored {len(scored)} ({time.time()-t0:.0f}s)")

    # ---- fusion + tournament + reasoning ----
    if args.top is not None:
        rubric["fusion"]["final_size"] = args.top
    expected = min(int(rubric["fusion"]["final_size"]), len(scored))
    top = finalize(scored, rubric)
    rows = []
    for rank, it in enumerate(top, start=1):
        rec = survivors[it.idx]
        rows.append(
            {
                "candidate_id": it.candidate_id,
                "rank": rank,
                "score": f"{it.final:.6f}",
                "reasoning": synthesize(it, rec, rank),
            }
        )

    # ---- enforce spec invariants, write CSV ----
    _self_check(rows, expected)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} | total {time.time()-t_start:.0f}s")


def _self_check(rows: list[dict], expected: int) -> None:
    assert len(rows) == expected, f"need exactly {expected} rows, got {len(rows)}"
    ids = [r["candidate_id"] for r in rows]
    assert len(set(ids)) == expected, "duplicate candidate_id"
    scores = [float(r["score"]) for r in rows]
    for s1, s2 in zip(scores, scores[1:]):
        assert s1 >= s2, "scores must be non-increasing"
    for (r1, r2) in zip(rows, rows[1:]):
        if r1["score"] == r2["score"]:
            assert r1["candidate_id"] < r2["candidate_id"], "tie-break must be id-ascending"


if __name__ == "__main__":
    main()
