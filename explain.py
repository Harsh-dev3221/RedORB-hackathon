"""Explain the full VERDICT breakdown for one candidate: J x C x A, rule by rule,
with the actual evidence sentences that fired. The recruiter-trust surface.

  python explain.py CAND_0010257
  python explain.py CAND_0010257 --json     # machine-readable (used by the API)
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict.availability import score_availability
from verdict.credibility import score_credibility
from verdict.evidence import record_from_dict
from verdict.judgment import judge, predicate_scores

ART = Path(__file__).parent / "artifacts"


def load_candidate(cid: str):
    ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    try:
        idx = ids.index(cid)
    except ValueError:
        raise SystemExit(f"{cid} not in index ({len(ids)} candidates)")
    counts = np.load(ART / "sent_counts.npy")
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    vecs = np.load(ART / "evidence_vectors.npy", mmap_mode="r")
    sv = np.asarray(vecs[offsets[idx] : offsets[idx + 1]], dtype=np.float32)
    rec = None
    with gzip.open(ART / "records.jsonl.gz", "rb") as f:
        for line in f:
            if line.strip():
                d = orjson.loads(line)
                if d["candidate_id"] == cid:
                    rec = record_from_dict(d)
                    break
    if rec is None:
        raise SystemExit(f"{cid} missing from records.jsonl.gz")
    return rec, sv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    probes = np.load(ART / "probes.npz")
    rec, sv = load_candidate(args.candidate_id)

    neg = probes["neg"].astype(np.float32)
    pred_vecs = {p: probes[f"pred_{p}"].astype(np.float32) for p in rubric["fuzzy_predicates"]}
    preds = predicate_scores(sv, pred_vecs, neg, rubric["predicate_scoring"])
    j, rules, notes, dnotes = judge(rec, preds, rubric)
    c, cflags = score_credibility(rec)
    a, aflags = score_availability(rec, rubric)
    f = rubric["fusion"]
    fused = math.exp(
        f["alpha"] * math.log(max(j, 1e-4))
        + f["beta"] * math.log(max(c, 1e-4))
        + f["gamma"] * math.log(max(a, 1e-4))
    )

    if args.json:
        print(orjson.dumps({
            "candidate_id": rec.candidate_id,
            "title": rec.current_title, "family": rec.current_family,
            "yoe": max(rec.yoe_stated, rec.yoe_timeline),
            "location": rec.location_bucket,
            "J": round(j, 4), "C": round(c, 4), "A": round(a, 4),
            "score": round(fused, 6),
            "rules": {k: round(v, 3) for k, v in rules.items()},
            "evidence": notes, "dampeners": dnotes,
            "credibility_flags": cflags, "availability_flags": aflags,
        }, option=orjson.OPT_INDENT_2).decode())
        return

    w = {**{k: v["weight"] for k, v in rubric["crisp_rules"].items()},
         **{k: v["weight"] for k, v in rubric["fuzzy_predicates"].items()}}
    print(f"\n{'='*74}")
    print(f"{rec.candidate_id} | {rec.current_title} | {max(rec.yoe_stated, rec.yoe_timeline):.1f} yrs | {rec.location_bucket}")
    print(f"{'='*74}")
    print(f"VERDICT: score={fused:.4f}   J={j:.3f}^{f['alpha']} x C={c:.3f}^{f['beta']} x A={a:.3f}^{f['gamma']}\n")
    print("JUDGMENT (rule x weight = contribution):")
    for k, v in sorted(rules.items(), key=lambda x: -w.get(x[0], 0) * x[1]):
        print(f"  {k:<34} {v:5.2f} x {w.get(k, 0):.2f} = {v * w.get(k, 0):.3f}")
    if notes:
        print("\nEVIDENCE FIRED:")
        for n in notes:
            print(f"  + {n}")
    if dnotes:
        print("\nDAMPENERS APPLIED:")
        for d in dnotes:
            print(f"  ! {d}")
    print("\nCREDIBILITY:", f"C={c:.3f}")
    for fl in cflags or ["  no flags - claims internally consistent"]:
        print(f"  {fl}")
    print("\nAVAILABILITY:", f"A={a:.3f}")
    sig = rec.signals
    print(f"  last_active={sig.get('last_active_date', '?')}  response_rate={sig.get('recruiter_response_rate', '?')}  "
          f"notice={rec.notice_days}d  mode={rec.work_mode}")
    for fl in aflags:
        print(f"  {fl}")
    print()


if __name__ == "__main__":
    main()
