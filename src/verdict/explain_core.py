"""Single-candidate explanation: the J x C x A breakdown as structured data.

Used by explain.py (CLI), dossier.py (shortlist export), and the API.
"""

from __future__ import annotations

import math

from .pipeline import Index, score_candidate


def build_explanation(idx: Index, cid: str, rubric: dict, probes: dict) -> dict:
    i = idx.id_to_idx.get(cid)
    if i is None:
        raise KeyError(f"{cid} not in index ({len(idx.ids)} candidates)")
    s = score_candidate(idx, i, rubric, probes)
    rec = idx.records[i]
    f = rubric["fusion"]
    fused = math.exp(
        f["alpha"] * math.log(max(s.j, 1e-4))
        + f["beta"] * math.log(max(s.c, 1e-4))
        + f["gamma"] * math.log(max(s.a, 1e-4))
    )
    weights = {**{k: v["weight"] for k, v in rubric["crisp_rules"].items()},
               **{k: v["weight"] for k, v in rubric["fuzzy_predicates"].items()}}
    contributions = sorted(
        ({"rule": k, "score": round(v, 3), "weight": weights.get(k, 0.0),
          "contribution": round(v * weights.get(k, 0.0), 4)}
         for k, v in s.rule_scores.items()),
        key=lambda x: -x["contribution"],
    )
    sig = rec.signals
    return {
        "candidate_id": cid,
        "title": rec.current_title,
        "family": rec.current_family,
        "yoe": round(max(rec.yoe_stated, rec.yoe_timeline), 1),
        "location": rec.location_bucket,
        "J": round(s.j, 4), "C": round(s.c, 4), "A": round(s.a, 4),
        "score": round(fused, 6),
        "contributions": contributions,
        "evidence": s.evidence_notes,
        "dampeners": s.dampener_notes,
        "flags": s.flags,
        "behavior": {
            "last_active": sig.get("last_active_date"),
            "response_rate": sig.get("recruiter_response_rate"),
            "notice_days": rec.notice_days,
            "work_mode": rec.work_mode,
            "expected_lpa": [rec.salary_min, rec.salary_max],
        },
    }
