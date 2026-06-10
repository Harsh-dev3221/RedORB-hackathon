"""Upload-time twin detection: flag near-duplicates of profiles already indexed.

Two channels, mirroring the claim-trust philosophy:
  EVIDENCE_TWIN - narrative mean-vector cosine ~ 1.0 (copied/templated career text)
  SIGNAL_TWIN   - behavioral signal vector nearly identical (cloned platform telemetry)

At 100K scale exact brute-force numpy beats LSH (one matmul, milliseconds);
LSH only earns its complexity past ~10M profiles.
"""

from __future__ import annotations

import numpy as np

from .evidence import LedgerRecord

EVIDENCE_COS_THRESHOLD = 0.985
SIGNAL_REL_TOLERANCE = 0.02

_SIGNAL_SPEC: list[tuple[str, float]] = [  # (signal, typical scale)
    ("profile_completeness_score", 100.0),
    ("profile_views_received_30d", 50.0),
    ("applications_submitted_30d", 20.0),
    ("recruiter_response_rate", 1.0),
    ("avg_response_time_hours", 96.0),
    ("connection_count", 1000.0),
    ("endorsements_received", 200.0),
    ("github_activity_score", 100.0),
    ("search_appearance_30d", 50.0),
    ("saved_by_recruiters_30d", 10.0),
    ("interview_completion_rate", 1.0),
    ("offer_acceptance_rate", 1.0),
]


def signal_vector(rec: LedgerRecord) -> np.ndarray | None:
    sig = rec.signals
    if not sig:
        return None
    return np.asarray(
        [float(sig.get(k) or 0.0) / scale for k, scale in _SIGNAL_SPEC],
        dtype=np.float32,
    )


def find_twins(
    new_records: list[LedgerRecord],
    new_means: np.ndarray,            # fp16/32 [m, 384]
    index_ids: list[str],
    index_means: np.ndarray,          # fp16 [n, 384]
    index_records: list[LedgerRecord] | None = None,
) -> list[list[str]]:
    """Return per-new-candidate twin flags ('EVIDENCE_TWIN: <id> cos=0.998', ...)."""
    flags: list[list[str]] = [[] for _ in new_records]
    if len(index_ids) == 0:
        return flags

    means = index_means.astype(np.float32)
    sims = new_means.astype(np.float32) @ means.T          # [m, n]
    for k in range(len(new_records)):
        j = int(np.argmax(sims[k]))
        cos = float(sims[k, j])
        if cos >= EVIDENCE_COS_THRESHOLD:
            flags[k].append(f"EVIDENCE_TWIN: narrative nearly identical to {index_ids[j]} (cos={cos:.3f})")

    if index_records is not None:
        sv_existing, sv_ids = [], []
        for rec in index_records:
            v = signal_vector(rec)
            if v is not None:
                sv_existing.append(v)
                sv_ids.append(rec.candidate_id)
        if sv_existing:
            mat = np.stack(sv_existing)                    # [n, d]
            for k, rec in enumerate(new_records):
                v = signal_vector(rec)
                if v is None:
                    continue
                d = np.abs(mat - v).max(axis=1)            # Chebyshev in scaled space
                j = int(np.argmin(d))
                if float(d[j]) <= SIGNAL_REL_TOLERANCE and sv_ids[j] != rec.candidate_id:
                    flags[k].append(
                        f"SIGNAL_TWIN: behavioral telemetry matches {sv_ids[j]} (max rel-diff {float(d[j]):.3f})"
                    )
    return flags
