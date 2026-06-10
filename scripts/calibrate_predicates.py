"""Inspect predicate similarity distributions on a sample to calibrate scoring.

Prints, per candidate: title family, per-predicate (pos_max, neg_max, pos-neg)
so we can see separation between genuinely relevant profiles and noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.evidence import build_record

ART = ROOT / "artifacts"

rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
sent_vecs = np.load(ART / "evidence_vectors.npy").astype(np.float32)
counts = np.load(ART / "sent_counts.npy")
probes = np.load(ART / "probes.npz")
artifact_ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
id_to_artifact = {cid: i for i, cid in enumerate(artifact_ids)}
offsets = np.zeros(len(counts) + 1, dtype=np.int64)
np.cumsum(counts, out=offsets[1:])

neg = probes["neg"].astype(np.float32)
pred_ids = list(rubric["fuzzy_predicates"].keys())
pred_vecs = {p: probes[f"pred_{p}"].astype(np.float32) for p in pred_ids}

cands = list(iter_candidates(sys.argv[1]))
print(f"{'id':>14} {'family':<16}", "  ".join(p[:14] for p in pred_ids))
rows = []
for c in cands:
    rec = build_record(c)
    i = id_to_artifact.get(rec.candidate_id)
    if i is None:
        continue
    sv = sent_vecs[offsets[i]:offsets[i + 1]]
    if not len(sv):
        continue
    neg_max_per_sent = (sv @ neg.T).max(axis=1)
    cells = []
    for p in pred_ids:
        pos = (sv @ pred_vecs[p].T).max(axis=1)
        diff = pos - neg_max_per_sent
        cells.append((float(pos.max()), float(diff.max())))
    rows.append((rec.candidate_id, rec.current_family, cells))
    print(f"{rec.candidate_id:>14} {rec.current_family:<16}",
          "  ".join(f"{p:.2f}/{d:+.2f}" for p, d in cells))

# distribution summary of the contrastive diff
print("\nper-predicate diff percentiles over sample (p50 / p75 / p90 / max):")
for k, p in enumerate(pred_ids):
    diffs = sorted(r[2][k][1] for r in rows)
    n = len(diffs)
    print(f"  {p:<32} {diffs[n//2]:+.3f} / {diffs[int(n*.75)]:+.3f} / {diffs[int(n*.9)]:+.3f} / {diffs[-1]:+.3f}")
