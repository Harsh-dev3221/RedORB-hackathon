"""Does the 8-sentence keyword-weighted evidence selector drop predicate-winning
sentences? (Hidden-gem risk: plain-language evidence without scorer keywords.)

Samples candidates whose raw narrative has >8 sentences, embeds kept vs dropped
sentences, and measures how often a DROPPED sentence would have raised any
fuzzy-predicate score materially.

Usage: python scripts/audit_evidence_cap.py <candidates.jsonl> [n_sample]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.embedder import embed_passages, get_model
from verdict.evidence import build_record, split_sentences

ART = ROOT / "artifacts"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 250

rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
probes = np.load(ART / "probes.npz")
neg = probes["neg"].astype(np.float32)
pred_vecs = {p: probes[f"pred_{p}"].astype(np.float32) for p in rubric["fuzzy_predicates"]}
floor = rubric["predicate_scoring"]["sim_floor"]

cases = []  # (cid, kept_sentences, dropped_sentences)
for c in iter_candidates(sys.argv[1]):
    rec = build_record(c)
    raw: list[str] = []
    for j in c.get("career_history") or []:
        raw.extend(split_sentences(j.get("description", "") or ""))
    raw.extend(split_sentences((c.get("profile") or {}).get("summary", "") or ""))
    kept = set(rec.sentences)
    dropped = [s for s in raw if s not in kept]
    if dropped:
        cases.append((rec.candidate_id, rec.sentences, dropped))
    if len(cases) >= N:
        break

print(f"candidates with dropped sentences in sample: {len(cases)}")
model = get_model()
hurt = 0
big_hurt = 0
examples = []
for cid, kept, dropped in cases:
    kv = embed_passages(model, kept) if kept else np.zeros((0, 384), np.float32)
    dv = embed_passages(model, dropped)
    k_negmax = (kv @ neg.T).max(axis=1) if len(kv) else np.array([0.0])
    d_negmax = (dv @ neg.T).max(axis=1)
    worst_gap = 0.0
    worst = None
    for pid, pv in pred_vecs.items():
        k_best = float(((kv @ pv.T).max(axis=1) - k_negmax).max()) if len(kv) else -1.0
        d_adj = (dv @ pv.T).max(axis=1) - d_negmax
        d_best = float(d_adj.max())
        gap = d_best - max(k_best, floor)
        if gap > worst_gap:
            worst_gap = gap
            worst = (pid, dropped[int(np.argmax(d_adj))])
    if worst_gap > 0.01:
        hurt += 1
    if worst_gap > 0.03:
        big_hurt += 1
        if len(examples) < 5:
            examples.append((cid, worst_gap, worst))

print(f"selector dropped a better predicate sentence (>0.01 adj-sim): {hurt}/{len(cases)}")
print(f"materially better (>0.03, ~1/3 of scoring range):            {big_hurt}/{len(cases)}")
for cid, gap, (pid, sent) in examples:
    print(f"  {cid} +{gap:.3f} on {pid}: \"{sent[:100]}\"")
