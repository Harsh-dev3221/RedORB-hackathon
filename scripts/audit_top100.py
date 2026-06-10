"""Audit a submission CSV against the ledger: honeypot leak, trap exposure, tone.

Usage: python scripts/audit_top100.py output/submission_full.csv
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.credibility import score_credibility
from verdict.evidence import record_from_dict

ART = ROOT / "artifacts"

with open(sys.argv[1], encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
want = {r["candidate_id"]: int(r["rank"]) for r in rows}

recs = {}
with gzip.open(ART / "records.jsonl.gz", "rb") as f:
    for line in f:
        if line.strip():
            d = orjson.loads(line)
            if d["candidate_id"] in want:
                recs[d["candidate_id"]] = record_from_dict(d)

n_impossible = 0
n_suspicious = 0
n_low_c = 0
worst = []
for cid, rank in sorted(want.items(), key=lambda x: x[1]):
    rec = recs.get(cid)
    if rec is None:
        print(f"rank {rank}: {cid} MISSING from ledger!")
        continue
    c, flags = score_credibility(rec)
    if rec.impossibilities:
        n_impossible += 1
        worst.append((rank, cid, "IMPOSSIBLE", rec.impossibilities[0]))
    elif rec.suspicions:
        n_suspicious += 1
    if c < 0.3:
        n_low_c += 1
        if not rec.impossibilities:
            worst.append((rank, cid, f"C={c:.2f}", (flags or ["?"])[0][:80]))

print(f"top-{len(want)} audit:")
print(f"  candidates with IMPOSSIBLE profiles (honeypot signature): {n_impossible}")
print(f"  candidates with minor suspicions: {n_suspicious}")
print(f"  candidates with credibility < 0.3: {n_low_c}")
for rank, cid, kind, detail in worst[:15]:
    print(f"  rank {rank:>3} {cid} [{kind}] {detail}")

# pool-wide: how many impossible profiles exist at all, and did we exclude them
n_pool_impossible = 0
with gzip.open(ART / "records.jsonl.gz", "rb") as f:
    for line in f:
        if line.strip():
            d = orjson.loads(line)
            if d.get("impossibilities"):
                n_pool_impossible += 1
print(f"\npool-wide profiles with impossibilities: {n_pool_impossible} (none should be in top-100)")
