"""Build artifacts/records.jsonl.gz for the current artifacts/candidate_ids.txt.

This avoids rerunning embeddings when we only changed the LedgerRecord schema or
rank wants the fast startup ledger.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import orjson
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.evidence import build_record, record_to_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=str(ART / "records.jsonl.gz"))
    args = ap.parse_args()

    wanted = set((ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines())
    found = []
    t0 = time.time()
    for c in tqdm(iter_candidates(args.candidates), desc="building records"):
        cid = c.get("candidate_id")
        if cid in wanted:
            found.append(build_record(c))
    found.sort(key=lambda r: r.candidate_id)
    order = {cid: i for i, cid in enumerate((ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines())}
    found.sort(key=lambda r: order[r.candidate_id])

    out = Path(args.out)
    with gzip.open(out, "wb") as f:
        for rec in found:
            f.write(orjson.dumps(record_to_dict(rec)))
            f.write(b"\n")
    print(f"wrote {out} with {len(found)}/{len(wanted)} records in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
