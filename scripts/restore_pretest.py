"""One-off: restore index to pre-ingest-test state recorded in output/pretest_state.txt."""

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
ART = ROOT / "artifacts"

n_target, gz_target = [int(x) for x in (ROOT / "output" / "pretest_state.txt").read_text().split()]
ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
counts = np.load(ART / "sent_counts.npy")
assert len(ids) == n_target + 1 and ids[-1] == "UPL_TEST001", "unexpected index state"
s_keep = int(counts[:n_target].sum())

np.save(ART / "tmp.npy", np.load(ART / "evidence_vectors.npy")[:s_keep])
os.replace(ART / "tmp.npy", ART / "evidence_vectors.npy")
np.save(ART / "tmp.npy", counts[:n_target])
os.replace(ART / "tmp.npy", ART / "sent_counts.npy")
np.save(ART / "tmp.npy", np.load(ART / "mean_vecs.npy")[:n_target])
os.replace(ART / "tmp.npy", ART / "mean_vecs.npy")
(ART / "candidate_ids.txt").write_text("\n".join(ids[:n_target]), encoding="utf-8")
with open(ART / "records.jsonl.gz", "r+b") as f:
    f.truncate(gz_target)
print(f"restored: {n_target} candidates, records.gz {gz_target} bytes")
