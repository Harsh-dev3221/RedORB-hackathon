"""Rebuild ONLY the probe vectors (ideal personas + predicate/negative queries).

This is what makes the index JD-agnostic: candidate embeddings never depend on
the rubric, so ranking a brand-new JD = compile/review rubric -> run this
(~seconds) -> rank.py. No re-embedding of 100K candidates.

  python scripts/build_probes.py                              # default rubric
  python scripts/build_probes.py --rubric artifacts/generated/rubric_new_role.json \
      --profiles artifacts/generated/profiles_new_role.json --out artifacts/probes.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.embedder import embed_queries, get_model

ART = ROOT / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", default=str(ART / "rubric_program.json"))
    ap.add_argument("--profiles", default=str(ART / "hypothetical_profiles.json"))
    ap.add_argument("--out", default=str(ART / "probes.npz"))
    args = ap.parse_args()

    rubric = orjson.loads(Path(args.rubric).read_bytes())
    hypo = orjson.loads(Path(args.profiles).read_bytes())
    model = get_model()
    probes: dict[str, np.ndarray] = {
        "ideal": embed_queries(model, hypo["ideal"]),
        "neg": embed_queries(model, rubric["predicate_negatives"]),
    }
    for pid, cfg in rubric["fuzzy_predicates"].items():
        probes[f"pred_{pid}"] = embed_queries(model, cfg["positives"])
    np.savez(args.out, **probes)
    n_q = sum(v.shape[0] for v in probes.values())
    print(f"wrote {args.out}: {len(probes)} probe groups, {n_q} query vectors")


if __name__ == "__main__":
    main()
