"""Index drift monitor: snapshot score distributions; alert when probe/threshold
recalibration is due (PSI > 0.25).

  python drift_monitor.py record    # take a snapshot (run after big ingests)
  python drift_monitor.py check     # compare latest vs baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict import drift
from verdict.pipeline import ART, load_index, load_probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["record", "check"])
    args = ap.parse_args()

    if args.command == "record":
        rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
        idx = load_index()
        probes = load_probes(ART / "probes.npz", rubric)
        snap = drift.record(idx, rubric, probes)
        print(orjson.dumps(snap, option=orjson.OPT_INDENT_2).decode())
    else:
        print(orjson.dumps(drift.check(), option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
