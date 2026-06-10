"""Shortlist dossier: top-N candidates as a markdown pack a recruiter can
forward to a hiring manager - each with the full evidence breakdown.

  python dossier.py --submission output/submission_full.csv --top 10 --out output/dossier.md
  python dossier.py --top 10                # re-ranks live instead of reading a CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict.explain_core import build_explanation
from verdict.pipeline import ART, load_index, load_probes, rank_pipeline

_AXIS_BAR = 14


def _bar(x: float) -> str:
    k = round(max(0.0, min(x, 1.0)) * _AXIS_BAR)
    return "#" * k + "." * (_AXIS_BAR - k)


def _section(rank: int, exp: dict) -> list[str]:
    lines = [
        f"## #{rank} - {exp['candidate_id']} | {exp['title']} ({exp['yoe']} yrs, {exp['location']})",
        "",
        f"**Score {exp['score']:.4f}**   "
        f"`J {exp['J']:.2f} {_bar(exp['J'])}`  "
        f"`C {exp['C']:.2f} {_bar(exp['C'])}`  "
        f"`A {exp['A']:.2f} {_bar(exp['A'])}`",
        "",
        "**Top contributions:**",
    ]
    for c in exp["contributions"][:5]:
        if c["contribution"] <= 0:
            break
        lines.append(f"- {c['rule']}: {c['score']:.2f} x w{c['weight']:.2f} = {c['contribution']:.3f}")
    if exp["evidence"]:
        lines += ["", "**Evidence from their own career narrative:**"]
        for e in exp["evidence"][:4]:
            lines.append(f"> {e}")
    concerns = exp["dampeners"] + exp["flags"]
    if concerns:
        lines += ["", "**Concerns (stated honestly):**"]
        for cnc in concerns[:3]:
            lines.append(f"- {cnc}")
    b = exp["behavior"]
    lines += ["", f"**Reachability:** last active {b['last_active']}, "
                  f"response rate {b['response_rate']}, notice {b['notice_days']}d, "
                  f"{b['work_mode']}, expects {b['expected_lpa'][0]:.0f}-{b['expected_lpa'][1]:.0f} LPA", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", help="existing submission CSV; omit to re-rank live")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="output/dossier.md")
    args = ap.parse_args()

    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    idx = load_index()
    probes = load_probes(ART / "probes.npz", rubric)

    if args.submission:
        with open(args.submission, encoding="utf-8") as f:
            ordered = [r["candidate_id"] for r in csv.DictReader(f)][: args.top]
    else:
        rubric["fusion"]["final_size"] = args.top
        ordered = [s.candidate_id for s in rank_pipeline(idx, rubric, probes)]

    lines = [
        f"# Shortlist dossier - {rubric['meta']['role']}",
        "",
        f"Top {len(ordered)} of {len(idx.ids)} candidates | rubric v{rubric['meta']['version']} | "
        "every claim below is traceable to the candidate's own profile",
        "",
    ]
    for rank, cid in enumerate(ordered, start=1):
        lines += _section(rank, build_explanation(idx, cid, rubric, probes))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(ordered)} candidates)")


if __name__ == "__main__":
    main()
