"""Compare two rubric versions' rankings: who moved, in, out - and why.

The recruiter-feedback loop made tangible: edit a weight or dampener in a
rubric copy, run this, and see exactly which candidates the change promotes
or buries before committing it.

  python rubric_diff.py --rubric-b artifacts/generated/rubric_v2.json
  python rubric_diff.py --rubric-b rubric_v2.json --probes-b probes_v2.npz --top 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict.pipeline import ART, load_index, load_probes, rank_pipeline

NEG_INF_RANK = 10**6


def _rank_map(top) -> dict[str, int]:
    return {s.candidate_id: r for r, s in enumerate(top, start=1)}


def _why(s_a, s_b, rubric_a: dict, rubric_b: dict) -> str:
    """Largest weighted rule-contribution change between the two runs."""
    wa = {**{k: v["weight"] for k, v in rubric_a["crisp_rules"].items()},
          **{k: v["weight"] for k, v in rubric_a["fuzzy_predicates"].items()}}
    wb = {**{k: v["weight"] for k, v in rubric_b["crisp_rules"].items()},
          **{k: v["weight"] for k, v in rubric_b["fuzzy_predicates"].items()}}
    deltas = []
    for rule in set(s_a.rule_scores) | set(s_b.rule_scores):
        ca = s_a.rule_scores.get(rule, 0.0) * wa.get(rule, 0.0)
        cb = s_b.rule_scores.get(rule, 0.0) * wb.get(rule, 0.0)
        if abs(cb - ca) > 1e-4:
            deltas.append((abs(cb - ca), rule, ca, cb))
    if s_a.j != s_b.j and not deltas:
        return "dampener change"
    if not deltas:
        return "tournament/availability reorder"
    deltas.sort(reverse=True)
    _, rule, ca, cb = deltas[0]
    return f"{rule}: {ca:.3f} -> {cb:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric-a", default=str(ART / "rubric_program.json"))
    ap.add_argument("--probes-a", default=str(ART / "probes.npz"))
    ap.add_argument("--rubric-b", required=True)
    ap.add_argument("--probes-b", help="probes for rubric B; required only if its predicates/negatives changed")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", help="optional markdown report path")
    args = ap.parse_args()

    rubric_a = orjson.loads(Path(args.rubric_a).read_bytes())
    rubric_b = orjson.loads(Path(args.rubric_b).read_bytes())
    rubric_a["fusion"]["final_size"] = rubric_b["fusion"]["final_size"] = args.top

    same_probe_inputs = (
        rubric_a["fuzzy_predicates"].keys() == rubric_b["fuzzy_predicates"].keys()
        and all(rubric_a["fuzzy_predicates"][p]["positives"] == rubric_b["fuzzy_predicates"][p]["positives"]
                for p in rubric_a["fuzzy_predicates"])
        and rubric_a["predicate_negatives"] == rubric_b["predicate_negatives"]
    )
    if not args.probes_b and not same_probe_inputs:
        raise SystemExit("rubric B changes predicates/negatives - build its probes first "
                         "(scripts/build_probes.py --rubric <B> --out probes_b.npz) and pass --probes-b")

    idx = load_index()
    probes_a = load_probes(Path(args.probes_a), rubric_a)
    probes_b = load_probes(Path(args.probes_b), rubric_b) if args.probes_b else probes_a

    top_a = rank_pipeline(idx, rubric_a, probes_a)
    top_b = rank_pipeline(idx, rubric_b, probes_b)
    ra, rb = _rank_map(top_a), _rank_map(top_b)
    by_id_a = {s.candidate_id: s for s in top_a}
    by_id_b = {s.candidate_id: s for s in top_b}

    entered = [cid for cid in rb if cid not in ra]
    dropped = [cid for cid in ra if cid not in rb]
    common = [cid for cid in ra if cid in rb]
    movers = sorted(((ra[c] - rb[c], c) for c in common), key=lambda x: -abs(x[0]))
    overlap = len(common) / max(len(ra), 1)

    lines = [
        f"# Rubric diff: {Path(args.rubric_a).name} -> {Path(args.rubric_b).name}",
        "",
        f"- top-{args.top} overlap: **{len(common)}/{len(ra)}** ({overlap:.0%})",
        f"- new entrants: **{len(entered)}**, dropped: **{len(dropped)}**",
        "",
        "## New entrants (rubric B promotes)",
        "| rank B | candidate | title | why (biggest contribution change) |",
        "|---|---|---|---|",
    ]
    for cid in sorted(entered, key=lambda c: rb[c])[:15]:
        s_b = by_id_b[cid]
        rec = idx.records[s_b.idx]
        why = _why(s_b, s_b, rubric_a, rubric_b) if cid not in by_id_a else _why(by_id_a[cid], s_b, rubric_a, rubric_b)
        lines.append(f"| {rb[cid]} | {cid} | {rec.current_title} | {why} |")
    lines += ["", "## Dropped (rubric B buries)", "| rank A | candidate | title | why |", "|---|---|---|---|"]
    for cid in sorted(dropped, key=lambda c: ra[c])[:15]:
        s_a = by_id_a[cid]
        rec = idx.records[s_a.idx]
        lines.append(f"| {ra[cid]} | {cid} | {rec.current_title} | "
                     f"{_why(s_a, s_a, rubric_a, rubric_b)} |")
    lines += ["", "## Biggest movers among common candidates",
              "| candidate | rank A -> B | why |", "|---|---|---|"]
    for delta, cid in movers[:15]:
        if delta == 0:
            break
        lines.append(f"| {cid} | {ra[cid]} -> {rb[cid]} ({'+' if delta > 0 else ''}{delta}) | "
                     f"{_why(by_id_a[cid], by_id_b[cid], rubric_a, rubric_b)} |")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
