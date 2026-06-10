"""Generate a run report for a candidate file and an already-built artifact set.

This is diagnostic only; it mirrors rank.py but keeps the intermediate J/C/A
numbers so we can judge whether the output is worth trusting.

Usage:
  python scripts/run_report.py --candidates path --out output/sample_run_report.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).parent.parent
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

from verdict.availability import score_availability
from verdict.credibility import score_credibility
from verdict.data import iter_candidates
from verdict.evidence import build_record
from verdict.fusion import Scored, finalize
from verdict.judgment import judge, predicate_scores
from verdict.recall import passes_gates, run_recall


def _gate_reason(rec, gates: dict) -> str:
    if rec.n_jobs < gates["min_career_entries"]:
        return "no career history"
    yoe = max(rec.yoe_stated, rec.yoe_timeline)
    if yoe < gates["yoe_min"]:
        return "below YOE gate"
    if yoe > gates["yoe_max"]:
        return "above YOE gate"
    if (
        gates["reject_abroad_without_relocation"]
        and rec.location_bucket == "abroad"
        and not rec.willing_to_relocate
    ):
        return "abroad/no relocation"
    return "passed"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="output/run_report.md")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    sent_vecs = np.load(ART / "evidence_vectors.npy")
    counts = np.load(ART / "sent_counts.npy")
    mean_vecs_all = np.load(ART / "mean_vecs.npy")
    pre_ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    id_to_pre = {cid: i for i, cid in enumerate(pre_ids)}
    probes = np.load(ART / "probes.npz")
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    all_records = []
    survivors = []
    surv_pool_idx = []
    gate_reasons = Counter()
    for c in iter_candidates(args.candidates):
        rec = build_record(c)
        all_records.append(rec)
        reason = _gate_reason(rec, rubric["gates"])
        gate_reasons[reason] += 1
        if reason == "passed":
            pi = id_to_pre.get(rec.candidate_id)
            if pi is not None:
                survivors.append(rec)
                surv_pool_idx.append(pi)

    surv_mean = mean_vecs_all[np.asarray(surv_pool_idx)].astype(np.float32)
    recall_idx = run_recall(survivors, surv_mean, probes["ideal"].astype(np.float32), rubric)

    neg = probes["neg"].astype(np.float32)
    pred_vecs = {
        pid: probes[f"pred_{pid}"].astype(np.float32)
        for pid in rubric["fuzzy_predicates"]
    }
    scored: list[Scored] = []
    details: dict[str, tuple[float, float, float, list[str], list[str]]] = {}
    for si in recall_idx:
        rec = survivors[int(si)]
        pi = surv_pool_idx[int(si)]
        sv = sent_vecs[offsets[pi] : offsets[pi + 1]].astype(np.float32)
        preds = predicate_scores(sv, pred_vecs, neg, rubric["predicate_scoring"])
        j, rules, notes, dnotes = judge(rec, preds, rubric)
        c_score, c_flags = score_credibility(rec)
        a_score, a_flags = score_availability(rec, rubric)
        item = Scored(
            idx=int(si),
            candidate_id=rec.candidate_id,
            j=j,
            c=c_score,
            a=a_score,
            rule_scores=rules,
            evidence_notes=notes,
            dampener_notes=dnotes,
            flags=c_flags + a_flags,
        )
        scored.append(item)
        details[rec.candidate_id] = (j, c_score, a_score, dnotes, c_flags + a_flags)

    original_final_size = rubric["fusion"]["final_size"]
    rubric["fusion"]["final_size"] = min(original_final_size, len(scored))
    top = finalize(scored, rubric)
    top_n = top[: min(args.top, len(top))]

    scores = [x.final for x in top]
    fam_counts = Counter()
    top_family_counts = Counter()
    rec_by_id = {r.candidate_id: r for r in survivors}
    raw_family_counts = Counter(r.current_family for r in all_records)
    for item in top:
        top_family_counts[rec_by_id[item.candidate_id].current_family] += 1
    for r in survivors:
        fam_counts[r.current_family] += 1

    concern_counts = Counter()
    for item in top:
        _j, _c, _a, dnotes, flags = details[item.candidate_id]
        if dnotes:
            concern_counts[dnotes[0]] += 1
        elif flags:
            concern_counts[flags[0].split(": ", 1)[0]] += 1
        else:
            concern_counts["none"] += 1

    out_lines = [
        "# VERDICT Sample Run Report",
        "",
        f"- Candidates file: `{args.candidates}`",
        f"- Runtime for report recompute: `{time.time() - t0:.2f}s`",
        f"- Raw candidates: `{len(all_records)}`",
        f"- Artifact candidates: `{len(pre_ids)}`",
        f"- Gate survivors available to rank: `{len(survivors)}`",
        f"- Recall/scored set: `{len(scored)}`",
        f"- Output rows in this diagnostic report: `{len(top)}`",
        f"- Evidence vectors: `{len(sent_vecs)}`",
        f"- Evidence sentences per artifact candidate: min `{int(counts.min())}`, mean `{float(counts.mean()):.2f}`, max `{int(counts.max())}`",
        "",
        "## Gate Breakdown",
        "",
    ]
    for reason, n in gate_reasons.most_common():
        out_lines.append(f"- {reason}: `{n}` ({_pct(n / max(len(all_records), 1))})")

    out_lines += [
        "",
        "## Score Distribution",
        "",
        f"- max: `{max(scores):.6f}`",
        f"- median: `{statistics.median(scores):.6f}`",
        f"- min: `{min(scores):.6f}`",
        "",
        "## Raw Current Title Families",
        "",
    ]
    for fam, n in raw_family_counts.most_common(10):
        out_lines.append(f"- {fam}: `{n}`")

    out_lines += ["", "## Ranked Family Mix", ""]
    for fam, n in top_family_counts.most_common(10):
        out_lines.append(f"- {fam}: `{n}`")

    out_lines += ["", "## Main Concerns In Ranked Output", ""]
    for concern, n in concern_counts.most_common(10):
        out_lines.append(f"- {concern}: `{n}`")

    out_lines += [
        "",
        "## Top Candidates",
        "",
        "| rank | candidate | title | family | final | J | C | A | concern |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(top_n, start=1):
        rec = rec_by_id[item.candidate_id]
        j, c_score, a_score, dnotes, flags = details[item.candidate_id]
        concern = dnotes[0] if dnotes else (flags[0] if flags else "")
        concern = concern.replace("|", "/")
        out_lines.append(
            f"| {rank} | {item.candidate_id} | {rec.current_title} | {rec.current_family} | "
            f"{item.final:.6f} | {j:.3f} | {c_score:.3f} | {a_score:.3f} | {concern} |"
        )

    best = top[0] if top else None
    if best:
        rec = rec_by_id[best.candidate_id]
        j, c_score, a_score, dnotes, flags = details[best.candidate_id]
        verdict = (
            "Worth trusting for this sample: the top candidate is a direct recommendation/search profile."
            if rec.current_family in {"search_engineer", "ml_engineer", "nlp_engineer", "data_scientist"}
            and j > 0.30
            and c_score > 0.5
            else "Needs tuning: the top candidate is not a clear direct AI/search fit."
        )
        out_lines += [
            "",
            "## Worth Check",
            "",
            f"- {verdict}",
            f"- Top candidate `{best.candidate_id}` is `{rec.current_title}` with J=`{j:.3f}`, C=`{c_score:.3f}`, A=`{a_score:.3f}`.",
        ]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
