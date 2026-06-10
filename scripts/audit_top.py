"""Audit the top-N ranked candidates against raw profiles and score details."""

from __future__ import annotations

import argparse
import csv
import sys
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
from verdict.judgment import judge, predicate_scores
from verdict.recall import passes_gates


def _load_ranked(path: Path, n: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [r["candidate_id"] for r in list(csv.DictReader(f))[:n]]


def _verdict(rec, j: float, c: float, a: float, dnotes: list[str], flags: list[str]) -> tuple[str, str]:
    direct = rec.current_family in {"ml_engineer", "search_engineer", "nlp_engineer", "data_scientist", "applied_scientist"}
    core = rec.corroborated_categories & {"ranking", "search", "vector_db", "embeddings", "ml_core", "mlops", "nlp"}
    severe = bool(dnotes) or any(f.startswith(("TIMELINE_IMPOSSIBLE", "STUFFING_PATTERN")) for f in flags)
    if j >= 0.75 and c >= 0.8 and direct and len(core) >= 3:
        return "fair/high", "strong direct fit with credible evidence"
    if j >= 0.72 and direct and c >= 0.6:
        return "fair", "direct fit, but credibility or availability has caveats"
    if j >= 0.65 and c >= 0.55:
        return "slightly high", "good evidence, but weaker credibility/availability than a top-10 ideal"
    if severe:
        return "too high", "serious dampener/credibility concern for this rank"
    return "unclear", "needs manual review"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", default="output/submission_gpu20k.csv")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="output/top10_audit.md")
    args = ap.parse_args()

    ids = _load_ranked(Path(args.submission), args.top)
    wanted = set(ids)
    raw = {}
    records = {}
    for c in iter_candidates(args.candidates):
        cid = c["candidate_id"]
        if cid in wanted:
            raw[cid] = c
            records[cid] = build_record(c)
    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    sent_vecs = np.load(ART / "evidence_vectors.npy")
    counts = np.load(ART / "sent_counts.npy")
    pre_ids = (ART / "candidate_ids.txt").read_text(encoding="utf-8").splitlines()
    id_to_pre = {cid: i for i, cid in enumerate(pre_ids)}
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    probes = np.load(ART / "probes.npz")
    neg = probes["neg"].astype(np.float32)
    pred_vecs = {
        pid: probes[f"pred_{pid}"].astype(np.float32)
        for pid in rubric["fuzzy_predicates"]
    }

    rows = []
    for rank, cid in enumerate(ids, 1):
        rec = records[cid]
        pi = id_to_pre[cid]
        sv = sent_vecs[offsets[pi]:offsets[pi + 1]].astype(np.float32)
        preds = predicate_scores(sv, pred_vecs, neg, rubric["predicate_scoring"])
        j, rules, notes, dnotes = judge(rec, preds, rubric)
        c, cflags = score_credibility(rec)
        a, aflags = score_availability(rec, rubric)
        label, reason = _verdict(rec, j, c, a, dnotes, cflags + aflags)
        rows.append((rank, cid, rec, j, c, a, rules, dnotes, cflags + aflags, label, reason))

    lines = [
        "# Top 10 Audit",
        "",
        f"- Submission: `{args.submission}`",
        f"- Candidate file: `{args.candidates}`",
        "",
        "| rank | candidate | title | family | J | C | A | audit | reason |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for rank, cid, rec, j, c, a, _rules, _dnotes, _flags, label, reason in rows:
        lines.append(
            f"| {rank} | {cid} | {rec.current_title} | {rec.current_family} | "
            f"{j:.3f} | {c:.3f} | {a:.3f} | {label} | {reason} |"
        )

    lines += ["", "## Candidate Notes", ""]
    for rank, cid, rec, j, c, a, rules, dnotes, flags, label, reason in rows:
        strengths = sorted(
            [(k, v) for k, v in rules.items() if v >= 0.55],
            key=lambda x: -x[1],
        )[:6]
        concerns = dnotes + flags
        lines += [
            f"### {rank}. {cid} - {rec.current_title}",
            "",
            f"- Audit: **{label}** - {reason}.",
            f"- Fit: J `{j:.3f}`, credibility `{c:.3f}`, availability `{a:.3f}`.",
            f"- Location/logistics: `{rec.location_bucket}`, notice `{rec.notice_days}` days, relocate `{rec.willing_to_relocate}`.",
            f"- Evidence categories: `{', '.join(sorted(rec.corroborated_categories)) or 'none'}`.",
            f"- Strong rules: `{', '.join(f'{k}:{v:.2f}' for k, v in strengths)}`.",
            f"- Concerns: `{'; '.join(concerns[:3]) or 'none'}`.",
            "- Best evidence:",
        ]
        for sent in rec.sentences[:4]:
            lines.append(f"  - {sent}")
        lines.append("")

    out = ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
