"""Inventory challenge trap patterns in the candidate pool and a submission.

This is not a hidden-label detector. The bundle does not expose honeypot labels,
so this script audits observable failure modes described by the docs:
impossible timelines, keyword-stuffed non-technical profiles, plain-language
hidden gems, and behavioral unavailability traps.

Examples:
  python scripts/audit_honeypots.py --candidates data/candidates.jsonl
  python scripts/audit_honeypots.py --candidates data/candidates.jsonl --submission output/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.evidence import LedgerRecord, build_record

REF_DATE = date(2026, 6, 1)
AI_CATS = {"llm", "llm_framework", "embeddings", "vector_db", "nlp", "ml_core", "search", "ranking"}
NONTECH = {"marketing", "sales", "hr", "design", "finance", "ops", "pm", "product"}
SHINY = re.compile(
    r"\b(rag|pinecone|qdrant|weaviate|milvus|llm|large language|langchain|llamaindex|openai|vector database)\b",
    re.I,
)
MEANING = re.compile(
    r"\b(recommendation|recommender|ranking|search relevance|learning-to-rank|"
    r"learning to rank|personalization|retrieval|collaborative filtering)\b",
    re.I,
)


def _days_since(raw: str | None) -> int:
    if not raw:
        return 999
    try:
        y, m, d = (int(x) for x in raw.split("-"))
        return max((REF_DATE - date(y, m, d)).days, 0)
    except ValueError:
        return 999


def _classify_impossibility(msg: str) -> str:
    text = msg.lower()
    if "before it existed" in text:
        return "company_before_founding"
    if "ends before it starts" in text:
        return "role_ends_before_start"
    if "overlap" in text:
        return "overlapping_full_time_roles"
    if "states" in text and "experience" in text and "career history totals" in text:
        return "stated_yoe_impossible"
    if "claims" in text and "total career" in text:
        return "skill_duration_impossible"
    if "advanced/expert" in text and "0 months" in text:
        return "zero_month_expert_cluster"
    return "other_impossible_profile"


def _behavioral_traps(rec: LedgerRecord) -> list[str]:
    sig = rec.signals
    rr = float(sig.get("recruiter_response_rate") or 0.0)
    idle = _days_since(sig.get("last_active_date"))
    out: list[str] = []
    if idle >= 180 and rr <= 0.10:
        out.append("stale_6mo_and_low_response")
    elif idle >= 240:
        out.append("stale_8mo")
    if rr <= 0.05 and not sig.get("open_to_work_flag"):
        out.append("passive_5pct_response")
    if rec.notice_days > 90:
        out.append("notice_over_90d")
    return out


def _keyword_stuffer(rec: LedgerRecord) -> bool:
    skills = set(rec.skill_categories)
    corro = set(rec.corroborated_categories)
    return rec.current_family in NONTECH and len(skills & AI_CATS) >= 4 and len(corro & AI_CATS) <= 2


def _hidden_gem(rec: LedgerRecord) -> bool:
    text = rec.narrative_text or ""
    return bool(MEANING.search(text)) and not SHINY.search(text) and rec.current_family not in NONTECH


def _load_submission(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return [r["candidate_id"] for r in csv.DictReader(f)]


def _examples(rows: Iterable[tuple], limit: int) -> list[tuple]:
    return list(rows)[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission")
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--max-top-impossible", type=int, default=0)
    ap.add_argument("--max-top-severe-behavioral", type=int, default=0)
    ap.add_argument("--min-top-hidden-gems", type=int, default=3)
    args = ap.parse_args()

    records: dict[str, LedgerRecord] = {}
    impossible = Counter()
    behavioral = Counter()
    keyword_stuffers: list[tuple[str, str]] = []
    hidden_gems: list[tuple[str, str]] = []
    examples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for cand in iter_candidates(args.candidates):
        rec = build_record(cand)
        records[rec.candidate_id] = rec
        for msg in rec.impossibilities:
            cat = _classify_impossibility(msg)
            impossible[cat] += 1
            if len(examples[cat]) < args.examples:
                examples[cat].append((rec.candidate_id, rec.current_title, msg))
        for cat in _behavioral_traps(rec):
            behavioral[cat] += 1
            if len(examples[cat]) < args.examples:
                examples[cat].append((rec.candidate_id, rec.current_title, cat))
        if _keyword_stuffer(rec):
            keyword_stuffers.append((rec.candidate_id, rec.current_title))
        if _hidden_gem(rec):
            hidden_gems.append((rec.candidate_id, rec.current_title))

    print(f"pool_records={len(records)}")
    print(f"impossible_profile_categories={dict(impossible)}")
    print(f"behavioral_trap_categories={dict(behavioral)}")
    print(f"keyword_stuffer_candidates={len(keyword_stuffers)}")
    print(f"plain_language_hidden_gem_candidates={len(hidden_gems)}")
    for cat, rows in sorted(examples.items()):
        print(f"{cat}_examples={rows}")

    if not args.submission:
        return

    ids = _load_submission(Path(args.submission))
    top_impossible: list[tuple[int, str, str, list[str]]] = []
    top_behavioral: list[tuple[int, str, str, list[str]]] = []
    top_keyword_stuffers: list[tuple[int, str, str]] = []
    top_hidden_gems: list[tuple[int, str, str]] = []
    missing = [cid for cid in ids if cid not in records]
    if missing:
        raise SystemExit(f"{len(missing)} submission ids missing from candidate pool, first={missing[0]}")

    for rank, cid in enumerate(ids, 1):
        rec = records[cid]
        if rec.impossibilities:
            top_impossible.append((rank, cid, rec.current_title, rec.impossibilities[:2]))
        traps = [x for x in _behavioral_traps(rec) if x != "notice_over_90d"]
        if traps:
            top_behavioral.append((rank, cid, rec.current_title, traps))
        if _keyword_stuffer(rec):
            top_keyword_stuffers.append((rank, cid, rec.current_title))
        if _hidden_gem(rec):
            top_hidden_gems.append((rank, cid, rec.current_title))

    print(f"submission_rows={len(ids)}")
    print(f"top_impossible_profiles={len(top_impossible)}")
    print(f"top_severe_behavioral_traps={len(top_behavioral)}")
    print(f"top_keyword_stuffers={len(top_keyword_stuffers)}")
    print(f"top_hidden_gems={len(top_hidden_gems)}")
    print(f"top_impossible_examples={top_impossible[:args.examples]}")
    print(f"top_behavioral_examples={top_behavioral[:args.examples]}")
    print(f"top_hidden_gem_examples={top_hidden_gems[:args.examples]}")

    failures = []
    if len(top_impossible) > args.max_top_impossible:
        failures.append(f"top impossible profiles {len(top_impossible)} > {args.max_top_impossible}")
    if len(top_behavioral) > args.max_top_severe_behavioral:
        failures.append(
            f"top severe behavioral traps {len(top_behavioral)} > {args.max_top_severe_behavioral}"
        )
    if len(top_hidden_gems) < args.min_top_hidden_gems:
        failures.append(f"top hidden gems {len(top_hidden_gems)} < {args.min_top_hidden_gems}")
    if top_keyword_stuffers:
        failures.append(f"top keyword stuffers {len(top_keyword_stuffers)} > 0")
    if failures:
        raise SystemExit("FAILED honeypot audit: " + "; ".join(failures))
    print("PASSED honeypot audit")


if __name__ == "__main__":
    main()
