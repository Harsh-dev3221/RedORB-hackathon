"""Quality gates for embedding/ranking optimization experiments.

Run this after producing a candidate ranking CSV. It is deliberately heuristic:
the hidden labels are unavailable, so these checks guard against known failure
modes from the challenge docs.

Example:
  python scripts/audit_quality_gates.py --submission output/trap_audit_top100.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verdict.data import iter_candidates
from verdict.evidence import record_from_dict, build_record

ART = ROOT / "artifacts"
REF_DATE = date(2026, 6, 1)

AI_CATS = {"llm", "llm_framework", "embeddings", "vector_db", "nlp", "ml_core", "search", "ranking"}
NONTECH = {"marketing", "sales", "hr", "design", "finance", "ops", "pm", "product"}
SHINY = re.compile(r"\b(rag|pinecone|qdrant|weaviate|milvus|llm|large language|langchain|llamaindex|openai|vector database)\b", re.I)
MEANING = re.compile(
    r"\b(recommendation|recommender|ranking|search relevance|learning-to-rank|"
    r"learning to rank|personalization|retrieval|collaborative filtering)\b",
    re.I,
)


def _honeypot_category(msg: str) -> str:
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


def _load_submission(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("submission has no rows")
    return [r["candidate_id"] for r in rows]


def _load_records(candidates_path: str | None = None) -> dict:
    if candidates_path:
        out = {}
        for cand in iter_candidates(candidates_path):
            rec = build_record(cand)
            out[rec.candidate_id] = rec
        return out
    path = ART / "records.jsonl.gz"
    if not path.exists():
        raise SystemExit("missing artifacts/records.jsonl.gz; pass --candidates to rebuild records for audit")
    out = {}
    with gzip.open(path, "rb") as f:
        for line in f:
            if line.strip():
                rec = record_from_dict(orjson.loads(line))
                out[rec.candidate_id] = rec
    return out


def _days_since(raw: str | None) -> int:
    if not raw:
        return 999
    try:
        y, m, d = (int(x) for x in raw.split("-"))
        return max((REF_DATE - date(y, m, d)).days, 0)
    except ValueError:
        return 999


def _severe_availability_risk(rec) -> tuple[bool, str]:
    rr = float(rec.signals.get("recruiter_response_rate") or 0.0)
    idle = _days_since(rec.signals.get("last_active_date"))
    open_to_work = bool(rec.signals.get("open_to_work_flag"))
    if idle >= 180 and rr <= 0.10:
        return True, f"last_active={idle}d response={rr:.0%}"
    if idle >= 240:
        return True, f"last_active={idle}d"
    if rr <= 0.05 and not open_to_work:
        return True, f"response={rr:.0%} not_open_to_work"
    return False, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--candidates", help="optional raw candidate file if records artifact is unavailable/stale")
    ap.add_argument("--min-hidden-gems", type=int, default=3)
    ap.add_argument("--max-honeypot-rate", type=float, default=0.10)
    ap.add_argument("--max-severe-unavailable", type=int, default=0)
    args = ap.parse_args()

    ids = _load_submission(Path(args.submission))
    records = _load_records(args.candidates)
    missing = [cid for cid in ids if cid not in records]
    if missing:
        raise SystemExit(f"{len(missing)} submission ids not found in records, first={missing[0]}")

    keyword_stuffers = []
    honeypots = []
    severe_unavailable = []
    hidden_gems = []
    honeypot_categories = Counter()
    long_notice = []

    for rank, cid in enumerate(ids, 1):
        rec = records[cid]
        skills = set(rec.skill_categories)
        corro = set(rec.corroborated_categories)
        ai_claimed = skills & AI_CATS
        ai_corro = corro & AI_CATS
        if rec.current_family in NONTECH and len(ai_claimed) >= 4 and len(ai_corro) <= 2:
            keyword_stuffers.append((rank, cid, rec.current_title, sorted(ai_claimed), sorted(ai_corro)))
        if rec.impossibilities:
            honeypots.append((rank, cid, rec.current_title, rec.impossibilities[:2]))
            honeypot_categories.update(_honeypot_category(x) for x in rec.impossibilities)
        risky, why = _severe_availability_risk(rec)
        if risky:
            severe_unavailable.append((rank, cid, rec.current_title, why))
        if rec.notice_days > 90:
            long_notice.append((rank, cid, rec.current_title, rec.notice_days))
        text = rec.narrative_text or ""
        if MEANING.search(text) and not SHINY.search(text) and rec.current_family not in NONTECH:
            hidden_gems.append((rank, cid, rec.current_title))

    failures = []
    if keyword_stuffers:
        failures.append(f"keyword-stuffer leaks: {len(keyword_stuffers)}")
    honeypot_rate = len(honeypots) / max(len(ids), 1)
    if honeypot_rate > args.max_honeypot_rate:
        failures.append(f"honeypot/impossible-profile rate {honeypot_rate:.1%} > {args.max_honeypot_rate:.1%}")
    if len(severe_unavailable) > args.max_severe_unavailable:
        failures.append(f"severe unavailable profiles: {len(severe_unavailable)} > {args.max_severe_unavailable}")
    if len(hidden_gems) < args.min_hidden_gems:
        failures.append(f"plain-language hidden gems: {len(hidden_gems)} < {args.min_hidden_gems}")

    print(f"rows={len(ids)}")
    print(f"keyword_stuffer_leaks={len(keyword_stuffers)}")
    print(f"honeypot_impossible_profiles={len(honeypots)} rate={honeypot_rate:.1%}")
    print(f"honeypot_categories={dict(honeypot_categories)}")
    print(f"severe_unavailable={len(severe_unavailable)}")
    print(f"long_notice_over_90d={len(long_notice)}")
    print(f"plain_language_hidden_gems={len(hidden_gems)}")
    if keyword_stuffers[:5]:
        print("keyword_stuffer_examples=", keyword_stuffers[:5])
    if hidden_gems[:5]:
        print("hidden_gem_examples=", hidden_gems[:5])
    if honeypots[:5]:
        print("honeypot_examples=", honeypots[:5])
    if severe_unavailable[:5]:
        print("severe_unavailable_examples=", severe_unavailable[:5])
    if long_notice[:5]:
        print("long_notice_examples=", long_notice[:5])

    if failures:
        raise SystemExit("FAILED quality gates: " + "; ".join(failures))
    print("PASSED quality gates")


if __name__ == "__main__":
    main()
