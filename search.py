"""Interactive candidate search over precomputed VERDICT artifacts.

This keeps the challenge `rank.py` intact and exposes the same data as a
recruiter-style search surface:

  python search.py --query "AI developer ML" --min-yoe 3 --availability \
    --good-companies --top 25 --out output/search_ai_ml.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import time
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).parent / "src"))

from verdict.availability import score_availability
from verdict.credibility import score_credibility
from verdict.embedder import embed_queries, get_model
from verdict.evidence import LedgerRecord, record_from_dict
from verdict.recall import bm25_rank, rrf_fuse

ART = Path(__file__).parent / "artifacts"

ROLE_PRESETS = {
    "ai_ml": {
        "families": {"ml_engineer", "applied_scientist", "nlp_engineer", "search_engineer", "data_scientist", "mlops_engineer"},
        "categories": {"ml_core", "mlops", "llm", "nlp", "search", "ranking", "embeddings", "vector_db"},
        "query": "AI developer machine learning production ML NLP LLM embeddings vector search ranking recommendation Python model deployment",
    },
    "backend": {
        "families": {"backend", "swe", "fullstack", "devops"},
        "categories": {"backend", "data_eng", "cloud", "devops"},
        "query": "backend engineer APIs microservices distributed systems databases Kafka Redis Python Java Go cloud production",
    },
    "data": {
        "families": {"data_engineer", "data_scientist", "analyst"},
        "categories": {"data_eng", "analytics", "ml_core"},
        "query": "data pipelines analytics Spark SQL Airflow warehouse machine learning data scientist",
    },
}


def _load_records() -> list[LedgerRecord]:
    path = ART / "records.jsonl.gz"
    if not path.exists():
        raise SystemExit("Missing artifacts/records.jsonl.gz. Run precompute.py first.")
    records = []
    with gzip.open(path, "rb") as f:
        for line in f:
            if line.strip():
                records.append(record_from_dict(orjson.loads(line)))
    return records


def _norm01(values: np.ndarray) -> np.ndarray:
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 1.0
    return (values - lo) / max(hi - lo, 1e-9)


def _parse_csv_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _product_score(rec: LedgerRecord) -> float:
    if rec.product_share >= 0.8:
        return 1.0
    if rec.product_share >= 0.5:
        return 0.75
    if rec.product_share > 0:
        return 0.45
    return 0.1


def _family_score(rec: LedgerRecord, families: set[str]) -> float:
    if not families:
        return 0.5
    if rec.current_family in families:
        return 1.0
    if set(rec.families) & families:
        return 0.75
    return 0.0


def _category_score(rec: LedgerRecord, categories: set[str]) -> float:
    if not categories:
        return 0.5
    # Corroborated categories count most; claimed categories get partial credit.
    corro = len(rec.corroborated_categories & categories)
    claimed = len((rec.skill_categories - rec.corroborated_categories) & categories)
    return min((corro + 0.35 * claimed) / max(len(categories), 1), 1.0)


def _availability_score(rec: LedgerRecord, rubric: dict, require_available: bool) -> tuple[float, list[str]]:
    a, flags = score_availability(rec, rubric)
    if require_available:
        if rec.notice_days > 90:
            a *= 0.55
        elif rec.notice_days > 60:
            a *= 0.75
        if float(rec.signals.get("recruiter_response_rate") or 0) < 0.25:
            a *= 0.75
    return a, flags


def _reason(rec: LedgerRecord, score_parts: dict, concerns: list[str]) -> str:
    strengths = []
    if score_parts["family"] >= 0.75:
        strengths.append(rec.current_title or rec.current_family.replace("_", " "))
    cats = sorted(rec.corroborated_categories & {"ml_core", "mlops", "llm", "nlp", "search", "ranking", "embeddings", "vector_db"})
    if cats:
        strengths.append("corroborated " + "/".join(cats[:4]))
    if rec.product_share >= 0.5:
        strengths.append(f"{rec.product_share:.0%} product-company tenure")
    rr = float(rec.signals.get("recruiter_response_rate") or 0)
    if rr >= 0.5:
        strengths.append(f"{rr:.0%} response rate")
    if rec.notice_days <= 30:
        strengths.append(f"{rec.notice_days}-day notice")
    concern = concerns[0] if concerns else ""
    txt = "; ".join(strengths[:4]) or "partial profile match"
    if concern:
        txt += f". Concern: {concern}"
    return txt[:280]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="plain-language search, e.g. 'AI developer ML'")
    ap.add_argument("--preset", choices=sorted(ROLE_PRESETS), default="ai_ml")
    ap.add_argument("--min-yoe", type=float, default=0.0)
    ap.add_argument("--max-yoe", type=float, default=50.0)
    ap.add_argument("--families", help="comma-separated normalized title families; overrides preset")
    ap.add_argument("--categories", help="comma-separated normalized skill categories; overrides preset")
    ap.add_argument("--availability", action="store_true", help="prefer reachable candidates: response, notice, logistics")
    ap.add_argument("--max-notice-days", type=int, help="hard filter for availability, e.g. 60")
    ap.add_argument("--min-response-rate", type=float, help="hard filter for recruiter response rate, e.g. 0.40")
    ap.add_argument("--good-companies", action="store_true", help="boost product-company/company-quality career history")
    ap.add_argument("--min-product-share", type=float, help="hard filter for product-company tenure share, e.g. 0.50")
    ap.add_argument("--location", choices=["any", "preferred", "tier1", "india"], default="any")
    ap.add_argument("--relocation-ok", action="store_true")
    ap.add_argument("--cuda", action="store_true", help="use CUDA for embedding the search query")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="output/search_results.csv")
    args = ap.parse_args()

    t0 = time.time()
    rubric = orjson.loads((ART / "rubric_program.json").read_bytes())
    records = _load_records()
    mean_vecs = np.load(ART / "mean_vecs.npy").astype(np.float32)

    preset = ROLE_PRESETS[args.preset]
    families = _parse_csv_set(args.families) or set(preset["families"])
    categories = _parse_csv_set(args.categories) or set(preset["categories"])
    query = f"{preset['query']} {args.query}"

    candidates: list[tuple[int, LedgerRecord]] = []
    for i, rec in enumerate(records):
        yoe = max(rec.yoe_stated, rec.yoe_timeline)
        response_rate = float(rec.signals.get("recruiter_response_rate") or 0)
        if yoe < args.min_yoe or yoe > args.max_yoe:
            continue
        if args.max_notice_days is not None and rec.notice_days > args.max_notice_days:
            continue
        if args.min_response_rate is not None and response_rate < args.min_response_rate:
            continue
        if args.min_product_share is not None and rec.product_share < args.min_product_share:
            continue
        if args.location == "preferred" and rec.location_bucket != "preferred":
            continue
        if args.location == "tier1" and rec.location_bucket not in {"preferred", "tier1"}:
            continue
        if args.location == "india" and rec.location_bucket == "abroad" and not (args.relocation_ok and rec.willing_to_relocate):
            continue
        candidates.append((i, rec))

    idx = np.asarray([i for i, _r in candidates], dtype=np.int64)
    subset = [r for _i, r in candidates]
    bm = bm25_rank([r.narrative_text for r in subset], query)
    if len(idx):
        model = get_model(cuda=args.cuda)
        query_vec = embed_queries(model, [query])[0]
        dense = mean_vecs[idx] @ query_vec
    else:
        dense = np.array([], dtype=np.float32)
    fam = np.asarray([_family_score(r, families) for r in subset], dtype=np.float32)
    cat = np.asarray([_category_score(r, categories) for r in subset], dtype=np.float32)
    prod = np.asarray([_product_score(r) for r in subset], dtype=np.float32)

    avail = []
    credibility = []
    all_concerns = []
    for rec in subset:
        c, cflags = score_credibility(rec)
        a, aflags = _availability_score(rec, rubric, args.availability)
        credibility.append(c)
        avail.append(a)
        all_concerns.append(cflags + aflags)
    credibility_arr = np.asarray(credibility, dtype=np.float32)
    avail_arr = np.asarray(avail, dtype=np.float32)

    # RRF keeps recall robust; weighted score controls final ordering.
    orders = [
        np.argsort(-bm, kind="stable")[: min(len(subset), 4000)],
        np.argsort(-dense, kind="stable")[: min(len(subset), 4000)],
        np.argsort(-fam, kind="stable")[: min(len(subset), 4000)],
        np.argsort(-cat, kind="stable")[: min(len(subset), 4000)],
    ]
    rrf = rrf_fuse(orders, len(subset), k=60) if subset else np.array([], dtype=np.float32)

    score = (
        0.22 * _norm01(bm)
        + 0.14 * _norm01(dense)
        + 0.22 * fam
        + 0.18 * cat
        + (0.10 if args.good_companies else 0.04) * prod
        + (0.16 if args.availability else 0.08) * avail_arr
        + 0.08 * credibility_arr
        + 0.10 * _norm01(rrf)
    )
    score *= np.maximum(credibility_arr, 0.05) ** 0.4

    order = np.argsort(-score, kind="stable")[: min(args.top, len(subset))]
    rows = []
    for rank, j in enumerate(order, 1):
        rec = subset[int(j)]
        parts = {
            "bm25": float(bm[j]),
            "dense": float(dense[j]),
            "family": float(fam[j]),
            "category": float(cat[j]),
            "product": float(prod[j]),
            "availability": float(avail_arr[j]),
            "credibility": float(credibility_arr[j]),
        }
        rows.append(
            {
                "rank": rank,
                "candidate_id": rec.candidate_id,
                "score": f"{float(score[j]):.6f}",
                "title": rec.current_title,
                "family": rec.current_family,
                "yoe": f"{max(rec.yoe_stated, rec.yoe_timeline):.1f}",
                "location": rec.location_bucket,
                "notice_days": rec.notice_days,
                "response_rate": f"{float(rec.signals.get('recruiter_response_rate') or 0):.2f}",
                "product_share": f"{rec.product_share:.2f}",
                "credibility": f"{parts['credibility']:.3f}",
                "availability": f"{parts['availability']:.3f}",
                "reasoning": _reason(rec, parts, all_concerns[int(j)]),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank", "candidate_id", "score", "title", "family", "yoe",
            "location", "notice_days", "response_rate", "product_share",
            "credibility", "availability", "reasoning",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"searched {len(records)} artifact records, filtered {len(subset)}, wrote {len(rows)} rows to {out} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
