"""L1 Evidence Builder: raw candidate JSON -> ledger record.

A profile is not a document; it is a bundle of claims with different costs of
fabrication. This module reconstructs the career timeline, normalizes entities,
extracts evidence sentences (narrative only - the skills list is NEVER part of
the embedded evidence), and links self-declared skills to corroborating
narrative text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from datetime import date

from .normalizer import (
    COMPANY_FOUNDED,
    classify_company,
    normalize_location,
    normalize_skill,
    normalize_title,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9+#&.\-]+")
_MAX_EVIDENCE_SENTENCES = 12

_CORE_EVIDENCE_TERMS: dict[str, float] = {
    "search": 3.0,
    "ranking": 3.0,
    "relevance": 3.0,
    "recommendation": 3.0,
    "recommender": 3.0,
    "retrieval": 3.0,
    "embedding": 2.8,
    "embeddings": 2.8,
    "vector": 2.7,
    "faiss": 2.7,
    "pinecone": 2.7,
    "milvus": 2.7,
    "qdrant": 2.7,
    "weaviate": 2.7,
    "bm25": 2.7,
    "elasticsearch": 2.4,
    "opensearch": 2.4,
    "lucene": 2.4,
    "semantic": 2.2,
    "personalization": 2.2,
    "personalized": 2.2,
    "ndcg": 2.5,
    "mrr": 2.5,
    "collaborative": 2.5,
    "factorization": 2.5,
    "matching": 2.0,
    "matched": 1.6,
    "ctr": 2.0,
    "click": 1.4,
    "map": 2.2,
    "a/b": 2.0,
    "experiment": 1.7,
    "experimentation": 1.7,
    "production": 1.8,
    "deployed": 1.7,
    "serving": 1.5,
    "latency": 1.4,
    "scale": 1.2,
    "users": 1.2,
    "traffic": 1.2,
    "nlp": 1.8,
    "query": 1.8,
    "intent": 1.5,
    "rag": 1.6,
    "llm": 1.4,
    "fine-tuned": 1.3,
    "fine-tuning": 1.3,
    "model": 1.0,
    "ml": 1.0,
    "machine": 1.0,
    "learning": 1.0,
    # broader production-search coverage beyond the fixed AI challenge JD
    "api": 2.0,
    "apis": 2.0,
    "microservice": 2.0,
    "microservices": 2.0,
    "distributed": 2.0,
    "backend": 1.8,
    "frontend": 1.6,
    "full-stack": 1.6,
    "fullstack": 1.6,
    "mobile": 1.6,
    "android": 1.6,
    "ios": 1.6,
    "react": 1.4,
    "node": 1.4,
    "django": 1.4,
    "fastapi": 1.4,
    "spring": 1.4,
    "kafka": 1.8,
    "redis": 1.6,
    "postgres": 1.6,
    "postgresql": 1.6,
    "mysql": 1.4,
    "mongodb": 1.4,
    "spark": 1.8,
    "airflow": 1.7,
    "etl": 1.6,
    "warehouse": 1.5,
    "analytics": 1.5,
    "dashboard": 1.2,
    "aws": 1.6,
    "gcp": 1.6,
    "azure": 1.6,
    "kubernetes": 1.7,
    "docker": 1.4,
    "terraform": 1.5,
    "ci/cd": 1.4,
    "devops": 1.4,
    "observability": 1.3,
    "monitoring": 1.2,
    "security": 1.5,
    "authentication": 1.4,
    "payments": 1.4,
}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) >= 25]


# Lexical evidence guards. Embeddings ignore negation and reward bare keyword
# lists, so aspirational sentences ("eager to learn embeddings...") and tool
# enumerations without a doing-verb are excluded from PREDICATE evidence.
# They stay in narrative_text (BM25 recall) - they just can't prove anything.
_ASPIRATIONAL = re.compile(
    r"\b(eager|hoping|aspiring|aiming|planning|wanting|looking|keen)\s+to\s+"
    r"(learn|transition|move|break into|upskill|grow into|pivot)"
    r"|\bhave not (yet )?(worked|used|done)"
    r"|\bno (prior|professional|production|hands.on) experience"
    r"|\binterested in (learning|transitioning|moving into)"
    r"|\bwould (love|like) to (learn|work on)",
    re.I,
)
_DOING_VERB = re.compile(
    r"\b(built|build|shipped|ship|deployed|deploy|designed|developed|implemented|"
    r"led|owned|own|created|launched|migrated|maintained|operated|ran|optimi[sz]ed|"
    r"architected|integrated|automated|scaled|reduced|improved)\b",
    re.I,
)


def is_evidence_grade(sentence: str) -> bool:
    """A sentence can serve as predicate evidence only if it describes work
    actually done - not aspiration, and not a bare tool list."""
    if _ASPIRATIONAL.search(sentence):
        return False
    commas = sentence.count(",")
    words = len(sentence.split())
    if commas >= 4 and commas / max(words, 1) > 0.18 and not _DOING_VERB.search(sentence):
        return False
    return True


def _evidence_score(sentence: str) -> float:
    toks = _WORD.findall(sentence.lower())
    score = 0.0
    for t in toks:
        hit = _CORE_EVIDENCE_TERMS.get(t, 0.0)
        if not hit and ("-" in t or "." in t):
            # hyphenated/dotted compounds ('embedding-based', 'node.js') must
            # still match their parts, else core evidence gets dropped
            hit = max(
                (_CORE_EVIDENCE_TERMS.get(p, 0.0) for p in re.split(r"[-.]", t)),
                default=0.0,
            )
        score += hit
    low = sentence.lower()
    if re.search(r"\b\d+(\.\d+)?\s*(m|million|k|thousand|gb|tb|ms|%)\b", low):
        score += 0.8
    if "real-time" in low or "real time" in low:
        score += 0.7
    if "offline evaluation" in low or "ab test" in low or "a/b test" in low:
        score += 1.0
    return score


def select_evidence_sentences(sentences: list[str]) -> list[str]:
    """Keep the most role-relevant narrative sentences, preserving source order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for sent in sentences:
        key = re.sub(r"\s+", " ", sent.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sent)
    sentences = deduped
    if len(sentences) <= _MAX_EVIDENCE_SENTENCES:
        return sentences
    scored = [(_evidence_score(s), i, s) for i, s in enumerate(sentences)]
    picked = sorted(scored, key=lambda x: (-x[0], x[1]))[:_MAX_EVIDENCE_SENTENCES]
    return [s for _score, _i, s in sorted(picked, key=lambda x: x[1])]


@dataclass
class LedgerRecord:
    candidate_id: str
    # profile basics
    headline: str = ""
    current_title: str = ""
    current_family: str = "other"
    current_level: int = 2
    current_mgr: bool = False
    yoe_stated: float = 0.0
    yoe_timeline: float = 0.0
    location_bucket: str = "abroad"
    willing_to_relocate: bool = False
    # career structure
    n_jobs: int = 0
    median_tenure_mo: float = 0.0
    product_share: float = 0.0
    services_share: float = 0.0
    research_share: float = 0.0
    services_only: bool = False
    research_only: bool = False
    families: list[str] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    trajectory_slope: float = 0.0
    mgr_track_recent: bool = False
    current_is_tech: bool = False
    # skills (normalized)
    skills: list[dict] = field(default_factory=list)  # {canon, category, proficiency, months, endorsements, corroborated}
    skill_categories: set[str] = field(default_factory=set)
    corroborated_categories: set[str] = field(default_factory=set)
    n_skills: int = 0
    n_expert_uncorroborated: int = 0
    assessment_scores: dict[str, float] = field(default_factory=dict)
    # education
    top_edu_tier: str = "unknown"
    edu_end_year: int | None = None
    # evidence sentences (narrative only)
    sentences: list[str] = field(default_factory=list)
    narrative_text: str = ""  # for BM25
    # credibility raw findings (filled by credibility.py)
    impossibilities: list[str] = field(default_factory=list)
    suspicions: list[str] = field(default_factory=list)
    # behavioral (verbatim copy)
    signals: dict = field(default_factory=dict)
    # salary/logistics
    notice_days: int = 0
    salary_min: float = 0.0
    salary_max: float = 0.0
    work_mode: str = "flexible"


def record_to_dict(rec: LedgerRecord) -> dict:
    """JSON-safe representation for the precomputed ledger artifact."""
    out = {}
    for f in fields(LedgerRecord):
        value = getattr(rec, f.name)
        if isinstance(value, set):
            value = sorted(value)
        out[f.name] = value
    return out


def record_from_dict(data: dict) -> LedgerRecord:
    """Restore a LedgerRecord from record_to_dict output."""
    rec = LedgerRecord(candidate_id=data["candidate_id"])
    field_names = {f.name for f in fields(LedgerRecord)}
    for key, value in data.items():
        if key not in field_names:
            continue
        if key in {"skill_categories", "corroborated_categories"}:
            value = set(value)
        setattr(rec, key, value)
    return rec


_PROF_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def build_record(c: dict, ref_year: int = 2026) -> LedgerRecord:
    p = c.get("profile", {})
    rec = LedgerRecord(candidate_id=c["candidate_id"])
    rec.headline = p.get("headline", "")
    rec.current_title = p.get("current_title", "")
    rec.current_family, rec.current_level, rec.current_mgr = normalize_title(
        rec.current_title or rec.headline
    )
    rec.yoe_stated = float(p.get("years_of_experience") or 0.0)
    sig = c.get("redrob_signals", {})
    rec.signals = sig
    rec.location_bucket = normalize_location(p.get("location", ""), p.get("country", ""))
    rec.willing_to_relocate = bool(sig.get("willing_to_relocate"))
    rec.notice_days = int(sig.get("notice_period_days") or 0)
    sal = sig.get("expected_salary_range_inr_lpa") or {}
    rec.salary_min = float(sal.get("min") or 0.0)
    rec.salary_max = float(sal.get("max") or 0.0)
    rec.work_mode = sig.get("preferred_work_mode", "flexible")

    # ---- career timeline ----------------------------------------------------
    jobs = c.get("career_history") or []
    rec.n_jobs = len(jobs)
    tenures: list[float] = []
    months_by_type: dict[str, float] = {"product": 0.0, "services": 0.0, "research": 0.0, "unknown": 0.0}
    total_months = 0.0
    sentences: list[str] = []
    narrative_parts: list[str] = []

    from .normalizer import TECH_FAMILIES  # local import to avoid cycle noise

    for j in sorted(jobs, key=lambda x: x.get("start_date") or ""):
        months = float(j.get("duration_months") or 0)
        tenures.append(months)
        total_months += months
        ctype = classify_company(j.get("company", ""), j.get("industry", ""))
        months_by_type[ctype] += months
        fam, lvl, _mgr = normalize_title(j.get("title", ""))
        rec.families.append(fam)
        rec.levels.append(lvl)
        desc = j.get("description", "") or ""
        narrative_parts.append(desc)
        sentences.extend(split_sentences(desc))

    summary = p.get("summary", "") or ""
    narrative_parts.append(summary)
    sentences.extend(split_sentences(summary))
    rec.sentences = select_evidence_sentences([s for s in sentences if is_evidence_grade(s)])
    rec.narrative_text = " ".join(narrative_parts).lower()

    rec.yoe_timeline = round(total_months / 12.0, 2)
    rec.median_tenure_mo = sorted(tenures)[len(tenures) // 2] if tenures else 0.0
    if total_months > 0:
        rec.product_share = months_by_type["product"] / total_months
        rec.services_share = months_by_type["services"] / total_months
        rec.research_share = months_by_type["research"] / total_months
        rec.services_only = months_by_type["services"] >= total_months * 0.999
        rec.research_only = months_by_type["research"] >= total_months * 0.999
    if len(rec.levels) >= 2:
        # simple slope: mean of consecutive level deltas (chronological order)
        deltas = [b - a for a, b in zip(rec.levels, rec.levels[1:])]
        rec.trajectory_slope = sum(deltas) / len(deltas)
    rec.mgr_track_recent = rec.current_mgr
    rec.current_is_tech = rec.current_family in TECH_FAMILIES

    # ---- education -----------------------------------------------------------
    tiers = [e.get("tier", "unknown") for e in (c.get("education") or [])]
    for t in ("tier_1", "tier_2", "tier_3", "tier_4"):
        if t in tiers:
            rec.top_edu_tier = t
            break
    end_years = [e.get("end_year") for e in (c.get("education") or []) if e.get("end_year")]
    rec.edu_end_year = max(end_years) if end_years else None

    # ---- skills (normalized + corroboration) ---------------------------------
    assess_raw = sig.get("skill_assessment_scores") or {}
    for k, v in assess_raw.items():
        canon, _cat = normalize_skill(k)
        rec.assessment_scores[canon] = float(v)

    # corroboration text = evidence-grade sentences ONLY: stuffed keyword lists
    # and aspirational mentions in the narrative must not corroborate anything
    corro_text = " ".join(rec.sentences).lower()
    for s in c.get("skills") or []:
        canon, cat = normalize_skill(s.get("name", ""))
        months = int(s.get("duration_months") or 0)
        prof = s.get("proficiency", "beginner")
        # corroboration: alias text in evidence-grade narrative OR assessment >= 60
        in_text = bool(canon) and (
            canon.replace("-", " ") in corro_text or canon in corro_text
        )
        assessed = rec.assessment_scores.get(canon, -1.0) >= 60.0
        corroborated = in_text or assessed
        rec.skills.append(
            {
                "canon": canon, "category": cat, "proficiency": prof,
                "months": months, "endorsements": int(s.get("endorsements") or 0),
                "corroborated": corroborated,
            }
        )
        rec.skill_categories.add(cat)
        if corroborated:
            rec.corroborated_categories.add(cat)
        if _PROF_RANK.get(prof, 1) >= 3 and not corroborated:
            rec.n_expert_uncorroborated += 1
    rec.n_skills = len(rec.skills)

    # ---- temporal impossibility raw checks (cheap; consumed by credibility) ---
    _temporal_checks(c, rec, ref_year)
    return rec


def _temporal_checks(c: dict, rec: LedgerRecord, ref_year: int) -> None:
    jobs = c.get("career_history") or []
    intervals: list[tuple[date, date, str]] = []
    for j in jobs:
        sd, ed = _parse_date(j.get("start_date")), _parse_date(j.get("end_date"))
        if sd is None:
            continue
        ed_eff = ed or date(ref_year, 6, 1)
        if ed_eff < sd:
            rec.impossibilities.append(f"role at {j.get('company','?')} ends before it starts")
            continue
        intervals.append((sd, ed_eff, j.get("company", "?")))
        # duration field vs date math
        months_field = float(j.get("duration_months") or 0)
        months_dates = (ed_eff - sd).days / 30.44
        if months_field > 0 and abs(months_field - months_dates) > 6:
            rec.suspicions.append(
                f"duration at {j.get('company','?')} says {months_field:.0f} mo but dates span {months_dates:.0f} mo"
            )
        # tenure vs known company founding year
        founded = COMPANY_FOUNDED.get((j.get("company") or "").strip().lower())
        if founded and sd.year < founded - 1:
            rec.impossibilities.append(
                f"claims joining {j.get('company')} in {sd.year}, before it existed (~{founded})"
            )
        if j.get("is_current") and ed is not None:
            rec.suspicions.append(f"current role at {j.get('company','?')} has an end date")

    # overlapping full-time roles (> 3 months overlap)
    intervals.sort()
    for (s1, e1, c1), (s2, e2, c2) in zip(intervals, intervals[1:]):
        overlap_days = (min(e1, e2) - max(s1, s2)).days
        if overlap_days > 92:
            rec.impossibilities.append(
                f"full-time roles at {c1} and {c2} overlap by ~{overlap_days // 30} months"
            )

    # stated YOE vs reconstructed timeline
    if rec.yoe_timeline > 0 and rec.yoe_stated > 0:
        delta = rec.yoe_stated - rec.yoe_timeline
        if delta > 3.0 and rec.yoe_stated > rec.yoe_timeline * 1.6:
            rec.impossibilities.append(
                f"states {rec.yoe_stated:.1f} yrs experience but career history totals {rec.yoe_timeline:.1f} yrs"
            )
        elif abs(delta) > 2.0:
            rec.suspicions.append(
                f"stated YOE {rec.yoe_stated:.1f} vs timeline {rec.yoe_timeline:.1f}"
            )

    # education chronology: career predates education end by a lot AND yoe counts it
    if rec.edu_end_year and intervals:
        first_start = intervals[0][0].year
        if first_start < rec.edu_end_year - 4:
            rec.suspicions.append(
                f"career start {first_start} predates graduation {rec.edu_end_year} by {rec.edu_end_year - first_start} yrs"
            )

    # skill duration vs total career length. Scan ALL skills before deciding:
    # an early mild violation must never shadow a later hard impossibility
    # (skill order in the profile is arbitrary). Still one message; don't spam.
    total_mo = rec.yoe_timeline * 12
    if total_mo > 0 and rec.skills:
        worst = max(rec.skills, key=lambda s: s["months"])
        if worst["months"] > total_mo + 36 and worst["months"] > total_mo * 1.5:
            rec.impossibilities.append(
                f"skill '{worst['canon']}' claims {worst['months']} mo of use vs {total_mo:.0f} mo total career"
            )
        elif worst["months"] > total_mo + 12:
            rec.suspicions.append(
                f"skill '{worst['canon']}' claims {worst['months']} mo of use vs {total_mo:.0f} mo total career"
            )
    # expert/advanced with ~zero usage
    zero_expert = [
        s["canon"] for s in rec.skills
        if _PROF_RANK.get(s["proficiency"], 1) >= 3 and s["months"] <= 1
    ]
    if len(zero_expert) >= 3:
        rec.impossibilities.append(
            f"{len(zero_expert)} skills claimed advanced/expert with ~0 months of use ({', '.join(zero_expert[:3])}...)"
        )
    elif len(zero_expert) >= 1:
        rec.suspicions.append(
            f"advanced/expert claim with ~0 months use: {', '.join(zero_expert)}"
        )
