"""L6 Reasoning Synthesizer: 1-2 sentence justification per ranked candidate.

Assembly, not generation: content slots are filled ONLY from the candidate's
ledger fields and fired-rule evidence - hallucination is structurally
impossible. Frame rotation (seeded by candidate_id) + differing evidence
gives variation; tone follows rank band; the top flag/dampener is always
voiced as an honest concern.
"""

from __future__ import annotations

import re

from .evidence import LedgerRecord
from .fusion import Scored

_PRED_LABELS = {
    "shipped_search_ranking_reco": "shipped search/ranking/recommendation work",
    "production_embeddings_retrieval": "production embeddings/vector retrieval",
    "ranking_evaluation": "ranking evaluation (NDCG/A-B testing)",
    "ml_production_scale": "production ML at scale",
    "nlp_ir_depth": "pre-LLM NLP/IR depth",
    "llm_production": "LLM fine-tuning/RAG in production",
}


def _strengths(s: Scored, rec: LedgerRecord) -> list[str]:
    out: list[str] = []
    fired = sorted(
        ((pid, sc) for pid, sc in s.rule_scores.items() if pid in _PRED_LABELS and sc >= 0.55),
        key=lambda x: -x[1],
    )
    for pid, _sc in fired[:2]:
        out.append(_PRED_LABELS[pid])
    if rec.product_share >= 0.6 and len(out) < 3:
        out.append(f"{rec.product_share:.0%} of career at product companies")
    cov = s.rule_scores.get("core_skill_coverage", 0)
    if cov >= 0.5 and len(out) < 3:
        k = round(cov * 7)
        out.append(f"corroborated skills across {k}/7 core areas")
    if not out and rec.current_is_tech:
        out.append(f"hands-on {rec.current_family.replace('_', ' ')} background")
    return out[:3]


def _concern(s: Scored, rec: LedgerRecord) -> str:
    if s.dampener_notes:
        return s.dampener_notes[0]
    for fl in s.flags:
        tag, _, detail = fl.partition(": ")
        if tag in ("TIMELINE_IMPOSSIBLE", "STUFFING_PATTERN", "GHOST", "LOW_RESPONSE",
                   "LONG_NOTICE", "SALARY_GAP", "UNSUPPORTED_SKILL_CLUSTER"):
            return detail or tag.lower().replace("_", " ")
    if rec.notice_days > 60:
        return f"{rec.notice_days}-day notice period"
    if s.rule_scores.get("production_embeddings_retrieval", 0) < 0.3:
        return "no direct evidence of production vector-retrieval work"
    return ""


def _behavior(rec: LedgerRecord) -> str:
    rr = float(rec.signals.get("recruiter_response_rate") or 0)
    bits = []
    if rr >= 0.5:
        bits.append(f"{rr:.0%} response rate")
    if rec.notice_days <= 30:
        bits.append(f"{rec.notice_days}-day notice")
    if rec.location_bucket == "preferred":
        bits.append("already in Pune/Noida")
    elif rec.location_bucket == "tier1":
        bits.append("Tier-1 city based")
    elif rec.willing_to_relocate:
        bits.append("open to relocation")
    return ", ".join(bits[:2])


_CLEAN = re.compile(r"\s+")


def synthesize(s: Scored, rec: LedgerRecord, rank: int) -> str:
    seed = sum(ord(ch) for ch in s.candidate_id) % 4
    yoe = max(rec.yoe_stated, rec.yoe_timeline)
    title = rec.current_title or rec.current_family.replace("_", " ")
    strengths = _strengths(s, rec)
    concern = _concern(s, rec)
    behavior = _behavior(rec)

    strong = s.j >= 0.45 and s.c >= 0.5
    solid = s.j >= 0.25 and s.c >= 0.5
    head = rank <= 10 and strong
    mid = (rank <= 50 and solid) or (10 < rank <= 50 and not strong)

    if head:
        frames = [
            "{title} with {yoe:.1f} yrs; {s1}{s2}.{beh}{con}",
            "{yoe:.1f} yrs experience, currently {title}: {s1}{s2}.{beh}{con}",
            "Strong fit - {title}, {yoe:.1f} yrs; evidence of {s1}{s2}.{beh}{con}",
            "{title} ({yoe:.1f} yrs) showing {s1}{s2}.{beh}{con}",
        ]
    elif mid:
        frames = [
            "Solid {title}, {yoe:.1f} yrs; {s1}{s2}.{beh}{con}",
            "{title} with {yoe:.1f} yrs and {s1}{s2}.{beh}{con}",
            "{yoe:.1f}-yr {title}; {s1}{s2}.{beh}{con}",
            "Credible fit: {title}, {yoe:.1f} yrs, {s1}{s2}.{beh}{con}",
        ]
    else:
        frames = [
            "Adjacent fit - {title}, {yoe:.1f} yrs; {s1}{s2}.{con_lead}{beh}",
            "{title} ({yoe:.1f} yrs): {s1}{s2}, below the bar on core asks.{con_lead}{beh}",
            "Borderline: {title} with {yoe:.1f} yrs; {s1}{s2}.{con_lead}{beh}",
            "{yoe:.1f}-yr {title}; partial match ({s1}{s2}).{con_lead}{beh}",
        ]

    s1 = strengths[0] if strengths else "general engineering background"
    s2 = f" and {strengths[1]}" if len(strengths) > 1 else ""
    beh = f" {behavior[0].upper()}{behavior[1:]}." if behavior else ""
    con = f" Concern: {concern}." if concern else ""
    con_lead = f" Main gap: {concern}." if concern else ""

    txt = frames[seed].format(
        title=title, yoe=yoe, s1=s1, s2=s2, beh=beh, con=con, con_lead=con_lead
    )
    txt = _CLEAN.sub(" ", txt).strip()
    # CSV hygiene: no internal double quotes
    return txt.replace('"', "'")[:280]
