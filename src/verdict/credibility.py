"""L2 Credibility Engine: C in [0,1] + named soft flags.

FEVER-style twist: the profile is its own evidence corpus. Impossibilities
(physical contradictions) collapse C catastrophically; divergences (unsupported
claims) reduce it proportionally. Honeypots need no special-casing - they
contradict themselves.
"""

from __future__ import annotations

from .evidence import LedgerRecord

# trending keywords whose presence in a non-tech profile signals stuffing
_TRENDY = {"rag", "llm", "fine-tuning", "prompt-engineering", "langchain", "agents",
           "embeddings", "vector-db", "transformers"}
_NONTECH_FAMILIES = {"marketing", "hr", "sales", "design", "finance", "ops", "analyst",
                     "pm", "product", "other"}


def score_credibility(rec: LedgerRecord) -> tuple[float, list[str]]:
    """Return (C, flags). Flags are short audit strings used in reasoning."""
    c = 1.0
    flags: list[str] = []

    # --- impossibilities: catastrophic, multiplicative ---
    if rec.impossibilities:
        c *= 0.03 ** min(len(rec.impossibilities), 2)  # 1 hit -> 0.03, 2+ -> ~0.001
        flags.append("TIMELINE_IMPOSSIBLE: " + rec.impossibilities[0])

    # --- proportional doubts ---
    n_susp = len(rec.suspicions)
    if n_susp:
        c *= max(0.55, 1.0 - 0.12 * n_susp)
        flags.append("CONSISTENCY_DOUBT: " + rec.suspicions[0])

    # uncorroborated advanced/expert claims
    if rec.n_skills > 0:
        uncorr_share = rec.n_expert_uncorroborated / rec.n_skills
        if uncorr_share > 0.4 and rec.n_expert_uncorroborated >= 4:
            c *= 0.6
            flags.append(
                f"UNSUPPORTED_SKILL_CLUSTER: {rec.n_expert_uncorroborated} advanced/expert "
                "skills with no narrative or assessment support"
            )
        elif uncorr_share > 0.25 and rec.n_expert_uncorroborated >= 3:
            c *= 0.8
            flags.append(
                f"UNSUPPORTED_SKILLS: {rec.n_expert_uncorroborated} high-proficiency skills uncorroborated"
            )

    # stuffing signature: trendy AI skills on a clearly non-technical career
    trendy_claimed = {s["canon"] for s in rec.skills} & _TRENDY
    families = set(rec.families) | {rec.current_family}
    is_nontech_career = families <= _NONTECH_FAMILIES
    if is_nontech_career and len(trendy_claimed) >= 3:
        c *= 0.35
        flags.append(
            f"STUFFING_PATTERN: {len(trendy_claimed)} trending AI skills on a "
            f"{rec.current_family} career with no technical roles"
        )
    # breadth without depth: huge skill list, low median duration
    if rec.n_skills >= 18:
        med_mo = sorted(s["months"] for s in rec.skills)[rec.n_skills // 2]
        if med_mo <= 6:
            c *= 0.7
            flags.append(f"BREADTH_NO_DEPTH: {rec.n_skills} skills, median {med_mo} months each")

    return max(c, 0.001), flags
