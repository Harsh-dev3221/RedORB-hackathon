"""L3 Judgment Engine: executes the rubric program over the Claim Ledger.

J = weighted crisp rules + fuzzy predicates (evidence-sentence retrieval),
multiplied by the JD's explicit disqualifier dampeners. Every fired rule
returns an evidence note for the Reasoning Synthesizer.
"""

from __future__ import annotations

import numpy as np

from .evidence import LedgerRecord


def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def predicate_scores(
    sent_vecs: np.ndarray,        # [n_sent, 384] this candidate's sentence vectors
    pred_vecs: dict[str, np.ndarray],   # predicate -> [n_pos, 384]
    neg_vecs: np.ndarray,         # [n_neg, 384]
    cfg: dict,
) -> dict[str, tuple[float, int]]:
    """Return predicate -> (score 0..1, best_sentence_idx)."""
    out: dict[str, tuple[float, int]] = {}
    if sent_vecs.shape[0] == 0:
        return {p: (0.0, -1) for p in pred_vecs}
    neg_max = (sent_vecs @ neg_vecs.T).max(axis=1)  # [n_sent]
    floor, full = cfg["sim_floor"], cfg["sim_full"]
    margin = cfg["neg_margin"]
    for pid, pv in pred_vecs.items():
        pos = (sent_vecs @ pv.T).max(axis=1)        # [n_sent]
        adj = pos - margin * np.maximum(neg_max, 0)
        best = int(np.argmax(adj))
        raw = float(adj[best])
        score = min(max((raw - floor) / (full - floor), 0.0), 1.0)
        out[pid] = (score, best if score > 0 else -1)
    return out


def judge(
    rec: LedgerRecord,
    preds: dict[str, tuple[float, int]],
    rubric: dict,
) -> tuple[float, dict[str, float], list[str], list[str]]:
    """Return (J, rule_scores, evidence_notes, dampener_notes)."""
    cr = rubric["crisp_rules"]
    scores: dict[str, float] = {}
    notes: list[str] = []

    a, b, c, d = cr["yoe_fit"]["trapezoid"]
    yoe = max(rec.yoe_stated, rec.yoe_timeline)
    scores["yoe_fit"] = _trapezoid(yoe, a, b, c, d)

    fam_cfg = cr["core_title_family"]
    if rec.current_family in fam_cfg["core_families"]:
        scores["core_title_family"] = 1.0
        notes.append(f"current role '{rec.current_title}' is squarely applied-ML/IR")
    elif any(f in fam_cfg["core_families"] for f in rec.families):
        scores["core_title_family"] = 0.7
        notes.append("prior applied-ML/IR role in career history")
    elif rec.current_family in fam_cfg["adjacent_families"]:
        scores["core_title_family"] = fam_cfg["adjacent_score"]
    else:
        scores["core_title_family"] = 0.0

    scores["product_company_tenure"] = rec.product_share
    if rec.product_share >= 0.6:
        notes.append(f"{rec.product_share:.0%} of career at product companies")

    scores["trajectory"] = 1.0 if rec.trajectory_slope > 0 else (0.6 if rec.trajectory_slope == 0 else 0.25)
    scores["stability"] = min(rec.median_tenure_mo / cr["stability"]["full_at_months"], 1.0)

    loc_cfg = cr["location_fit"]
    loc = loc_cfg["scores"].get(rec.location_bucket, 0.25)
    if rec.location_bucket in ("india_other", "abroad") and rec.willing_to_relocate:
        loc = min(loc + loc_cfg["relocator_bonus"], 0.85)
    scores["location_fit"] = loc

    gh = float(rec.signals.get("github_activity_score", -1))
    scores["external_validation"] = 0.5 if gh < 0 else min(gh / 70.0, 1.0)

    cov_cfg = cr["core_skill_coverage"]
    core_set = set(cov_cfg["core_categories"])
    covered = (rec.corroborated_categories if cov_cfg["corroborated_only"] else rec.skill_categories) & core_set
    scores["core_skill_coverage"] = len(covered) / len(core_set)
    if len(covered) >= 4:
        notes.append(f"corroborated depth across {len(covered)} core areas ({', '.join(sorted(covered)[:4])})")

    j = sum(cr[k]["weight"] * v for k, v in scores.items())

    for pid, pcfg in rubric["fuzzy_predicates"].items():
        s, best_idx = preds.get(pid, (0.0, -1))
        scores[pid] = s
        j += pcfg["weight"] * s
        if s >= 0.55 and best_idx >= 0 and best_idx < len(rec.sentences):
            snippet = rec.sentences[best_idx]
            notes.append(f"[{pid}] \"{snippet[:110]}\"")

    # --- dampeners: the JD's explicit disqualifier logic ---
    dmp = rubric["dampeners"]
    dnotes: list[str] = []
    if rec.research_only:
        j *= dmp["research_only"]["factor"]
        dnotes.append("entire career in research roles - JD screens out research-only backgrounds")
    if rec.services_only:
        j *= dmp["services_only"]["factor"]
        dnotes.append("entire career at IT-services/consulting firms - JD explicitly screens this out")
    h = dmp["title_hopper"]
    if rec.n_jobs >= h["min_jobs"] and rec.median_tenure_mo < h["median_tenure_below_mo"]:
        j *= h["factor"]
        dnotes.append(f"median tenure ~{rec.median_tenure_mo:.0f} months across {rec.n_jobs} roles - hop-rate concern")
    m = dmp["manager_no_code"]
    if rec.current_level >= 6 and rec.mgr_track_recent:
        j *= m["director_factor"]
        dnotes.append("director+ track - JD requires hands-on production coding")
    elif rec.mgr_track_recent:
        j *= m["factor"]
        dnotes.append("management-track current role - JD requires recent production code")
    claimed = rec.skill_categories
    corro = rec.corroborated_categories
    nlp_ir = {"nlp", "search", "ranking", "embeddings", "vector_db", "llm"}
    if ({"vision", "speech", "robotics"} & corro) and not (nlp_ir & corro) and "ml_core" not in corro:
        j *= dmp["cv_speech_only"]["factor"]
        dnotes.append("CV/speech focus without NLP/IR evidence - JD flags this as re-learning fundamentals")
    if "llm_framework" in claimed and not (corro & (nlp_ir | {"ml_core"})):
        j *= dmp["framework_only_ai"]["factor"]
        dnotes.append("framework-level AI claims without corroborated ML/IR depth")

    tech_fams = {
        "ml_engineer", "applied_scientist", "nlp_engineer", "search_engineer",
        "data_scientist", "data_engineer", "mlops_engineer", "backend", "fullstack",
        "swe", "devops",
    }
    ml_ir_fams = {"ml_engineer", "applied_scientist", "nlp_engineer", "search_engineer", "data_scientist"}
    has_tech_role = rec.current_family in tech_fams or bool(set(rec.families) & tech_fams)
    has_ml_ir_role = rec.current_family in ml_ir_fams or bool(set(rec.families) & ml_ir_fams)
    fuzzy_strength = max(
        (scores.get(pid, 0.0) for pid in rubric["fuzzy_predicates"]),
        default=0.0,
    )
    core_evidence = corro & (nlp_ir | {"ml_core", "mlops"})
    if not has_tech_role and not core_evidence:
        j *= 0.20
        dnotes.append("non-technical career with no corroborated ML/IR evidence")
    elif not has_ml_ir_role and len(core_evidence) < 2 and fuzzy_strength < 0.55:
        j *= 0.40
        dnotes.append("adjacent tech profile without enough corroborated ML/IR depth")
    elif not core_evidence and fuzzy_strength < 0.35:
        j *= 0.55
        dnotes.append("product/company fit is not backed by core ML/IR evidence")

    return min(j, 1.0), scores, notes, dnotes
