"""L5 Verdict Fusion + finalist tournament.

Score = J^alpha * C^beta * A^gamma (log-space). The head (~300) is then
re-ordered by a deterministic Bradley-Terry-style tournament over per-rule
comparisons - comparison beats absolute scoring at the top of the list
(the RankGPT lesson, executed without an LLM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scored:
    idx: int                      # index into the recall-set record list
    candidate_id: str
    j: float
    c: float
    a: float
    rule_scores: dict[str, float]
    evidence_notes: list[str]
    dampener_notes: list[str]
    flags: list[str] = field(default_factory=list)
    fused: float = 0.0
    final: float = 0.0


def fuse(items: list[Scored], rubric: dict) -> None:
    f = rubric["fusion"]
    for it in items:
        log_s = (
            f["alpha"] * math.log(max(it.j, 1e-4))
            + f["beta"] * math.log(max(it.c, 1e-4))
            + f["gamma"] * math.log(max(it.a, 1e-4))
        )
        it.fused = math.exp(log_s)


# rules that matter most for head-of-list pairwise comparison
_KEY_RULES = [
    ("shipped_search_ranking_reco", 3.0),
    ("production_embeddings_retrieval", 3.0),
    ("ranking_evaluation", 1.5),
    ("core_skill_coverage", 1.5),
    ("product_company_tenure", 1.5),
    ("core_title_family", 1.0),
    ("yoe_fit", 0.7),
]


def tournament(head: list[Scored], rubric: dict) -> None:
    """Borda/Bradley-Terry aggregation with fused score as the absolute anchor."""
    n = len(head)
    if n <= 2:
        for it in head:
            it.final = it.fused
        return
    feats = np.zeros((n, len(_KEY_RULES) + 3), dtype=np.float32)
    for i, it in enumerate(head):
        for k, (rule, w) in enumerate(_KEY_RULES):
            feats[i, k] = w * it.rule_scores.get(rule, 0.0)
        feats[i, -3] = 2.0 * it.c
        feats[i, -2] = 1.2 * it.a
        feats[i, -1] = -0.4 * len(it.flags)
    # pairwise margins -> win prob -> mean (Borda strength)
    strength = np.zeros(n, dtype=np.float64)
    for i in range(n):
        margins = feats[i] - feats  # [n, d]
        wins = 1.0 / (1.0 + np.exp(-margins.sum(axis=1)))
        strength[i] = (wins.sum() - 0.5) / (n - 1)  # exclude self (=0.5)
    # Keep absolute fused magnitude. Earlier versions normalized the head to
    # 0..1, which made weak-fit samples look artificially excellent.
    fused = np.array([it.fused for it in head], dtype=np.float64)
    blend = rubric["fusion"]["tournament_blend"]
    for i, it in enumerate(head):
        multiplier = 1.0 + blend * (strength[i] - 0.5)
        it.final = float(it.fused * max(multiplier, 0.75))


def finalize(items: list[Scored], rubric: dict) -> list[Scored]:
    """Fuse all, tournament the head, return final top-N sorted (ties: id asc)."""
    fuse(items, rubric)
    items.sort(key=lambda x: (-x.fused, x.candidate_id))
    head_n = int(rubric["fusion"]["head_size"])
    head, tail = items[:head_n], items[head_n:]
    tournament(head, rubric)
    head.sort(key=lambda x: (-x.final, x.candidate_id))
    # tail keeps fused order, scaled below the head's final floor
    floor = head[-1].final if head else 1.0
    for k, it in enumerate(tail):
        it.final = floor * 0.95 * (1.0 - k / max(len(tail), 1) * 0.5)
    out = head + tail
    n_final = int(rubric["fusion"]["final_size"])
    return out[:n_final]
