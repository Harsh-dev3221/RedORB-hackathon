"""Index drift monitoring: detect when accumulated ingests shift the score
population enough that probe/threshold recalibration is due.

Mechanism: a deterministic ~2% sample of the pool is fully scored; each
snapshot stores the fused-score decile profile + per-predicate firing rates.
Drift = Population Stability Index (PSI) of the current profile against the
baseline. Industry convention: PSI < 0.10 stable, 0.10-0.25 moderate shift,
> 0.25 recalibrate.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import orjson

from .pipeline import Index, score_candidate

LOG = Path(__file__).resolve().parents[2] / "artifacts" / "drift_log.jsonl"
SAMPLE_MOD = 50  # ~2% deterministic sample


def _in_sample(cid: str) -> bool:
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % SAMPLE_MOD == 0


def snapshot(idx: Index, rubric: dict, probes: dict) -> dict:
    fused, js, cs, As = [], [], [], []
    fire_counts = {pid: 0 for pid in rubric["fuzzy_predicates"]}
    f = rubric["fusion"]
    n = 0
    for i, cid in enumerate(idx.ids):
        if not _in_sample(cid):
            continue
        n += 1
        s = score_candidate(idx, i, rubric, probes)
        score = math.exp(
            f["alpha"] * math.log(max(s.j, 1e-4))
            + f["beta"] * math.log(max(s.c, 1e-4))
            + f["gamma"] * math.log(max(s.a, 1e-4))
        )
        fused.append(score)
        js.append(s.j); cs.append(s.c); As.append(s.a)
        for pid in fire_counts:
            if s.rule_scores.get(pid, 0.0) >= 0.5:
                fire_counts[pid] += 1
    fused_arr = np.asarray(fused)
    deciles = np.percentile(fused_arr, np.arange(10, 100, 10)).tolist()
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_indexed": len(idx.ids),
        "n_sampled": n,
        "fused_deciles": [round(x, 5) for x in deciles],
        "mean_J": round(float(np.mean(js)), 4),
        "mean_C": round(float(np.mean(cs)), 4),
        "mean_A": round(float(np.mean(As)), 4),
        "predicate_fire_rate": {p: round(c / max(n, 1), 4) for p, c in fire_counts.items()},
    }


def psi(baseline_deciles: list[float], current_scores_deciles: list[float],
        baseline_n: int = 10) -> float:
    """PSI between two decile profiles: treat each as implied 10-bin histograms
    on the baseline's bin edges (approximation via decile displacement)."""
    base = np.asarray(baseline_deciles)
    cur = np.asarray(current_scores_deciles)
    # each decile boundary moving past a neighbor ~ one bin of mass moved
    edges = np.concatenate([[-np.inf], base, [np.inf]])
    # current implied mass per baseline bin: fraction of current deciles in each bin
    counts, _ = np.histogram(cur, bins=edges)
    p_cur = np.maximum(counts / len(cur), 1e-4)
    p_base = np.full(len(p_cur), 1.0 / len(p_cur))  # baseline deciles are uniform by construction
    p_cur = p_cur / p_cur.sum()
    return float(np.sum((p_cur - p_base) * np.log(p_cur / p_base)))


def record(idx: Index, rubric: dict, probes: dict) -> dict:
    snap = snapshot(idx, rubric, probes)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("ab") as fh:
        fh.write(orjson.dumps(snap) + b"\n")
    return snap


def check(threshold_moderate: float = 0.10, threshold_alert: float = 0.25) -> dict:
    if not LOG.exists():
        raise SystemExit("no drift log yet - run `drift_monitor.py record` first")
    snaps = [orjson.loads(line) for line in LOG.read_bytes().splitlines() if line.strip()]
    if len(snaps) < 2:
        return {"status": "baseline-only", "snapshots": len(snaps), "psi": 0.0,
                "baseline": snaps[0]}
    base, cur = snaps[0], snaps[-1]
    score_psi = psi(base["fused_deciles"], cur["fused_deciles"])
    pred_shifts = {
        p: round(cur["predicate_fire_rate"].get(p, 0) - r, 4)
        for p, r in base["predicate_fire_rate"].items()
        if abs(cur["predicate_fire_rate"].get(p, 0) - r) > 0.03
    }
    status = ("RECALIBRATE" if score_psi > threshold_alert
              else "moderate-shift" if score_psi > threshold_moderate
              else "stable")
    return {
        "status": status, "psi": round(score_psi, 4),
        "snapshots": len(snaps),
        "baseline_ts": base["ts"], "current_ts": cur["ts"],
        "indexed_then_now": [base["n_indexed"], cur["n_indexed"]],
        "mean_JCA_then": [base["mean_J"], base["mean_C"], base["mean_A"]],
        "mean_JCA_now": [cur["mean_J"], cur["mean_C"], cur["mean_A"]],
        "predicate_fire_rate_shifts": pred_shifts,
        "advice": ("rebuild probes (scripts/build_probes.py) and recalibrate predicate floors "
                   "(scripts/calibrate_predicates.py)" if status == "RECALIBRATE" else
                   "no action needed" if status == "stable" else
                   "watch next ingests; recalibration not yet required"),
    }
