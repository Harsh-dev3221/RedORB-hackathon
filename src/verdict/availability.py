"""L4 Availability Engine: A in [0,1] from the 23 redrob_signals + logistics.

Geometric blend of four sub-scores so a zero in any group genuinely hurts.
Sentinels (-1 = no data) are treated as unknown-mild-prior, never as zero.
Popularity signals are capped (Indeed's popularity-bias lesson).
"""

from __future__ import annotations

import math
from datetime import date

from .evidence import LedgerRecord

REF_DATE = date(2026, 6, 1)


def _days_since(s: str | None) -> float:
    if not s:
        return 999.0
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return max((REF_DATE - date(y, m, d)).days, 0)
    except ValueError:
        return 999.0


def score_availability(rec: LedgerRecord, rubric: dict) -> tuple[float, list[str]]:
    sig = rec.signals
    flags: list[str] = []
    cfg = rubric["availability"]

    # --- engagement ---
    days_idle = _days_since(sig.get("last_active_date"))
    recency = math.exp(-days_idle / 60.0)  # ~0.6 at 1 mo, ~0.22 at 3 mo, ~0.05 at 6 mo
    open_tw = 1.0 if sig.get("open_to_work_flag") else 0.55
    completeness = 0.5 + 0.5 * float(sig.get("profile_completeness_score") or 50) / 100.0
    engagement = recency * open_tw * completeness
    if days_idle > 150:
        flags.append(f"GHOST: last active {days_idle:.0f} days ago")

    # --- responsiveness ---
    rr = float(sig.get("recruiter_response_rate") or 0.0)
    resp_rate = 0.15 + 0.85 * rr
    rt_h = float(sig.get("avg_response_time_hours") or 72.0)
    resp_time = math.exp(-rt_h / 96.0)
    icr = float(sig.get("interview_completion_rate") if sig.get("interview_completion_rate") is not None else 0.7)
    oar = float(sig.get("offer_acceptance_rate") if sig.get("offer_acceptance_rate") is not None else -1)
    oar_s = 0.7 if oar < 0 else 0.3 + 0.7 * oar  # -1 sentinel -> mild prior
    responsiveness = resp_rate * (0.5 + 0.5 * resp_time) * (0.4 + 0.6 * icr) * oar_s
    if rr < 0.15:
        flags.append(f"LOW_RESPONSE: {rr:.0%} recruiter response rate")

    # --- logistics ---
    if rec.notice_days <= 30:
        notice = 1.0
    elif rec.notice_days <= 60:
        notice = 0.8
    elif rec.notice_days <= 90:
        notice = 0.6
    else:
        notice = 0.45
        flags.append(f"LONG_NOTICE: {rec.notice_days}-day notice period")
    loc_scores = cfg["location_scores"]
    loc = loc_scores.get(rec.location_bucket, 0.3)
    if rec.location_bucket in ("india_other", "abroad") and rec.willing_to_relocate:
        loc = max(loc, loc_scores["relocator"])
    band_lo, band_hi = cfg["salary_band_lpa"]
    if rec.salary_min <= 0:
        salary = 0.9
    elif rec.salary_min <= band_hi:
        salary = 1.0
    elif rec.salary_min <= band_hi * 1.4:
        salary = 0.7
        flags.append(f"SALARY_STRETCH: expects {rec.salary_min:.0f}+ LPA")
    else:
        salary = 0.4
        flags.append(f"SALARY_GAP: expects {rec.salary_min:.0f}+ LPA vs ~{band_hi:.0f} band")
    mode = 1.0 if rec.work_mode in ("hybrid", "onsite", "flexible") else 0.85
    logistics = notice * loc * salary * mode

    # --- market corroboration (capped, smallest weight) ---
    saves = min(float(sig.get("saved_by_recruiters_30d") or 0), 10.0)
    views = min(float(sig.get("profile_views_received_30d") or 0), 50.0)
    verified = 0.7 + 0.1 * sum(
        bool(sig.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected")
    )
    corro = (0.6 + 0.4 * (saves / 10.0 * 0.5 + views / 50.0 * 0.5)) * verified

    w = cfg["weights"]  # geometric blend
    a = (
        max(engagement, 1e-4) ** w["engagement"]
        * max(responsiveness, 1e-4) ** w["responsiveness"]
        * max(logistics, 1e-4) ** w["logistics"]
        * max(min(corro, 1.0), 1e-4) ** w["corroboration"]
    )
    # Challenge trap: a perfect-on-paper profile that is both stale and
    # non-responsive is not practically hireable. Keep this as a cap rather
    # than another multiplicative term so the failure mode is auditable.
    if days_idle >= 180 and rr <= 0.10:
        flags.append(f"UNREACHABLE: last active {days_idle:.0f} days ago with {rr:.0%} response rate")
        a = min(a, 0.12)
    elif days_idle >= 240:
        flags.append(f"STALE_PROFILE: last active {days_idle:.0f} days ago")
        a = min(a, 0.18)
    elif rr <= 0.05 and not sig.get("open_to_work_flag"):
        flags.append(f"PASSIVE_LOW_RESPONSE: not open to work and {rr:.0%} response rate")
        a = min(a, 0.20)
    if rec.notice_days > 120:
        flags.append(f"EXTREME_NOTICE: {rec.notice_days}-day notice period")
        a = min(a, 0.45)
    elif rec.notice_days > 90:
        a = min(a, 0.60)
    return min(a, 1.0), flags
