"""Small-sample Redrob sandbox for Streamlit/HuggingFace Spaces.

The official Stage-3 command is still `rank.py`; this app is a lightweight
demo surface for the submission portal's sandbox requirement. It accepts <=100
candidates and ranks them with the same claim ledger, rubric rules,
credibility, availability, fusion, and reasoning code. Fuzzy evidence predicates
use a deterministic lexical approximation so the sandbox does not need the full
~1GB precomputed vector artifact.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import orjson
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.availability import score_availability
from verdict.credibility import score_credibility
from verdict.data import iter_candidates
from verdict.evidence import build_record
from verdict.fusion import Scored, finalize
from verdict.judgment import judge
from verdict.reasoning import synthesize
from verdict.recall import passes_gates

MAX_SAMPLE = 100
TOKEN = re.compile(r"[a-z0-9+#\-]{2,}")
STOP = frozenset(
    "the a an and or of to in for with on at by from as is are was were be been "
    "this that it its we our i my you your they their he she his her them us".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if t not in STOP}


@st.cache_data(show_spinner=False)
def load_rubric() -> dict:
    return orjson.loads((ROOT / "artifacts" / "rubric_program.json").read_bytes())


def _load_uploaded_candidates(raw: bytes, name: str) -> list[dict]:
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        data = orjson.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError("JSON upload must be a candidate object or an array of candidate objects")

    out = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(orjson.loads(line))
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL on line {line_no}: {exc}") from exc
    return out


@st.cache_data(show_spinner=False)
def _load_sample_candidates() -> list[dict]:
    sample = ROOT / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
    if sample.exists():
        with NamedTemporaryFile("wb", suffix=".json", delete=False) as tmp:
            tmp.write(sample.read_bytes())
            tmp_path = tmp.name
        try:
            return list(iter_candidates(tmp_path))[:50]
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return []


def _lexical_predicates(rec, rubric: dict) -> dict[str, tuple[float, int]]:
    """Approximate fuzzy predicates without embeddings for small sandbox runs."""
    out: dict[str, tuple[float, int]] = {}
    sent_tokens = [_tokens(s) for s in rec.sentences]
    neg_tokens = set()
    for neg in rubric.get("predicate_negatives", []):
        neg_tokens |= _tokens(neg)

    for pid, cfg in rubric["fuzzy_predicates"].items():
        phrase_sets = [_tokens(p) for p in cfg.get("positives", [])]
        best_score = 0.0
        best_idx = -1
        for i, toks in enumerate(sent_tokens):
            if not toks:
                continue
            score = 0.0
            for pset in phrase_sets:
                if not pset:
                    continue
                overlap = len(toks & pset) / math.sqrt(len(toks) * len(pset))
                score = max(score, overlap)
            neg_overlap = len(toks & neg_tokens) / max(len(toks), 1)
            adjusted = max(score - 0.35 * neg_overlap, 0.0)
            if adjusted > best_score:
                best_score = adjusted
                best_idx = i
        # Calibrated only for the sandbox: exact phrase/category evidence tends
        # to land above 0.55, weak lexical contact below 0.3.
        out[pid] = (min(best_score * 1.8, 1.0), best_idx if best_score >= 0.30 else -1)
    return out


def rank_sample(candidates: list[dict], rubric: dict) -> list[dict]:
    records = [build_record(c) for c in candidates]
    scored: list[Scored] = []
    for idx, rec in enumerate(records):
        if not passes_gates(rec, rubric["gates"]):
            continue
        preds = _lexical_predicates(rec, rubric)
        j, rules, notes, dnotes = judge(rec, preds, rubric)
        c_score, c_flags = score_credibility(rec, rubric.get("credibility"))
        a_score, a_flags = score_availability(rec, rubric)
        scored.append(
            Scored(
                idx=idx,
                candidate_id=rec.candidate_id,
                j=j,
                c=c_score,
                a=a_score,
                rule_scores=rules,
                evidence_notes=notes,
                dampener_notes=dnotes,
                flags=c_flags + a_flags,
            )
        )

    if not scored:
        return []
    local_rubric = dict(rubric)
    local_rubric["fusion"] = dict(rubric["fusion"])
    local_rubric["fusion"]["final_size"] = min(len(scored), MAX_SAMPLE)
    local_rubric["fusion"]["head_size"] = min(int(rubric["fusion"]["head_size"]), len(scored))
    top = finalize(scored, local_rubric)

    rows = []
    for rank, item in enumerate(top, 1):
        rec = records[item.idx]
        rows.append(
            {
                "candidate_id": item.candidate_id,
                "rank": rank,
                "score": f"{item.final:.6f}",
                "reasoning": synthesize(item, rec, rank),
                "title": rec.current_title,
                "credibility": f"{item.c:.3f}",
                "availability": f"{item.a:.3f}",
            }
        )
    return rows


def rows_to_csv(rows: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in writer.fieldnames})
    return out.getvalue()


def main() -> None:
    st.set_page_config(page_title="VERDICT Redrob Sandbox", layout="wide")
    st.title("VERDICT Redrob Sandbox")
    st.caption("Small-sample demo for <=100 candidates. Official full-pool submission uses rank.py.")

    rubric = load_rubric()
    uploaded = st.file_uploader("Upload candidates JSONL or JSON array", type=["jsonl", "json"])
    use_sample = st.button("Load bundled sample candidates", use_container_width=False)

    candidates: list[dict] = []
    if uploaded is not None:
        try:
            candidates = _load_uploaded_candidates(uploaded.getvalue(), uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
            return
    elif use_sample:
        candidates = _load_sample_candidates()

    if not candidates:
        st.info("Upload a small candidate file or load the bundled sample.")
        return
    if len(candidates) > MAX_SAMPLE:
        st.error(f"Sandbox limit is {MAX_SAMPLE} candidates; uploaded {len(candidates)}.")
        return

    with st.spinner("Ranking sample..."):
        rows = rank_sample(candidates, rubric)
    if not rows:
        st.warning("No uploaded candidates passed the hard gates.")
        return

    st.metric("Candidates ranked", len(rows))
    st.dataframe(rows, hide_index=True, use_container_width=True)
    csv_text = rows_to_csv(rows)
    st.download_button(
        "Download ranked CSV",
        data=csv_text,
        file_name="sandbox_ranked_candidates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
