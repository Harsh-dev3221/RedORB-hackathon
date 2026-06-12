"""Small-sample Redrob sandbox for Streamlit/HuggingFace Spaces.

The official Stage-3 command is still `rank.py`; this app is the hosted
small-sample demo for the submission portal's sandbox requirement. It accepts
<=100 candidates, embeds their evidence sentences on CPU at runtime, and ranks
them with the same claim ledger, rubric predicates, credibility, availability,
fusion, and reasoning code used by the local pipeline.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import orjson
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from verdict.availability import score_availability
from verdict.credibility import score_credibility
from verdict.data import iter_candidates
from verdict.embedder import embed_passages, embed_queries, get_model
from verdict.evidence import build_record
from verdict.fusion import Scored, finalize
from verdict.judgment import judge, predicate_scores
from verdict.reasoning import synthesize
from verdict.recall import passes_gates

MAX_SAMPLE = 100
DISPLAY_TOP_N = 25
EMBED_BATCH_SIZE = 8
EMBED_THREADS = 4


@st.cache_data(show_spinner=False)
def load_rubric() -> dict:
    return orjson.loads((ROOT / "artifacts" / "rubric_program.json").read_bytes())


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return get_model(threads=EMBED_THREADS, cuda=False)


@st.cache_data(show_spinner=False)
def _probe_texts(rubric: dict) -> tuple[list[str], dict[str, tuple[int, int]], list[str]]:
    positives: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for pid, cfg in rubric["fuzzy_predicates"].items():
        start = len(positives)
        positives.extend(cfg.get("positives", []))
        slices[pid] = (start, len(positives))
    negatives = list(rubric.get("predicate_negatives", []))
    return positives, slices, negatives


def build_probe_vectors(model, rubric: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    positives, slices, negatives = _probe_texts(rubric)
    pos_vecs = embed_queries(model, positives) if positives else np.zeros((0, 384), dtype=np.float32)
    neg_vecs = embed_queries(model, negatives) if negatives else np.zeros((0, 384), dtype=np.float32)
    pred_vecs = {pid: pos_vecs[start:end] for pid, (start, end) in slices.items()}
    return pred_vecs, neg_vecs


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
    sample = ROOT / "samples" / "top100_candidates.json"
    if not sample.exists():
        sample = ROOT / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
    if sample.exists():
        with NamedTemporaryFile("wb", suffix=".json", delete=False) as tmp:
            tmp.write(sample.read_bytes())
            tmp_path = tmp.name
        try:
            return list(iter_candidates(tmp_path))[:MAX_SAMPLE]
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return []


def rank_sample(candidates: list[dict], rubric: dict, model) -> list[dict]:
    records = [build_record(c) for c in candidates]
    pred_vecs, neg_vecs = build_probe_vectors(model, rubric)
    scored: list[Scored] = []
    for idx, rec in enumerate(records):
        if not passes_gates(rec, rubric["gates"]):
            continue
        sent_vecs = embed_passages(model, rec.sentences, batch_size=EMBED_BATCH_SIZE)
        preds = predicate_scores(sent_vecs, pred_vecs, neg_vecs, rubric["predicate_scoring"])
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
    st.caption("Working small-sample ranker for <=100 candidates. Official full-pool submission uses rank.py.")

    rubric = load_rubric()
    uploaded = st.file_uploader("Upload candidates JSONL or JSON array", type=["jsonl", "json"])

    source = "bundled top-100 sample"
    if uploaded is not None:
        try:
            candidates = _load_uploaded_candidates(uploaded.getvalue(), uploaded.name)
            source = f"uploaded file: {uploaded.name}"
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        candidates = _load_sample_candidates()

    if not candidates:
        st.error("Bundled sample is missing. Upload a small candidate file to run the sandbox.")
        return
    if len(candidates) > MAX_SAMPLE:
        st.error(f"Sandbox limit is {MAX_SAMPLE} candidates; uploaded {len(candidates)}.")
        return

    st.info(f"Ranking source: {source}. Loaded {len(candidates)} candidates.")
    with st.spinner("Loading CPU embedding model and running VERDICT ranking..."):
        model = load_embedding_model()
        rows = rank_sample(candidates, rubric, model)
    if not rows:
        st.warning("No uploaded candidates passed the hard gates.")
        return

    visible_rows = rows[:DISPLAY_TOP_N]
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates loaded", len(candidates))
    c2.metric("Candidates ranked", len(rows))
    c3.metric("Showing", f"Top {len(visible_rows)}")

    st.subheader(f"Top {len(visible_rows)} ranked candidates")
    st.dataframe(visible_rows, hide_index=True, use_container_width=True)
    csv_text = rows_to_csv(rows)
    st.download_button(
        "Download full ranked CSV",
        data=csv_text,
        file_name="sandbox_ranked_candidates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
