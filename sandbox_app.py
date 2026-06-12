"""Small-sample Redrob sandbox for Streamlit/HuggingFace Spaces.

The official Stage-3 command is still `rank.py`; this app is the hosted
small-sample demo for the submission portal's sandbox requirement. It accepts
<=100 candidates, embeds their evidence sentences on CPU at runtime, and ranks
them with the same claim ledger, rubric predicates, credibility, availability,
fusion, and reasoning code used by the local pipeline.
"""

from __future__ import annotations

import csv
import html
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


def _jd_summary(rubric: dict) -> dict[str, str]:
    meta = rubric.get("meta", {})
    role = meta.get("role", "Senior AI Engineer - Founding Team, Redrob AI")
    core = rubric.get("crisp_rules", {}).get("core_skill_coverage", {})
    core_skills = ", ".join(core.get("core_categories", []))
    return {
        "role": role,
        "core_skills": core_skills,
        "mode": "CPU-only sample ranking with runtime embeddings",
    }


@st.cache_data(show_spinner=False)
def load_rubric() -> dict:
    return orjson.loads((ROOT / "artifacts" / "rubric_program.json").read_bytes())


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        return get_model(threads=EMBED_THREADS, cuda=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastEmbed is not installed in this Streamlit environment. "
            "Redeploy with Python 3.11 selected in Streamlit Advanced settings "
            "and requirements.txt installed."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Embedding model failed to load. This sandbox requires Python 3.11 "
            "because FastEmbed/ONNX Runtime is not reliable on Streamlit's "
            "default Python 3.14 runtime."
        ) from exc


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


def render_ranked_results(rows: list[dict]) -> None:
    for row in rows:
        rank = html.escape(str(row["rank"]))
        title = html.escape(str(row["title"]))
        candidate_id = html.escape(str(row["candidate_id"]))
        reasoning = html.escape(str(row["reasoning"]))
        score = html.escape(str(row["score"]))
        credibility = html.escape(str(row["credibility"]))
        availability = html.escape(str(row["availability"]))
        st.markdown(
            f"""
<div class="candidate-row">
  <div class="candidate-rank">#{rank}</div>
  <div class="candidate-main">
    <div class="candidate-title">{title}</div>
    <div class="candidate-id">{candidate_id}</div>
    <div class="candidate-reason">{reasoning}</div>
  </div>
  <div class="candidate-scores">
    <div><span>Score</span><strong>{score}</strong></div>
    <div><span>Credibility</span><strong>{credibility}</strong></div>
    <div><span>Availability</span><strong>{availability}</strong></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="VERDICT Redrob Sandbox", layout="wide")
    rubric = load_rubric()
    jd = _jd_summary(rubric)

    st.title("VERDICT Redrob Candidate Ranking Sandbox")
    st.caption("Hosted small-sample demo. The full 100K submission is reproduced with rank.py.")
    st.markdown(
        """
<style>
.candidate-row {
  display: grid;
  grid-template-columns: 72px minmax(360px, 1fr) 280px;
  gap: 18px;
  align-items: start;
  padding: 16px 18px;
  margin: 10px 0;
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 8px;
  background: rgba(15, 23, 42, 0.28);
}
.candidate-rank {
  font-size: 22px;
  font-weight: 700;
  line-height: 1;
  color: #e5e7eb;
}
.candidate-title {
  font-size: 17px;
  font-weight: 700;
  color: #f8fafc;
}
.candidate-id {
  margin-top: 2px;
  font-size: 13px;
  color: #94a3b8;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.candidate-reason {
  margin-top: 10px;
  color: #e2e8f0;
  line-height: 1.45;
  white-space: normal;
  overflow-wrap: anywhere;
}
.candidate-scores {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.candidate-scores div {
  padding: 8px 10px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 6px;
  background: rgba(2, 6, 23, 0.32);
}
.candidate-scores span {
  display: block;
  font-size: 11px;
  color: #94a3b8;
}
.candidate-scores strong {
  display: block;
  margin-top: 2px;
  font-size: 14px;
  color: #f8fafc;
}
@media (max-width: 900px) {
  .candidate-row {
    grid-template-columns: 52px 1fr;
  }
  .candidate-scores {
    grid-column: 2;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Job Description Target")
    jd_cols = st.columns([1.4, 1, 1])
    jd_cols[0].metric("Hiring role", jd["role"])
    jd_cols[1].metric("Sample limit", f"{MAX_SAMPLE} candidates")
    jd_cols[2].metric("Visible output", f"Top {DISPLAY_TOP_N}")
    st.markdown(
        "This sandbox ranks candidates for the released Redrob JD: a hands-on "
        "founding-team Senior AI Engineer focused on production search, ranking, "
        "recommendation, retrieval, embeddings, NLP/IR, and real user impact."
    )
    st.caption(f"Core rubric categories: {jd['core_skills']}")

    with st.expander("What runs in this sandbox", expanded=False):
        st.write(
            "For the bundled or uploaded sample, the app builds the candidate claim ledger, "
            "embeds evidence sentences on CPU, scores rubric predicates, applies credibility "
            "and behavioral availability signals, then fuses the ranking and writes grounded reasoning."
        )
        st.write(
            "The production submission uses precomputed vectors for the full 100K pool so the "
            "official rank step remains under the five-minute CPU-only budget."
        )

    st.divider()
    st.subheader("Candidate Sample")
    uploaded = st.file_uploader(
        "Upload candidates JSONL or JSON array",
        type=["jsonl", "json"],
        help="Upload up to 100 candidate records. If nothing is uploaded, the bundled top-100 sample is ranked automatically.",
    )

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
        try:
            model = load_embedding_model()
            rows = rank_sample(candidates, rubric, model)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    if not rows:
        st.warning("No uploaded candidates passed the hard gates.")
        return

    visible_rows = rows[:DISPLAY_TOP_N]
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates loaded", len(candidates))
    c2.metric("Candidates ranked", len(rows))
    c3.metric("Showing", f"Top {len(visible_rows)}")

    st.subheader(f"Top {len(visible_rows)} ranked candidates")
    render_ranked_results(visible_rows)
    csv_text = rows_to_csv(rows)
    st.download_button(
        "Download full ranked CSV",
        data=csv_text,
        file_name="sandbox_ranked_candidates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
