"""L0 Feel Compiler: free-text JD -> draft rubric program (Gemini API, OFFLINE only).

The committed artifacts/rubric_program.json is the human-reviewed version of
this output. Re-run this script only to regenerate drafts after a JD change;
it never runs inside the rank step (no-network rule).

Requires: pip install google-genai, and GEMINI_API_KEY in the environment.

Usage:
  python -m verdict.feel_compiler --jd docs/job_description.md --out artifacts/generated/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_SYSTEM = """You are a recruiting-intelligence compiler. You read a job description
written as free text ("the feel") and compile it into an executable rubric program.

Output STRICT JSON with this exact shape:
{
  "gates": {...hard binary qualification checks...},
  "crisp_rules": {rule_id: {"weight": float, "explain": str, ...params}},
  "fuzzy_predicates": {pred_id: {"weight": float, "positives": [5 paraphrase
      sentences describing concrete EXPERIENCE EVIDENCE, written like career-history
      sentences], "explain": str}},
  "dampeners": {id: {"factor": float 0-1, "explain": str}},
  "ideal_profiles": [5 short career narratives of the ideal candidate],
  "anti_profiles": [4 short career narratives of explicit non-fits]
}

Rules:
- Weights across crisp_rules + fuzzy_predicates must sum to 1.0.
- Dampeners encode the JD's explicit disqualifiers ("we will not move forward if...").
- Read between the lines: what the JD MEANS, not just what it says.
- Fuzzy predicate positives must describe evidence found in narrative career text,
  never keyword lists.
"""


def compile_jd(jd_text: str, model: str = "gemini-2.5-flash") -> str:
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=f"{_SYSTEM}\n\n--- JOB DESCRIPTION ---\n{jd_text}",
        config={"response_mime_type": "application/json", "temperature": 0.2},
    )
    return resp.text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True)
    ap.add_argument("--out", default="artifacts/generated")
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    draft = compile_jd(jd_text, args.model)
    (out_dir / "rubric_draft.json").write_text(draft, encoding="utf-8")
    print(f"draft written to {out_dir / 'rubric_draft.json'} - review before promoting to artifacts/rubric_program.json")


if __name__ == "__main__":
    main()
