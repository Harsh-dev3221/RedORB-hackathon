"""Raw resume text -> candidate-schema JSON via Gemini (ingest-time only, never rank-time).

Requires GEMINI_API_KEY in the environment and `pip install google-genai`.
"""

from __future__ import annotations

import hashlib
import os

_PROMPT = """You convert a raw resume into JSON matching this exact schema (the Redrob
candidate profile schema). Output ONLY the JSON object.

{
  "candidate_id": "<leave as UPLOAD_PLACEHOLDER>",
  "profile": {"anonymized_name": str, "headline": str, "summary": str, "location": str,
              "country": str, "years_of_experience": number, "current_title": str,
              "current_company": str, "current_company_size": "1-10|11-50|51-200|201-500|501-1000|1001-5000|5001-10000|10001+",
              "current_industry": str},
  "career_history": [{"company": str, "title": str, "start_date": "YYYY-MM-DD",
                      "end_date": "YYYY-MM-DD or null", "duration_months": int,
                      "is_current": bool, "industry": str, "company_size": str,
                      "description": str}],
  "education": [{"institution": str, "degree": str, "field_of_study": str,
                 "start_year": int, "end_year": int, "grade": str|null, "tier": "unknown"}],
  "skills": [{"name": str, "proficiency": "beginner|intermediate|advanced|expert",
              "endorsements": 0, "duration_months": int}],
  "certifications": [], "languages": []
}

Rules:
- Copy career descriptions as faithful prose; do NOT invent achievements, employers,
  numbers, or skills that are not in the resume.
- duration_months computed from the dates; estimate conservative proficiency from the text.
- Omit redrob_signals entirely (platform signals do not exist for an upload).

RESUME:
"""


def parse_resume_text(text: str, model: str = "gemini-2.5-flash") -> dict:
    from google import genai
    import orjson

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=_PROMPT + text,
        config={"response_mime_type": "application/json", "temperature": 0.0},
    )
    cand = orjson.loads(resp.text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:7].upper()
    cand["candidate_id"] = f"UPL_{digest}"
    cand.setdefault("redrob_signals", {})
    return cand
