"""Smoke-test the Gemini raw-resume parser (1 API call). Reads key from .env."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from verdict.resume_parser import parse_resume_text

RESUME = """
Priya Sharma
Senior Machine Learning Engineer - Bengaluru, India
priya@example.com

Experience:
Flipmart (e-commerce, ~2000 employees) - Senior ML Engineer, Jan 2022 - present
Rebuilt product search relevance: hybrid lexical + embedding retrieval with a
cross-encoder reranker, serving 5M daily queries. Own the NDCG/MRR offline harness
and A/B experimentation for all ranking launches.

DataWeave (B2B SaaS, ~300 employees) - ML Engineer, Jun 2018 - Dec 2021
Built text-classification and entity-extraction pipelines for catalog matching.

Education: B.Tech Computer Science, NIT Trichy, 2014-2018
Skills: Python (expert), PyTorch, Elasticsearch, FAISS, sentence-transformers, SQL
"""

cand = parse_resume_text(RESUME, model=sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash")
import json

print(json.dumps(cand, indent=2)[:2200])
