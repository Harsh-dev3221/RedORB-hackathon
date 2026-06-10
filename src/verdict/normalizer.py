"""Entity Normalizer: titles, skills, companies, industries, locations.

LinkedIn's Learning-to-Retrieve lesson: standardized entities are load-bearing.
Everything downstream (ABM recall, crisp rules, credibility checks) reads
normalized entities, never raw strings.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- skills ----

# canonical skill -> category. Categories drive ABM recall + corroboration.
SKILL_CATEGORIES: dict[str, str] = {
    # core JD categories
    "embeddings": "embeddings", "sentence-transformers": "embeddings", "bge": "embeddings",
    "e5": "embeddings", "openai-embeddings": "embeddings", "word2vec": "embeddings",
    "vector-db": "vector_db", "faiss": "vector_db", "pinecone": "vector_db",
    "weaviate": "vector_db", "qdrant": "vector_db", "milvus": "vector_db",
    "opensearch": "search", "elasticsearch": "search", "solr": "search", "lucene": "search",
    "bm25": "search", "semantic-search": "search", "hybrid-search": "search",
    "ranking": "ranking", "learning-to-rank": "ranking", "recommendation-systems": "ranking",
    "ndcg": "ranking", "ab-testing": "ranking", "information-retrieval": "search",
    "nlp": "nlp", "ner": "nlp", "text-classification": "nlp", "spacy": "nlp", "nltk": "nlp",
    "transformers": "llm", "llm": "llm", "rag": "llm", "fine-tuning": "llm", "lora": "llm",
    "qlora": "llm", "peft": "llm", "prompt-engineering": "llm", "langchain": "llm_framework",
    "llamaindex": "llm_framework", "openai-api": "llm_framework", "agents": "llm_framework",
    "machine-learning": "ml_core", "deep-learning": "ml_core", "pytorch": "ml_core",
    "tensorflow": "ml_core", "scikit-learn": "ml_core", "xgboost": "ml_core",
    "lightgbm": "ml_core", "keras": "ml_core", "statistical-modeling": "ml_core",
    "feature-engineering": "ml_core", "model-deployment": "mlops", "mlops": "mlops",
    "mlflow": "mlops", "kubeflow": "mlops", "weights-and-biases": "mlops",
    "bentoml": "mlops", "triton": "mlops", "onnx": "mlops", "model-monitoring": "mlops",
    # adjacent
    "python": "backend", "java": "backend", "golang": "backend", "scala": "backend",
    "sql": "data_eng", "spark": "data_eng", "airflow": "data_eng", "kafka": "data_eng",
    "dbt": "data_eng", "snowflake": "data_eng", "databricks": "data_eng",
    "apache-beam": "data_eng", "etl": "data_eng", "data-pipelines": "data_eng",
    "hadoop": "data_eng", "flink": "data_eng",
    "aws": "cloud", "gcp": "cloud", "azure": "cloud", "docker": "devops",
    "kubernetes": "devops", "terraform": "devops", "ci-cd": "devops", "linux": "devops",
    "fastapi": "backend", "flask": "backend", "django": "backend", "rest-api": "backend",
    "grpc": "backend", "microservices": "backend", "redis": "backend",
    "postgresql": "backend", "mongodb": "backend", "system-design": "backend",
    "distributed-systems": "backend",
    # out-of-domain (used by disqualifier + stuffing checks)
    "computer-vision": "vision", "image-classification": "vision", "object-detection": "vision",
    "opencv": "vision", "gans": "vision", "diffusion-models": "vision", "yolo": "vision",
    "speech-recognition": "speech", "tts": "speech", "asr": "speech", "audio": "speech",
    "robotics": "robotics", "ros": "robotics", "slam": "robotics",
    "react": "frontend", "angular": "frontend", "vue": "frontend", "javascript": "frontend",
    "typescript": "frontend", "html": "frontend", "css": "frontend", "tailwind": "frontend",
    "nextjs": "frontend", "nodejs": "frontend",
    "android": "mobile", "ios": "mobile", "flutter": "mobile", "react-native": "mobile",
    "photoshop": "design", "figma": "design", "illustrator": "design", "ui-ux": "design",
    "canva": "design", "graphic-design": "design",
    "seo": "marketing", "content-writing": "marketing", "social-media": "marketing",
    "digital-marketing": "marketing", "copywriting": "marketing", "google-ads": "marketing",
    "recruitment": "hr", "talent-acquisition": "hr", "hr-operations": "hr", "payroll": "hr",
    "excel": "analytics", "power-bi": "analytics", "tableau": "analytics",
    "data-analysis": "analytics", "business-analysis": "analytics", "looker": "analytics",
    "selenium": "qa", "manual-testing": "qa", "automation-testing": "qa", "jmeter": "qa",
    "sales": "sales", "crm": "sales", "salesforce": "sales", "accounting": "finance",
    "project-management": "pm", "agile": "pm", "scrum": "pm", "jira": "pm",
}

# raw alias -> canonical skill (lowercased keys; applied after light cleanup)
SKILL_ALIASES: dict[str, str] = {
    "sentence transformers": "sentence-transformers", "sbert": "sentence-transformers",
    "openai embeddings": "openai-embeddings", "text embeddings": "embeddings",
    "embedding models": "embeddings", "vector embeddings": "embeddings",
    "vector databases": "vector-db", "vector database": "vector-db", "vector db": "vector-db",
    "vector search": "vector-db", "ann search": "vector-db", "hnsw": "vector-db",
    "elastic search": "elasticsearch", "elastic": "elasticsearch",
    "open search": "opensearch", "apache solr": "solr",
    "learning to rank": "learning-to-rank", "ltr": "learning-to-rank",
    "recsys": "recommendation-systems", "recommender systems": "recommendation-systems",
    "recommendation engines": "recommendation-systems", "recommendations": "recommendation-systems",
    "search relevance": "ranking", "relevance tuning": "ranking", "search ranking": "ranking",
    "information retrieval": "information-retrieval", "ir": "information-retrieval",
    "a/b testing": "ab-testing", "ab testing": "ab-testing", "experimentation": "ab-testing",
    "natural language processing": "nlp", "text mining": "nlp",
    "named entity recognition": "ner",
    "large language models": "llm", "llms": "llm", "genai": "llm", "generative ai": "llm",
    "gen ai": "llm", "retrieval augmented generation": "rag", "retrieval-augmented generation": "rag",
    "fine tuning": "fine-tuning", "finetuning": "fine-tuning", "fine-tuning llms": "fine-tuning",
    "llm fine-tuning": "fine-tuning", "llm finetuning": "fine-tuning",
    "prompt engineering": "prompt-engineering", "prompting": "prompt-engineering",
    "llama index": "llamaindex", "llama-index": "llamaindex",
    "openai api": "openai-api", "chatgpt": "openai-api", "gpt-4": "openai-api",
    "ai agents": "agents", "agentic ai": "agents", "autogen": "agents", "crewai": "agents",
    "hugging face": "transformers", "huggingface": "transformers",
    "machine learning": "machine-learning", "ml": "machine-learning",
    "deep learning": "deep-learning", "dl": "deep-learning", "neural networks": "deep-learning",
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn",
    "statistical modeling": "statistical-modeling", "statistics": "statistical-modeling",
    "feature engineering": "feature-engineering",
    "model deployment": "model-deployment", "model serving": "model-deployment",
    "ml ops": "mlops", "weights & biases": "weights-and-biases", "wandb": "weights-and-biases",
    "w&b": "weights-and-biases", "ml flow": "mlflow",
    "apache spark": "spark", "pyspark": "spark", "spark streaming": "spark",
    "apache airflow": "airflow", "apache kafka": "kafka", "apache beam": "apache-beam",
    "apache flink": "flink", "data engineering": "data-pipelines",
    "data pipelines": "data-pipelines", "data warehousing": "etl",
    "google cloud": "gcp", "google cloud platform": "gcp", "amazon web services": "aws",
    "microsoft azure": "azure", "k8s": "kubernetes", "ci/cd": "ci-cd",
    "rest apis": "rest-api", "rest": "rest-api", "api development": "rest-api",
    "postgres": "postgresql", "mongo": "mongodb", "node.js": "nodejs", "node js": "nodejs",
    "next.js": "nextjs", "react.js": "react", "reactjs": "react", "vue.js": "vue",
    "js": "javascript", "ts": "typescript",
    "computer vision": "computer-vision", "cv": "computer-vision",
    "image classification": "image-classification", "object detection": "object-detection",
    "diffusion": "diffusion-models", "stable diffusion": "diffusion-models",
    "speech recognition": "speech-recognition", "text to speech": "tts",
    "text-to-speech": "tts", "speech to text": "asr", "audio processing": "audio",
    "ui/ux": "ui-ux", "ux design": "ui-ux", "ui design": "ui-ux",
    "adobe photoshop": "photoshop", "adobe illustrator": "illustrator",
    "graphic design": "graphic-design",
    "content writing": "content-writing", "social media marketing": "social-media",
    "digital marketing": "digital-marketing",
    "talent acquisition": "talent-acquisition", "hr operations": "hr-operations",
    "recruiting": "recruitment", "hiring": "recruitment",
    "ms excel": "excel", "microsoft excel": "excel", "powerbi": "power-bi",
    "power bi": "power-bi", "data analysis": "data-analysis",
    "business analysis": "business-analysis",
    "manual testing": "manual-testing", "test automation": "automation-testing",
    "automation testing": "automation-testing",
    "project management": "project-management", "distributed systems": "distributed-systems",
    "system design": "system-design", "go": "golang",
}

_CLEAN_RE = re.compile(r"[^a-z0-9+#&/.\- ]+")


def normalize_skill(raw: str) -> tuple[str, str]:
    """Return (canonical_skill, category). Unknown skills keep cleaned name, category 'other'."""
    s = _CLEAN_RE.sub("", raw.strip().lower()).strip()
    s = SKILL_ALIASES.get(s, s)
    if s in SKILL_CATEGORIES:
        return s, SKILL_CATEGORIES[s]
    hy = s.replace(" ", "-")
    if hy in SKILL_CATEGORIES:
        return hy, SKILL_CATEGORIES[hy]
    return s, "other"


# ---------------------------------------------------------------- titles ----

# (pattern, family) - first match wins; order matters.
_TITLE_FAMILIES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"machine learning|ml engineer|\bml\b|ai engineer|artificial intelligence", re.I), "ml_engineer"),
    (re.compile(r"applied scientist", re.I), "applied_scientist"),
    (re.compile(r"\bnlp\b|natural language", re.I), "nlp_engineer"),
    (re.compile(r"search|relevance|ranking|recommendation|recsys|information retrieval", re.I), "search_engineer"),
    (re.compile(r"data scientist|decision scientist", re.I), "data_scientist"),
    (re.compile(r"research (scientist|engineer|fellow)|researcher", re.I), "research"),
    (re.compile(r"data engineer|analytics engineer|big data|etl", re.I), "data_engineer"),
    (re.compile(r"mlops|ml ops|ml platform|ml infra", re.I), "mlops_engineer"),
    (re.compile(r"devops|sre|site reliability|platform engineer|cloud engineer|infrastructure", re.I), "devops"),
    (re.compile(r"backend|back-end|back end|server|api engineer", re.I), "backend"),
    (re.compile(r"full ?stack", re.I), "fullstack"),
    (re.compile(r"frontend|front-end|front end|ui engineer|react|web developer", re.I), "frontend"),
    (re.compile(r"android|ios|mobile", re.I), "mobile"),
    (re.compile(r"qa|sdet|test engineer|quality", re.I), "qa"),
    (re.compile(r"civil|mechanical|electrical|electronics|automobile|construction|site engineer", re.I), "other"),
    (re.compile(r"data analyst|business analyst|analyst", re.I), "analyst"),
    (re.compile(r"product manager|product owner", re.I), "product"),
    (re.compile(r"project manager|program manager|scrum master|delivery", re.I), "pm"),
    (re.compile(r"designer|design|ux|ui\b", re.I), "design"),
    (re.compile(r"marketing|seo|content|social media|brand|growth", re.I), "marketing"),
    (re.compile(r"sales|business development|account", re.I), "sales"),
    (re.compile(r"\bhr\b|human resources|recruiter|talent", re.I), "hr"),
    (re.compile(r"finance|accountant|accounting|audit", re.I), "finance"),
    (re.compile(r"consultant|consulting", re.I), "consultant"),
    (re.compile(r"support|customer success|operations|ops\b", re.I), "ops"),
    (re.compile(r"software (engineer|developer)|sde|programmer|developer|engineer", re.I), "swe"),
]

# IC seniority ladder for trajectory slope; manager-track tracked separately.
_SENIORITY: list[tuple[re.Pattern, int, bool]] = [  # (pattern, level, is_manager_track)
    (re.compile(r"intern|trainee|apprentice", re.I), 0, False),
    (re.compile(r"\bjunior\b|\bjr\b|associate|graduate", re.I), 1, False),
    (re.compile(r"\bcto\b|\bceo\b|chief|founder|\bvp\b|vice president", re.I), 7, True),
    (re.compile(r"director|head of", re.I), 6, True),
    (re.compile(r"principal|distinguished", re.I), 5, False),
    (re.compile(r"staff|architect", re.I), 4, False),
    (re.compile(r"engineering manager|manager", re.I), 5, True),
    (re.compile(r"\blead\b|tech lead|team lead", re.I), 4, False),
    (re.compile(r"\bsenior\b|\bsr\b", re.I), 3, False),
]

# Families considered "writes production ML/IR-adjacent code" for ABM
TECH_FAMILIES = {
    "ml_engineer", "applied_scientist", "nlp_engineer", "search_engineer",
    "data_scientist", "data_engineer", "mlops_engineer", "backend", "fullstack",
    "swe", "devops",
}
CORE_FAMILIES = {"ml_engineer", "applied_scientist", "nlp_engineer", "search_engineer", "data_scientist"}


def normalize_title(raw: str) -> tuple[str, int, bool]:
    """Return (family, seniority_level 0-7, is_manager_track). Default level 2 (mid)."""
    family = "other"
    for pat, fam in _TITLE_FAMILIES:
        if pat.search(raw):
            family = fam
            break
    level, mgr = 2, False
    for pat, lvl, m in _SENIORITY:
        if pat.search(raw):
            level, mgr = lvl, m
            break
    return family, level, mgr


# ------------------------------------------------------------- companies ----

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "tata consultancy", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl", "hcl technologies", "hcltech",
    "tech mahindra", "mindtree", "ltimindtree", "l&t infotech", "lti", "mphasis",
    "deloitte", "ernst & young", "ey", "kpmg", "pwc", "pricewaterhousecoopers",
    "dxc", "dxc technology", "genpact", "virtusa", "hexaware", "zensar",
    "birlasoft", "coforge", "niit", "cgi", "atos", "ust global", "ust",
    "mu sigma", "infogain", "sonata software", "cyient", "kpit", "globallogic",
    "epam", "thoughtworks",
}
SERVICES_INDUSTRIES = {"it services", "consulting", "outsourcing", "bpo", "staffing"}
RESEARCH_MARKERS = re.compile(r"university|institute|academy|\biit\b|\biisc\b|research lab|labs?$|college|cnrs|max planck", re.I)
RESEARCH_INDUSTRIES = {"research", "academia", "higher education", "education"}
PRODUCT_INDUSTRIES = {
    "saas", "software product", "internet", "e-commerce", "ecommerce", "fintech",
    "marketplace", "gaming", "edtech", "healthtech", "social media", "streaming",
    "consumer tech", "b2b software", "enterprise software", "cybersecurity",
    "adtech", "logistics tech", "foodtech", "traveltech", "proptech", "ai",
    "artificial intelligence", "technology", "software",
}

# founding years for recognizable companies (temporal honeypot check)
COMPANY_FOUNDED: dict[str, int] = {
    "openai": 2015, "anthropic": 2021, "mistral": 2023, "mistral ai": 2023,
    "xai": 2023, "perplexity": 2022, "perplexity ai": 2022, "hugging face": 2016,
    "huggingface": 2016, "stability ai": 2020, "cohere": 2019, "deepmind": 2010,
    "databricks": 2013, "snowflake": 2012, "scale ai": 2016, "together ai": 2022,
    "zepto": 2021, "cred": 2018, "meesho": 2015, "razorpay": 2014, "groww": 2016,
    "sharechat": 2015, "jio": 2016, "reliance jio": 2016, "phonepe": 2015,
    "swiggy": 2014, "zomato": 2008, "unacademy": 2015, "postman": 2014,
    "hasura": 2017, "zerodha": 2010, "upstox": 2009, "slice": 2016,
    "jupiter": 2019, "fi money": 2021, "krutrim": 2023, "sarvam": 2023,
    "sarvam ai": 2023, "ola krutrim": 2023,
}


def classify_company(company: str, industry: str) -> str:
    """Return 'services' | 'research' | 'product' | 'unknown'."""
    c = company.strip().lower()
    ind = (industry or "").strip().lower()
    if c in CONSULTING_COMPANIES or any(c.startswith(x + " ") for x in ("tcs", "infosys", "wipro")):
        return "services"
    if ind in SERVICES_INDUSTRIES:
        return "services"
    if ind in RESEARCH_INDUSTRIES or RESEARCH_MARKERS.search(company):
        return "research"
    if ind in PRODUCT_INDUSTRIES:
        return "product"
    if ind:  # named industry that isn't services/research -> operating company
        return "product"
    return "unknown"


# -------------------------------------------------------------- locations ----

_CITY_BUCKETS = {
    "pune": "preferred", "noida": "preferred", "greater noida": "preferred",
    "delhi": "tier1", "new delhi": "tier1", "gurgaon": "tier1", "gurugram": "tier1",
    "delhi ncr": "tier1", "ncr": "tier1", "faridabad": "tier1", "ghaziabad": "tier1",
    "mumbai": "tier1", "navi mumbai": "tier1", "thane": "tier1",
    "hyderabad": "tier1", "bangalore": "tier1", "bengaluru": "tier1",
    "chennai": "tier1", "kolkata": "tier1", "ahmedabad": "tier1",
}


def normalize_location(location: str, country: str) -> str:
    """Return 'preferred' | 'tier1' | 'india_other' | 'abroad'."""
    if (country or "").strip().lower() not in {"india", "in"}:
        return "abroad"
    loc = (location or "").lower()
    for city, bucket in _CITY_BUCKETS.items():
        if city in loc:
            return bucket
    return "india_other"
