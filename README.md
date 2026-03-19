# Content Recycler 🔁

> AI-powered tool to revive and repurpose old social media posts using RAG (Retrieval-Augmented Generation).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CONTENT RECYCLER                     │
│                                                         │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │  DATA    │───▶│  VECTOR   │───▶│   LLM GENERATOR  │  │
│  │  LAYER   │    │  LAYER    │    │   (Rework Post)  │  │
│  │ CSV/JSON │    │ ChromaDB  │    │ Groq / OpenAI /  │  │
│  └──────────┘    └───────────┘    │ Gemini           │  │
│       │               │           └──────────────────┘  │
│       ▼               ▼                    │            │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │  Pandas  │    │Embeddings │    │  BLEU EVALUATOR  │  │
│  │DataFrame │    │(MiniLM-L6)│    │  (eval.py)       │  │
│  │          │    │           │    │                  │  │
│  └──────────┘    └───────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │     DEPLOYMENT        │
              │  Docker + Compose     │
              │  GitHub Actions CI    │
              └───────────────────────┘
```

**Data Flow:**
```
old_posts.csv
    → ingestion.py   (chunk + embed + store in ChromaDB)
    → retrieval.py   (semantic search by keyword/topic)
    → generator.py   (LLM rewrites post for new platform)
    → eval.py        (BLEU score: original vs recycled)
    → app.py         (CLI or Streamlit UI output)
```

---

## Project Structure

```
content-recycler/
│
├── data/
│   └── old_posts.csv            # Historical posts dataset (260 posts)
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py             # Load CSV → embed → upsert to ChromaDB
│   ├── retrieval.py             # Semantic search from ChromaDB
│   ├── generator.py             # Prompt template + LLM call
│   └── eval.py                  # BLEU score calculation + logging
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py        # Unit tests: ingestion pipeline
│   ├── test_retrieval.py        # Unit tests: semantic retrieval
│   └── test_generator.py        # Integration + mock tests
│
├── vector_db/                   # Auto-created by ChromaDB (gitignored)
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD pipeline
│
├── app.py                       # Main entry point (CLI + Streamlit)
├── conftest.py                  # Pytest path configuration
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template (safe to commit)
├── .env                         # Actual keys (NEVER commit)
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Language | Python 3.10+ | Modern, async-ready |
| Vector DB | ChromaDB | Lightweight, persistent, local |
| Embeddings | `all-MiniLM-L6-v2` | Free, no API key needed |
| LLM | Groq (LLaMA 3.1) / OpenAI / Gemini | Cost-efficient, fast |
| Evaluation | `sacrebleu` + `nltk` | Industry-standard BLEU scoring |
| UI | Streamlit | Fast prototype UI |
| Containerization | Docker + Docker Compose | Reproducible deployment |
| CI/CD | GitHub Actions | Automated testing on push |

---

## Quick Start

### Option A — Local Setup

```bash
# 1. Clone and enter project
git clone <your-repo-url>
cd content-recycler

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY)

# 5. Index the dataset
python app.py --ingest

# 6. Recycle your first post
python app.py --recycle "AI productivity tips" --platform LinkedIn --evaluate

# 7. Launch the Streamlit UI
streamlit run app.py
```

### Option B — Docker

```bash
# 1. Configure keys
cp .env.example .env
# Edit .env

# 2. Build and launch
docker-compose up --build

# Open http://localhost:8501 in your browser

# 3. (Optional) Run ingestion as a one-shot service
docker-compose --profile ingest up ingest
```

---

## CLI Reference

```bash
# Index all posts from old_posts.csv into ChromaDB
python app.py --ingest

# Wipe and re-index
python app.py --ingest --reset

# Recycle a post matching a topic query
python app.py --recycle "remote work productivity" --platform LinkedIn

# Recycle + show BLEU evaluation score
python app.py --recycle "AI for beginners" --platform Twitter --evaluate

# Filter source posts by platform
python app.py --recycle "startup growth" --source-platform LinkedIn --platform Instagram --evaluate

# Show evaluation history stats
python app.py --stats

# Launch Streamlit UI
streamlit run app.py
```

---

## How It Works

### 1. Ingestion (`src/ingestion.py`)
The dataset (`data/old_posts.csv`) is loaded with Pandas. Long posts are chunked at sentence boundaries (max 512 tokens). Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` — a fast, free model that needs no API key. Embeddings are upserted into a persistent ChromaDB collection stored in `./vector_db/`.

### 2. Retrieval (`src/retrieval.py`)
A natural language query (e.g. `"remote work tips"`) is embedded using the same model. ChromaDB performs approximate nearest-neighbour search using cosine similarity. Results are returned with metadata and similarity scores. An optional platform filter narrows results to specific platforms.

### 3. Generation (`src/generator.py`)
The top retrieved post is passed into a structured prompt template that instructs the LLM to rewrite the content for a new platform (e.g. LinkedIn → Twitter), update any outdated references, and optimise for the target platform's best practices. Supports Groq (LLaMA 3.1), OpenAI (`gpt-4o-mini`), and Google Gemini (`gemini-2.0-flash-lite`).

### 4. Evaluation (`src/eval.py`)
BLEU score is computed between the original and recycled texts using `sacrebleu` (with an `nltk` fallback). Each evaluation result is appended to `eval_scores.jsonl` for tracking over time.

---

## Example Output

**Query:** `"AI productivity"`

**Original post (Facebook, 2023):**
> Attending a virtual conference on AI.

**Recycled for LinkedIn (2026):**
> Advancing my knowledge on the latest AI innovations and trends at a leading industry summit. Excited to connect with experts and learn how AI can drive business growth and transformation. #AI #DigitalTransformation #LeadershipDevelopment

**BLEU Score:** `0.013` — Creative recycling confirmed (low overlap = fresh content)

---

## BLEU Score Results

| # | Original Platform | Target Platform | BLEU Score | Assessment |
|---|---|---|---|---|
| 1 | Facebook | LinkedIn | 0.013 | Fresh rewrite |
| 2 | Instagram | Twitter | 0.036 | Fresh rewrite |
| 3 | Instagram | Instagram | 0.026 | Fresh rewrite |

**BLEU Score Interpretation:**

| Range | Meaning |
|---|---|
| 0.0 – 0.2 | ⚠️ Too different — core message may be lost |
| 0.2 – 0.5 | ✅ Ideal — fresh but faithful to original |
| 0.5 – 1.0 | 🔁 Too similar — not enough creative reworking |

---

## Dataset

The included dataset (`data/old_posts.csv`) was built from two sources:

1. **enriched_posts.json** — 60 LinkedIn-style posts with engagement scores, tone, and tags
2. **sentimentdataset.csv** — 200 multi-platform social posts (Twitter, Instagram, Facebook) with sentiment labels, timestamps, likes, and retweet counts

**Combined:** 260 posts across LinkedIn, Twitter, Instagram, and Facebook.

CSV columns: `post_id, platform, original_text, engagement_score, date_posted, tone, tags`

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only unit tests (skip integration)
pytest tests/test_ingestion.py tests/test_retrieval.py -v

# Run a specific test class
pytest tests/test_retrieval.py::TestRetrievePosts -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `openai`, `gemini`, or `groq` |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `GOOGLE_API_KEY` | — | Required if using Gemini |
| `GROQ_API_KEY` | — | Required if using Groq |
| `CHROMA_PERSIST_DIR` | `./vector_db` | Where ChromaDB stores vectors |
| `DATA_PATH` | `./data/old_posts.csv` | Dataset location |
| `EVAL_LOG_FILE` | `./eval_scores.jsonl` | Evaluation log path |

---

## License

MIT — see `LICENSE` for details.