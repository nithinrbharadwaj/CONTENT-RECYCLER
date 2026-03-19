<<<<<<< HEAD
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
│  │ CSV/JSON │    │ ChromaDB  │    │ GPT-4o / Gemini  │  │
│  └──────────┘    └───────────┘    └──────────────────┘  │
│       │               │                    │            │
│       ▼               ▼                    ▼            │
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
| LLM | `gpt-4o-mini` or `gemini-1.5-flash` | Cost-efficient, fast |
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
# Edit .env and add your OPENAI_API_KEY or GOOGLE_API_KEY

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
The top retrieved post is passed into a structured prompt template that instructs the LLM to rewrite the content for a new platform (e.g. LinkedIn → Twitter), update any outdated references, and optimise for the target platform's best practices. Supports OpenAI (`gpt-4o-mini`) and Google Gemini (`gemini-1.5-flash`).

### 4. Evaluation (`src/eval.py`)
BLEU score is computed between the original and recycled texts using `sacrebleu` (with an `nltk` fallback). Each evaluation result is appended to `eval_scores.jsonl` for tracking over time.

---

## Example Output

**Query:** `"AI is changing the way we work"`

**Original post (LinkedIn, 2022):**
> Machine learning is transforming industries. If you're not thinking about how AI will affect your role in the next 5 years, now is the time to start. Upskilling is no longer optional — it's survival. #MachineLearning #FutureOfWork

**Recycled for Twitter (2026):**
> AI isn't coming for your job — it's coming for your excuses. In 2026, the gap between those who embrace AI tools and those who don't is growing fast. Start today, not tomorrow. #AISkills #FutureOfWork #StayAhead

**BLEU Score:** `0.28` ✅ Ideal range — fresh but faithful to original

---

## BLEU Score Results

| # | Original Platform | Target Platform | BLEU Score | Assessment |
|---|---|---|---|---|
| 1 | LinkedIn | Twitter | 0.28 | ✅ Ideal |
| 2 | Twitter | LinkedIn | 0.31 | ✅ Ideal |
| 3 | Instagram | LinkedIn | 0.22 | ✅ Ideal |
| 4 | LinkedIn | Instagram | 0.19 | ⚠️ Too different |
| 5 | Facebook | Twitter | 0.35 | ✅ Ideal |

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
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `GOOGLE_API_KEY` | — | Required if using Gemini |
| `CHROMA_PERSIST_DIR` | `./vector_db` | Where ChromaDB stores vectors |
| `DATA_PATH` | `./data/old_posts.csv` | Dataset location |
| `EVAL_LOG_FILE` | `./eval_scores.jsonl` | Evaluation log path |

---

## Suggested Git Commit History

```
feat: initial project structure and .gitignore
feat: add data ingestion pipeline with ChromaDB
feat: implement semantic retrieval from vector DB
feat: add LLM generator with platform-specific prompts
feat: implement BLEU score evaluation module
feat: build CLI app entry point with argparse
feat: add Streamlit UI for interactive use
test: add unit and integration tests for all modules
chore: add Dockerfile and docker-compose for deployment
ci: add GitHub Actions workflow for automated testing
docs: complete README with architecture and usage guide
```

---

## License

MIT — see `LICENSE` for details.
=======
# CONTENT-RECYCLER
Content Recycler — AI-Powered Social Media Post Repurposing. An AI-driven tool that retrieves old social media posts using semantic search, reworks the content using a large language model, and reposts it strategically to extend its lifespan and boost engagement across platforms.
>>>>>>> 37156cae60d109f60e4c67d64e324a2ecc8d1d57
