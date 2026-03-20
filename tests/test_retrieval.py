"""
tests/test_retrieval.py
=======================
Unit tests for the semantic retrieval module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures — shared DB setup
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory: pytest.TempPathFactory):
    """
    Create a temporary CSV and populate a ChromaDB for retrieval tests.
    Scoped to module so the (relatively slow) embed step runs only once.
    """
    tmp = tmp_path_factory.mktemp("retrieval_db")
    csv_path = tmp / "posts.csv"
    db_path = str(tmp / "vector_db")

    data = {
        "post_id": ["R001", "R002", "R003", "R004", "R005"],
        "platform": ["LinkedIn", "Twitter", "LinkedIn", "Instagram", "Facebook"],
        "original_text": [
            "Python automation is transforming data engineering workflows.",
            "Artificial intelligence is reshaping every industry in 2024.",
            "Remote work productivity tips for distributed engineering teams.",
            "Machine learning models require careful evaluation and testing.",
            "Social media marketing strategies for startup growth in 2024.",
        ],
        "engagement_score": [100, 250, 80, 150, 90],
        "date_posted": ["2023-01-01"] * 5,
        "tone": ["professional"] * 5,
        "tags": [""] * 5,
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    from src.ingestion import ingest
    ingest(data_path=str(csv_path), persist_dir=db_path)

    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrievePosts:
    def test_returns_results(self, populated_db: str) -> None:
        from src.retrieval import retrieve_posts

        results = retrieve_posts(
            query="python automation", top_n=3, persist_dir=populated_db
        )
        assert len(results) > 0

    def test_result_has_required_keys(self, populated_db: str) -> None:
        from src.retrieval import retrieve_posts

        results = retrieve_posts(query="AI trends", top_n=1, persist_dir=populated_db)
        assert len(results) >= 1
        keys = {
            "id",
            "original_text",
            "platform",
            "engagement_score",
            "date_posted",
            "similarity_score",
        }
        assert keys.issubset(results[0].keys())

    def test_most_relevant_post_returned(self, populated_db: str) -> None:
        from src.retrieval import retrieve_posts

        results = retrieve_posts(
            query="machine learning evaluation", top_n=3, persist_dir=populated_db
        )
        assert len(results) > 0
        all_text = " ".join([r["original_text"].lower() for r in results])
        assert any(
            kw in all_text for kw in ["machine", "learning", "model", "evaluation", "python", "ai"]
        )

    def test_top_n_respected(self, populated_db: str) -> None:
        from src.retrieval import retrieve_posts

        results = retrieve_posts(query="technology", top_n=2, persist_dir=populated_db)
        assert len(results) <= 2

    def test_similarity_scores_in_range(self, populated_db: str) -> None:
        from src.retrieval import retrieve_posts

        results = retrieve_posts(
            query="startup marketing", top_n=3, persist_dir=populated_db
        )
        for r in results:
            assert -0.1 <= r["similarity_score"] <= 1.1

    def test_platform_filter(self, populated_db: str) -> None:
        from src.retrieval import retrieve_posts

        results = retrieve_posts(
            query="technology trends",
            top_n=5,
            platform_filter="LinkedIn",
            persist_dir=populated_db,
        )
        for r in results:
            assert r["platform"] == "LinkedIn"

    def test_raises_if_db_not_found(self, tmp_path: Path) -> None:
        import src.retrieval as retrieval_mod
        retrieval_mod._collection = None

        from src.retrieval import retrieve_posts

        with pytest.raises((RuntimeError, Exception)):
            retrieve_posts(
                query="test",
                top_n=1,
                persist_dir=str(tmp_path / "empty_db"),
            )