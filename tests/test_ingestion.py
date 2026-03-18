"""
tests/test_ingestion.py
=======================
Unit tests for the data ingestion pipeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_csv(tmp_path: Path) -> str:
    """Create a minimal CSV dataset for testing."""
    data = {
        "post_id": ["T001", "T002", "T003"],
        "platform": ["LinkedIn", "Twitter", "Instagram"],
        "original_text": [
            "Machine learning is transforming industries. Automation is key.",
            "Python is the best language for data science workflows.",
            "Remote work has changed how we think about productivity forever.",
        ],
        "engagement_score": [120, 85, 200],
        "date_posted": ["2023-01-10", "2023-03-15", "2022-11-20"],
        "tone": ["professional", "informative", "casual"],
        "tags": ["#ML #AI", "#Python #DataScience", "#RemoteWork"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_posts.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture()
def vector_db_dir(tmp_path: Path) -> str:
    return str(tmp_path / "vector_db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadPosts:
    def test_loads_correctly(self, sample_csv: str) -> None:
        from src.ingestion import load_posts

        df = load_posts(sample_csv)
        assert len(df) == 3
        assert "original_text" in df.columns
        assert "platform" in df.columns

    def test_raises_on_missing_file(self) -> None:
        from src.ingestion import load_posts

        with pytest.raises(FileNotFoundError):
            load_posts("/nonexistent/path/data.csv")

    def test_raises_on_missing_columns(self, tmp_path: Path) -> None:
        from src.ingestion import load_posts

        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_posts(str(bad_csv))

    def test_drops_empty_texts(self, tmp_path: Path) -> None:
        from src.ingestion import load_posts

        data = {
            "post_id": ["X1", "X2"],
            "platform": ["Twitter", "LinkedIn"],
            "original_text": ["   ", "Valid post content here"],
            "engagement_score": [0, 50],
            "date_posted": ["2023-01-01", "2023-01-02"],
        }
        csv_path = tmp_path / "partial.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)
        df = load_posts(str(csv_path))
        assert len(df) == 1


class TestChunkText:
    def test_short_text_unchanged(self) -> None:
        from src.ingestion import chunk_text

        text = "Short post."
        chunks = chunk_text(text, max_chars=500)
        assert chunks == [text]

    def test_long_text_is_split(self) -> None:
        from src.ingestion import chunk_text

        text = ". ".join(["Sentence number " + str(i) for i in range(100)])
        chunks = chunk_text(text, max_chars=200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 210  # small tolerance for edge splitting

    def test_returns_list(self) -> None:
        from src.ingestion import chunk_text

        assert isinstance(chunk_text("hello world"), list)
        assert len(chunk_text("hello world")) >= 1


class TestIngest:
    def test_db_gets_populated(self, sample_csv: str, vector_db_dir: str) -> None:
        from src.ingestion import ingest

        count = ingest(data_path=sample_csv, persist_dir=vector_db_dir)
        assert count > 0

    def test_reset_clears_and_repopulates(self, sample_csv: str, vector_db_dir: str) -> None:
        from src.ingestion import ingest

        count1 = ingest(data_path=sample_csv, persist_dir=vector_db_dir)
        count2 = ingest(data_path=sample_csv, persist_dir=vector_db_dir, reset=True)
        # After reset + re-ingest, should have same number of chunks
        assert count1 == count2

    def test_double_ingest_no_duplicate(self, sample_csv: str, vector_db_dir: str) -> None:
        from src.ingestion import ingest

        count1 = ingest(data_path=sample_csv, persist_dir=vector_db_dir)
        count2 = ingest(data_path=sample_csv, persist_dir=vector_db_dir)
        # ChromaDB upsert should be idempotent
        assert count1 == count2
