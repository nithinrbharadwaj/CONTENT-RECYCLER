"""
tests/test_generator.py
=======================
Tests for the generator module and the full RAG pipeline.
Uses unittest.mock to avoid consuming real API credits during CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

MOCK_RECYCLED_TEXT = (
    "AI is fundamentally transforming how we approach daily work in 2026. "
    "Embrace automation, adopt a growth mindset, and lead with data-driven decisions. "
    "#AITrends #FutureOfWork #Innovation"
)


def _make_openai_mock(text: str = MOCK_RECYCLED_TEXT) -> MagicMock:
    """Return a mock that mimics the OpenAI client response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = text
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 80
    mock_response.usage.total_tokens = 200
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Unit tests — generator.py
# ---------------------------------------------------------------------------

class TestRecyclePost:
    @patch("src.generator.OpenAI", return_value=_make_openai_mock())
    def test_returns_dict_with_required_keys(self, mock_openai: MagicMock) -> None:
        from src.generator import recycle_post

        result = recycle_post(
            original_text="AI is changing the world.",
            source_platform="LinkedIn",
            original_date="2022-06-01",
            target_platform="Twitter",
            provider="openai",
        )
        required = {"recycled_text", "original_text", "source_platform", "target_platform", "usage"}
        assert required.issubset(result.keys())

    @patch("src.generator.OpenAI", return_value=_make_openai_mock())
    def test_recycled_text_is_non_empty(self, mock_openai: MagicMock) -> None:
        from src.generator import recycle_post

        result = recycle_post(
            original_text="Productivity hacks for remote engineers.",
            source_platform="Twitter",
            original_date="2021-09-15",
            target_platform="LinkedIn",
            provider="openai",
        )
        assert isinstance(result["recycled_text"], str)
        assert len(result["recycled_text"]) > 0

    @patch("src.generator.OpenAI", return_value=_make_openai_mock())
    def test_usage_dict_populated(self, mock_openai: MagicMock) -> None:
        from src.generator import recycle_post

        result = recycle_post(
            original_text="Python tips.",
            source_platform="Instagram",
            original_date="2022-01-01",
            target_platform="LinkedIn",
            provider="openai",
        )
        usage = result["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert usage["total_tokens"] > 0

    def test_invalid_provider_raises(self) -> None:
        from src.generator import recycle_post

        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            recycle_post(
                original_text="Some text.",
                source_platform="Twitter",
                original_date="2023-01-01",
                target_platform="LinkedIn",
                provider="unknown_llm",
            )

    def test_missing_openai_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        import importlib, src.generator as gen_mod
        importlib.reload(gen_mod)

        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            gen_mod._generate_openai("test prompt")


# ---------------------------------------------------------------------------
# Integration test — full pipeline (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Shared populated DB for pipeline tests."""
    tmp = tmp_path_factory.mktemp("pipeline_db")
    csv_path = tmp / "posts.csv"
    db_path = str(tmp / "vector_db")

    data = {
        "post_id": ["P001", "P002", "P003"],
        "platform": ["LinkedIn", "Twitter", "LinkedIn"],
        "original_text": [
            "Artificial intelligence is reshaping the way companies hire and manage talent.",
            "Python remains the top language for machine learning and data science in 2023.",
            "Remote work is here to stay. Here are 5 habits to stay productive at home.",
        ],
        "engagement_score": [300, 175, 220],
        "date_posted": ["2022-04-10", "2023-02-20", "2022-12-05"],
        "tone": ["professional", "informative", "casual"],
        "tags": ["#AI #Hiring", "#Python #ML", "#RemoteWork"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    from src.ingestion import ingest
    ingest(data_path=str(csv_path), persist_dir=db_path)
    return db_path


class TestFullPipeline:
    @patch("src.generator.OpenAI", return_value=_make_openai_mock())
    def test_pipeline_runs_end_to_end(self, mock_openai: MagicMock, pipeline_db: str) -> None:
        """Retrieve → generate → evaluate without errors."""
        from src.retrieval import retrieve_posts
        from src.generator import recycle_post
        from src.eval import evaluate

        # Step 1: Retrieve
        results = retrieve_posts(query="AI in the workplace", top_n=1, persist_dir=pipeline_db)
        assert len(results) >= 1
        top = results[0]

        # Step 2: Generate
        result = recycle_post(
            original_text=top["original_text"],
            source_platform=top["platform"],
            original_date=top["date_posted"],
            target_platform="Twitter",
            provider="openai",
        )
        assert result["recycled_text"]

        # Step 3: Evaluate
        report = evaluate(
            original=top["original_text"],
            recycled=result["recycled_text"],
            log_to_file=False,
        )
        assert 0.0 <= report["bleu_score"] <= 1.0

    @patch("src.generator.OpenAI", return_value=_make_openai_mock())
    def test_pipeline_with_platform_filter(self, mock_openai: MagicMock, pipeline_db: str) -> None:
        from src.retrieval import retrieve_posts
        from src.generator import recycle_post

        results = retrieve_posts(
            query="programming language",
            top_n=3,
            platform_filter="Twitter",
            persist_dir=pipeline_db,
        )
        # May return 0 results if only Twitter posts filtered — just ensure no crash
        if results:
            top = results[0]
            result = recycle_post(
                original_text=top["original_text"],
                source_platform=top["platform"],
                original_date=top["date_posted"],
                target_platform="LinkedIn",
                provider="openai",
            )
            assert isinstance(result["recycled_text"], str)
