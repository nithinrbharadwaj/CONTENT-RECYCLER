"""
retrieval.py
============
Semantic search against the ChromaDB collection.

Usage (standalone):
    python -m src.retrieval "AI trends" --top 3
    python -m src.retrieval "productivity tips" --platform LinkedIn --top 5
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vector_db")
COLLECTION_NAME: str = "social_posts"
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded)
# ---------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=persist_dir)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_posts(
    query: str,
    top_n: int = 3,
    platform_filter: Optional[str] = None,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-N most semantically similar posts for a given query.

    Parameters
    ----------
    query           : Natural language search string
    top_n           : Number of results to return
    platform_filter : If set (e.g. "LinkedIn"), only return posts from that platform
    persist_dir     : ChromaDB persistence directory

    Returns
    -------
    List of dicts with keys:
        id, original_text, platform, engagement_score, date_posted,
        tone, tags, similarity_score
    """
    embedder = _get_embedder()

    try:
        collection = _get_collection(persist_dir)
    except Exception as e:
        raise RuntimeError(
            f"Could not open ChromaDB collection '{COLLECTION_NAME}' at '{persist_dir}'. "
            "Have you run ingestion first?  Error: " + str(e)
        ) from e

    query_embedding = embedder.encode([query])[0].tolist()

    # Build optional where filter
    where: Optional[Dict] = None
    if platform_filter:
        where = {"platform": {"$eq": platform_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_n, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    posts = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity
        similarity = round(1.0 - dist, 4)
        posts.append(
            {
                "id": meta.get("post_id", ""),
                "original_text": meta.get("original_text", doc),
                "platform": meta.get("platform", ""),
                "engagement_score": meta.get("engagement_score", 0),
                "date_posted": meta.get("date_posted", ""),
                "tone": meta.get("tone", ""),
                "tags": meta.get("tags", ""),
                "similarity_score": similarity,
            }
        )

    return posts


def print_results(results: List[Dict[str, Any]]) -> None:
    """Pretty-print retrieval results to stdout."""
    if not results:
        print("No results found.")
        return
    print(f"\n{'='*70}")
    print(f"  TOP {len(results)} RESULTS")
    print(f"{'='*70}")
    for i, post in enumerate(results, 1):
        print(
            f"\n[{i}] Platform: {post['platform']}  |  Engagement: {post['engagement_score']}"
            f"  |  Date: {post['date_posted']}  |  Similarity: {post['similarity_score']:.4f}"
        )
        print(f"    Tone: {post['tone']}  |  Tags: {post['tags']}")
        print(
            f"    Text: {post['original_text'][:200]}{'…' if len(post['original_text']) > 200 else ''}"
        )
    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve semantically similar posts.")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument(
        "--top", type=int, default=3, help="Number of results (default: 3)"
    )
    parser.add_argument(
        "--platform", default=None, help="Filter by platform, e.g. LinkedIn"
    )
    parser.add_argument(
        "--db", default=CHROMA_PERSIST_DIR, help="ChromaDB persist directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()
    results = retrieve_posts(
        query=args.query,
        top_n=args.top,
        platform_filter=args.platform,
        persist_dir=args.db,
    )
    print_results(results)
