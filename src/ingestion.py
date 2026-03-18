"""
ingestion.py
============
Loads old_posts.csv → chunks long posts → generates embeddings
via SentenceTransformers → upserts into a persistent ChromaDB collection.

Usage (standalone):
    python -m src.ingestion            # ingest
    python -m src.ingestion --reset    # wipe DB then re-ingest
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./vector_db")
DATA_PATH: str = os.getenv("DATA_PATH", "./data/old_posts.csv")
COLLECTION_NAME: str = "social_posts"
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_TOKENS: int = 512  # approx character limit (~4 chars/token)
MAX_CHUNK_CHARS: int = MAX_CHUNK_TOKENS * 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_posts(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV and return a cleaned DataFrame."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)
    required = {"post_id", "platform", "original_text", "engagement_score", "date_posted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df.dropna(subset=["original_text"])
    df["original_text"] = df["original_text"].astype(str).str.strip()
    df = df[df["original_text"] != ""]
    log.info("Loaded %d posts from %s", len(df), path)
    return df


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split text into chunks of at most `max_chars` characters.
    Tries to split on sentence boundaries first.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""
    for sent in sentences:
        candidate = (current + ". " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent[:max_chars]  # hard cut if a single sentence is too long
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def get_chroma_collection(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest(
    data_path: str = DATA_PATH,
    persist_dir: str = CHROMA_PERSIST_DIR,
    reset: bool = False,
) -> int:
    """
    Main ingestion entry-point.

    Parameters
    ----------
    data_path   : path to the CSV dataset
    persist_dir : ChromaDB persistence directory
    reset       : if True, wipe the collection before re-indexing

    Returns
    -------
    int  number of documents upserted
    """
    df = load_posts(data_path)

    # -- Embedding model --------------------------------------------------
    log.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    # -- ChromaDB ---------------------------------------------------------
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    if reset:
        log.warning("--reset flag set: deleting existing collection '%s'", COLLECTION_NAME)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # -- Build documents --------------------------------------------------
    ids: List[str] = []
    texts: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        chunks = chunk_text(str(row["original_text"]))
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{row['post_id']}_chunk{chunk_idx}"
            ids.append(doc_id)
            texts.append(chunk)
            metadatas.append(
                {
                    "post_id": str(row["post_id"]),
                    "platform": str(row["platform"]),
                    "engagement_score": float(row["engagement_score"]),
                    "date_posted": str(row["date_posted"]),
                    "tone": str(row.get("tone", "")),
                    "tags": str(row.get("tags", "")),
                    "chunk_index": chunk_idx,
                    "original_text": chunk,  # stored for retrieval convenience
                }
            )

    log.info("Generating embeddings for %d chunks…", len(texts))
    embedding_list = embedder.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embedding_list.tolist()

    # -- Upsert in batches ------------------------------------------------
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        log.info("Upserted chunk batch %d–%d", start, min(end, len(ids)))

    total = collection.count()
    log.info("Ingestion complete. Collection '%s' now contains %d documents.", COLLECTION_NAME, total)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest social posts into ChromaDB.")
    parser.add_argument("--data", default=DATA_PATH, help="Path to old_posts.csv")
    parser.add_argument("--db", default=CHROMA_PERSIST_DIR, help="ChromaDB persist directory")
    parser.add_argument("--reset", action="store_true", help="Wipe DB before re-indexing")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    count = ingest(data_path=args.data, persist_dir=args.db, reset=args.reset)
    print(f"\n✅ Ingestion complete — {count} chunks in vector DB.")
