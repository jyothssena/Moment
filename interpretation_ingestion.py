# vector_db.py
# Vector DB layer for Project Moment using ChromaDB.
# Handles ingestion, embedding, and two-stage retrieval (vector → category filter).
#
# Dependencies:
#   pip install chromadb sentence-transformers

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-mpnet-base-v2"   # local, no API key needed; 768 dims
COLLECTION_NAME = "moment_interpretations"
VECTOR_CANDIDATE_POOL = 200   # stage 1: broad vector recall
FINAL_CANDIDATES     = 25     # stage 2: after resonance filter


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Moment:
    user_id: str
    passage: str                          # raw book passage (context, not embedded)
    interpretation: str                   # what gets embedded
    #resonance_categories: list[str]       # e.g. ["epistemic_humility", "mortality"]
    moment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    book_title: Optional[str] = None
    author: Optional[str] = None


@dataclass
class MatchResult:
    moment_id: str
    user_id: str
    interpretation: str
    book_title: Optional[str]
    resonance_categories: list[str]
    vector_score: float                   # cosine similarity from stage 1


# ── Client setup ──────────────────────────────────────────────────────────────

def build_client(persist_dir: str = "./moment_chroma_db") -> chromadb.ClientAPI:
    """Persistent local ChromaDB client. Swap for HttpClient to use a remote server."""
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Get or create the moments collection with OpenAI embeddings.
    ChromaDB calls the embedding function automatically on upsert/query
    so we never need to manage vectors manually.
    """
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )  # model downloads automatically on first run (~420MB)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_moment(collection: chromadb.Collection, moment: Moment) -> str:
    """
    Embed the interpretation and upsert into ChromaDB.
    Resonance categories are stored as a comma-separated metadata string
    because ChromaDB metadata values must be scalar.
    Returns the moment_id.
    """
    collection.upsert(
        ids=[moment.moment_id],
        documents=[moment.interpretation],   # ChromaDB embeds this automatically
        metadatas=[{
            "user_id":               moment.user_id,
            "passage":               moment.passage,
            "book_title":            moment.book_title or "",
            "author":                moment.author or "",
            #"resonance_categories":  ",".join(moment.resonance_categories),
        }],
    )
    return moment.moment_id


def ingest_batch(collection: chromadb.Collection, moments: list[Moment]) -> list[str]:
    """Batch upsert — more efficient than one-by-one for bulk imports."""
    if not moments:
        return []
    collection.upsert(
        ids=[m.moment_id for m in moments],
        documents=[m.interpretation for m in moments],
        metadatas=[{
            "user_id":               m.user_id,
            "passage":               m.passage,
            "book_title":            m.book_title or "",
            "author":                m.author or "",
            #"resonance_categories":  ",".join(m.resonance_categories),
        } for m in moments],
    )
    return [m.moment_id for m in moments]


# ── Two-stage retrieval ───────────────────────────────────────────────────────

def find_matches(
    collection: chromadb.Collection,
    query_moment: Moment,
    exclude_user_id: Optional[str] = None,
) -> list[MatchResult]:
    """
    Two-stage matching pipeline:
      Stage 1 — Vector search: retrieve top VECTOR_CANDIDATE_POOL by cosine similarity.
      Stage 2 — Resonance filter: re-rank by category overlap, return top FINAL_CANDIDATES.

    exclude_user_id: pass the querying user's ID to avoid self-matches.
    """
    # ── Stage 1: broad vector recall ──────────────────────────────────────────
    results = collection.query(
        query_texts=[query_moment.interpretation],
        n_results=VECTOR_CANDIDATE_POOL,
        where={"user_id": {"$ne": exclude_user_id}} if exclude_user_id else None,
        include=["metadatas", "distances", "documents"],
    )

    ids       = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance (0 = identical)

    # ── Stage 2: resonance category overlap ───────────────────────────────────
    query_cats = set(query_moment.resonance_categories)

    scored: list[tuple[int, MatchResult]] = []
    for mid, meta, dist in zip(ids, metadatas, distances):
        candidate_cats = set(meta["resonance_categories"].split(",")) if meta["resonance_categories"] else set()
        overlap = len(query_cats & candidate_cats)
        if overlap == 0:
            continue  # no shared resonance — skip entirely

        scored.append((
            overlap,
            MatchResult(
                moment_id=mid,
                user_id=meta["user_id"],
                interpretation=meta.get("documents", ""),
                book_title=meta["book_title"] or None,
                resonance_categories=list(candidate_cats),
                vector_score=1 - dist,   # convert distance → similarity
            ),
        ))

    # Sort by resonance overlap desc, then vector score desc as tiebreaker
    scored.sort(key=lambda x: (x[0], x[1].vector_score), reverse=True)
    return [r for _, r in scored[:FINAL_CANDIDATES]]


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_moment(collection: chromadb.Collection, moment_id: str) -> None:
    collection.delete(ids=[moment_id])


def delete_user_moments(collection: chromadb.Collection, user_id: str) -> None:
    """Hard delete all moments for a user (e.g. account deletion)."""
    collection.delete(where={"user_id": user_id})


# ── JSON ingestion ────────────────────────────────────────────────────────────

def load_passages(passages_path: str) -> dict[str, str]:
    """
    Load passages file into a lookup dict: { passage_id -> passage_text }.
    Expected format:
    [
      { "id": "p_001", "text": "We are all just walking each other home." },
      ...
    ]
    """
    import json
    with open(passages_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["passage_id"]: r["cleaned_passage_text"] for r in records}


def load_moments_from_json(moments_path: str, passages_path: str) -> list[Moment]:
    """
    Load and join moments + passages.

    Moments file format:
    [
      {
        "user_id": "user_001",
        "passage_id": "p_001",
        "interpretation": "...",
        "resonance_categories": ["mortality", "companionship"],
        "book_title": "...",   // optional
        "author": "..."        // optional
      },
      ...
    ]
    moment_id is auto-generated if not present.
    Moments with an unresolvable passage_id are skipped with a warning.
    """
    import json
    passages = load_passages(passages_path)

    with open(moments_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    moments, skipped = [], 0
    for r in records:
        pid = r["passage_id"]
        if pid not in passages:
            print(f"  [warn] passage_id '{pid}' not found — skipping moment {r.get('moment_id', '?')}")
            skipped += 1
            continue
        moments.append(Moment(
            user_id=r["user_id"],
            passage=passages[pid],
            interpretation=r["cleaned_interpretation"],
            moment_id=r.get("interpretation__id", str(uuid.uuid4())),
            book_title=r.get("book_title"),
        ))

    if skipped:
        print(f"  [warn] {skipped} moment(s) skipped due to missing passage_id.")
    return moments


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest moments from a JSON file.")
    parser.add_argument("moments_file", help="Path to moments JSON file")
    parser.add_argument("passages_file", help="Path to passages JSON file")
    args = parser.parse_args()

    client     = build_client()
    collection = get_collection(client)

    moments = load_moments_from_json(args.moments_file, args.passages_file)
    ingested = ingest_batch(collection, moments)
    print(f"Ingested {len(ingested)} moments. Collection size: {collection.count()}")

    # Run a quick match against the first moment as a sanity check
    """
    if moments:
        query = moments[0]
        matches = find_matches(collection, query, exclude_user_id=query.user_id)
        print(f"\nSample matches for: \"{query.interpretation[:60]}...\"")
        for m in matches:
            print(f"  [{m.vector_score:.3f}] {m.book_title or 'untitled'} — {m.resonance_categories}")
    """