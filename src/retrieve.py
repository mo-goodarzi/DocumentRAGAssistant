"""Question → ranked chunks."""

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from openai import OpenAI

import config

client = OpenAI()


@dataclass
class Source:
    n: int          # 1-indexed rank, used as the [n] citation marker
    text: str
    source: str
    page: int
    distance: float


@lru_cache(maxsize=1)
def get_collection():
    """Cached — reopening the client per query is wasted work, and step 9
    needs a single shared handle anyway."""
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR)
    ).get_collection(name=config.COLLECTION_NAME)


def retrieve(question: str, k: int | None = None) -> list[Source]:
    """Top-k chunks for a question, most relevant first.

    The question must be embedded with the same model used at ingestion —
    both read config.EMBED_MODEL so they cannot drift apart.
    """
    k = k or config.TOP_K

    query_embedding = client.embeddings.create(
        model=config.EMBED_MODEL, input=[question]
    ).data[0].embedding

    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        Source(
            n=i + 1,
            text=doc,
            source=meta["source"],
            page=meta["page"],
            distance=dist,
        )
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        )
    ]