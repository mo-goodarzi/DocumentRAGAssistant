"""Embed chunks and store them in a local Chroma collection.

    python src/ingest.py              # incremental — skips files already indexed
    python src/ingest.py --rebuild    # drop the collection and start over
"""

import argparse
import sys

import chromadb
from openai import OpenAI
from tqdm import tqdm

import config
from chunker import Chunk, chunk_document

client = OpenAI()


def get_collection(rebuild: bool = False):
    """Open (or create) the persistent collection.

    hnsw:space must be set at creation and cannot be changed later. Cosine is
    chosen to match how the embeddings are meant to be compared — the same
    formula you wrote by hand in experiment 02. Chroma's default is L2, which
    would rank differently and silently.
    """
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    if rebuild:
        try:
            chroma.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # nothing to delete on a first run

    return chroma.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one API call.

    Batching matters: GDPR alone is ~400 chunks. One request per chunk means
    hundreds of round trips and rate-limit backoff; batches of ~96 means a few.
    """
    response = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    # Sort by index — the API does not guarantee response order matches input.
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def already_indexed(collection, source: str) -> bool:
    hit = collection.get(where={"source": source}, limit=1)
    return bool(hit["ids"])


def add_chunks(collection, chunks: list[Chunk]) -> None:
    """Embed and insert in batches, with a progress bar."""
    for start in tqdm(range(0, len(chunks), config.EMBED_BATCH), unit="batch"):
        batch = chunks[start : start + config.EMBED_BATCH]
        texts = [c.text for c in batch]

        collection.add(
            ids=[c.id for c in batch],
            documents=texts,
            embeddings=embed(texts),
            metadatas=[
                {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                for c in batch
            ],
        )


def main(rebuild: bool) -> None:
    pdfs = sorted(config.CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {config.CORPUS_DIR}")

    collection = get_collection(rebuild)
    added = 0

    for pdf_path in pdfs:
        if not rebuild and already_indexed(collection, pdf_path.name):
            print(f"skip   {pdf_path.name}")
            continue

        print(f"chunk  {pdf_path.name}")
        chunks = chunk_document(pdf_path)
        if not chunks:
            print("       no text extracted — scanned PDF? skipping")
            continue

        print(f"embed  {len(chunks)} chunks")
        add_chunks(collection, chunks)
        added += len(chunks)

    print(f"\ncollection holds {collection.count()} chunks ({added} added this run)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    main(parser.parse_args().rebuild)