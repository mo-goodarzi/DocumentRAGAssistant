"""Validate and ingest user-uploaded PDFs at runtime.

Ingestion at build time could assume well-formed input. Runtime uploads cannot:
files arrive from strangers, embedding costs money per page, and a scanned PDF
produces an index that silently answers nothing. Everything here exists to fail
early and say why.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pdfplumber
from openai import OpenAI

import config
from chunker import chunk_document
from session import get_session_collection

log = logging.getLogger(__name__)
client = OpenAI()


class UploadRejected(Exception):
    """Validation failure with a message intended for the user."""


@dataclass
class IngestResult:
    filename: str
    pages: int
    chunks: int
    cost_usd: float


def validate(path: Path, filename: str, existing_files: int) -> int:
    """Check a PDF before spending anything on it. Returns the page count.

    Order matters: cheap checks first, so a 200MB file is rejected on size
    before anything tries to parse it.
    """
    if existing_files >= config.MAX_FILES_PER_SESSION:
        raise UploadRejected(
            f"Limit of {config.MAX_FILES_PER_SESSION} documents per session. "
            "Remove one to add another."
        )

    size_mb = path.stat().st_size / 1_048_576
    if size_mb > config.MAX_UPLOAD_MB:
        raise UploadRejected(
            f"{filename} is {size_mb:.1f}MB — the limit is {config.MAX_UPLOAD_MB}MB."
        )

    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)

            if n_pages > config.MAX_PAGES_PER_FILE:
                raise UploadRejected(
                    f"{filename} has {n_pages} pages — the limit is "
                    f"{config.MAX_PAGES_PER_FILE}."
                )

            # A scanned PDF has no text layer. Sample a few pages rather than
            # all of them: cheap, and enough to tell the difference between a
            # scan and a document that merely opens with a title page.
            sample = pdf.pages[: min(5, n_pages)]
            extracted = sum(len(p.extract_text() or "") for p in sample)

            if extracted < 200:
                raise UploadRejected(
                    f"{filename} has no extractable text in its first pages. "
                    "Scanned documents need OCR, which this app does not do."
                )

    except UploadRejected:
        raise
    except Exception as exc:
        # Encrypted, corrupt, or not actually a PDF.
        raise UploadRejected(f"Could not read {filename}: {exc}")

    return n_pages


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def ingest_upload(
    session_id: str,
    file_bytes: bytes,
    filename: str,
    existing_files: int = 0,
    on_progress: Callable[[float, str], None] | None = None,
) -> IngestResult:
    """Validate, chunk, embed and index one uploaded PDF.

    `on_progress(fraction, message)` is called as batches complete. A 140-page
    document takes well over a minute to embed; without feedback the user
    assumes the app has hung and closes the tab.
    """
    # chunk_document works from a path, and the uploaded file only exists in
    # memory — so it goes to a temp file that is removed on the way out.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        path.write_bytes(file_bytes)

        n_pages = validate(path, filename, existing_files)

        if on_progress:
            on_progress(0.05, f"Reading {filename} ({n_pages} pages)")

        chunks = chunk_document(path)

    if not chunks:
        raise UploadRejected(
            f"No usable text extracted from {filename} after cleaning."
        )

    collection = get_session_collection(session_id)

    # Re-uploading the same filename replaces it rather than duplicating —
    # chunk ids are content-derived, so an edited file would otherwise leave
    # both generations in the index.
    existing = collection.get(where={"source": filename}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    total_batches = (len(chunks) + config.EMBED_BATCH - 1) // config.EMBED_BATCH
    embedded_tokens = 0

    for i, start in enumerate(range(0, len(chunks), config.EMBED_BATCH)):
        batch = chunks[start : start + config.EMBED_BATCH]
        texts = [c.text for c in batch]

        collection.add(
            ids=[c.id for c in batch],
            documents=texts,
            embeddings=embed_batch(texts),
            metadatas=[
                {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                for c in batch
            ],
        )

        embedded_tokens += sum(len(t) for t in texts) // 4  # rough token estimate

        if on_progress:
            done = (i + 1) / total_batches
            on_progress(
                0.05 + 0.95 * done,
                f"Indexing {filename} — batch {i + 1} of {total_batches}",
            )

    cost = embedded_tokens * config.PRICE_EMBED / 1_000_000
    log.info("ingested %s: %d pages, %d chunks, ~$%.5f",
             filename, n_pages, len(chunks), cost)

    return IngestResult(
        filename=filename, pages=n_pages, chunks=len(chunks), cost_usd=cost
    )
