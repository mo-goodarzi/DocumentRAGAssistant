"""Split page text into overlapping token windows with citation metadata."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


import tiktoken

import config
from extract import extract_pages

encoder = tiktoken.get_encoding("cl100k_base")

MIN_CHUNK_TOKENS = 40   # a shorter trailing window is a fragment, not a passage


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Slide a fixed-size token window across `text`, stepping by size - overlap.

    Token-based rather than character-based because the constraint we're
    respecting — context window, embedding input limit — is measured in tokens.
    A 500-character window varies wildly in token count depending on the text.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the window never advances")

    tokens = encoder.encode(text)
    if not tokens:
        return []

    step = size - overlap
    chunks: list[str] = []

    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]

        # Trailing scrap: the overlap already carried this text into the
        # previous chunk, so emitting it alone adds noise and no information.
        if len(window) < MIN_CHUNK_TOKENS and chunks:
            break

        chunks.append(encoder.decode(window))

        # This window reached the end. Without this, the next iteration would
        # emit a shorter, fully-redundant tail.
        if start + size >= len(tokens):
            break

    return chunks


def chunk_document(pdf_path: Path) -> list[Chunk]:
    """Chunk one PDF into retrievable units carrying source and page.

    DECISION — chunks never span page boundaries. Chunking happens per page,
    so a sentence broken across a page break is split between two chunks.
    The trade: we lose a little context at the seams, and in exchange every
    chunk has exactly one page number, so every citation is precise. For a
    legal assistant, a citation the user cannot verify is worth nothing, so
    the trade goes this way. A general-knowledge corpus might choose the
    opposite.
    """
    source = pdf_path.name
    chunks: list[Chunk] = []

    for page_no, text in extract_pages(pdf_path):
        for i, piece in enumerate(chunk_text(text, config.CHUNK_TOKENS, config.CHUNK_OVERLAP)):
            # Stable across runs: same input → same id → re-ingesting updates
            # in place instead of creating duplicates. Text prefix is included
            # so an edited document produces new ids rather than silently
            # keeping stale content under an old one.
            digest = hashlib.sha1(
                f"{source}|{page_no}|{i}|{piece[:200]}".encode()
            ).hexdigest()[:16]

            chunks.append(
                Chunk(
                    id=digest,
                    text=piece,
                    source=source,
                    page=page_no,
                    chunk_index=i,
                )
            )

    return chunks


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1])
    chunks = chunk_document(path)
    print(f"{path.name}: {len(chunks)} chunks\n")

    for c in chunks[4:7]:
        n_tokens = len(encoder.encode(c.text))
        print(f"--- {c.id}  p.{c.page}  #{c.chunk_index}  ({n_tokens} tokens) ---")
        print(c.text[:300].replace("\n", " "), "...\n")