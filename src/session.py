"""Per-session Chroma collections.

The fixed-corpus version had one global collection. Once documents arrive at
runtime from different visitors, that breaks: one person's upload would answer
another person's question. Each session gets its own collection instead, named
`session_<uuid>`, and the shared demo corpus stays read-only under its own name.

Collections are ephemeral by design. They are deleted when the user clears them,
and any collection older than SESSION_TTL_HOURS is swept on startup — otherwise
a public deployment accumulates orphaned indexes until the disk fills.
"""

import logging
import re
import time
import uuid

import chromadb

import config

log = logging.getLogger(__name__)

SESSION_PREFIX = "session_"
_SESSION_NAME_RE = re.compile(rf"^{SESSION_PREFIX}[0-9a-f]{{32}}$")


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def new_session_id() -> str:
    """A collection-name-safe session id. Chroma names must be alphanumeric,
    underscore or hyphen, so the uuid is stripped of dashes."""
    return uuid.uuid4().hex


def collection_name(session_id: str) -> str:
    return f"{SESSION_PREFIX}{session_id}"


def get_session_collection(session_id: str):
    """Open (creating if needed) this session's collection.

    Cosine space must match the demo collection and the query path — a
    collection created with the default L2 would rank differently, and nothing
    would error to tell you.
    """
    return _client().get_or_create_collection(
        name=collection_name(session_id),
        metadata={"hnsw:space": "cosine", "created_at": time.time()},
    )


def get_demo_collection():
    """The shared, pre-built EU regulation index. Read-only — never written to
    at runtime, so one visitor cannot pollute what everyone else sees."""
    return _client().get_collection(config.COLLECTION_NAME)


def session_documents(session_id: str) -> dict[str, int]:
    """Filenames currently indexed for this session, with chunk counts."""
    try:
        collection = get_session_collection(session_id)
        metas = collection.get(include=["metadatas"])["metadatas"]
    except Exception:
        return {}

    counts: dict[str, int] = {}
    for m in metas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return counts


def delete_document(session_id: str, source: str) -> int:
    """Remove one uploaded document from the session index."""
    collection = get_session_collection(session_id)
    hits = collection.get(where={"source": source}, include=[])
    if hits["ids"]:
        collection.delete(ids=hits["ids"])
    return len(hits["ids"])


def clear_session(session_id: str) -> None:
    """Drop the whole collection. Called by the clear button and at session end."""
    try:
        _client().delete_collection(collection_name(session_id))
        log.info("cleared session %s", session_id[:8])
    except Exception as exc:
        log.warning("could not clear session %s: %s", session_id[:8], exc)


def sweep_expired(ttl_hours: float | None = None) -> int:
    """Delete session collections older than the TTL.

    Called once at app startup. Chroma has no TTL of its own, so without this a
    public deployment leaks a collection per visitor forever. The demo
    collection is protected by the name check — only `session_<32 hex>` names
    are ever considered.
    """
    ttl = (ttl_hours or config.SESSION_TTL_HOURS) * 3600
    now = time.time()
    client = _client()
    removed = 0

    for coll in client.list_collections():
        name = coll.name if hasattr(coll, "name") else str(coll)

        if not _SESSION_NAME_RE.match(name):
            continue  # demo corpus, or something we did not create

        created = (coll.metadata or {}).get("created_at")
        # No timestamp means it predates this field — treat as expired rather
        # than leaving it around forever.
        if created is None or (now - created) > ttl:
            try:
                client.delete_collection(name)
                removed += 1
            except Exception as exc:
                log.warning("sweep failed for %s: %s", name, exc)

    if removed:
        log.info("swept %d expired session collection(s)", removed)
    return removed
