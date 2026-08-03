"""Central configuration. Every tuning knob lives here so experiments stay controlled."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# parents[1] because this file is at <repo>/src/config.py — so ROOT is the repo,
# regardless of where you invoke python from.
ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
CHROMA_DIR = ROOT / ".chroma"
EVAL_DIR = ROOT / "eval"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "5"))

EMBED_BATCH = 96  # texts per embeddings API call

# USD per 1M tokens — fill these in from the OpenAI pricing page and re-check
# occasionally. Used by your cost accounting in step 1 and the API response.
PRICE_IN = float(os.getenv("PRICE_IN", "0.15"))
PRICE_OUT = float(os.getenv("PRICE_OUT", "0.60"))
PRICE_EMBED = float(os.getenv("PRICE_EMBED", "0.02"))



# Runtime upload limits. These are cost controls as much as safety controls:
# every page a stranger uploads is embedding spend on your key.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_PAGES_PER_FILE = int(os.getenv("MAX_PAGES_PER_FILE", "200"))
MAX_FILES_PER_SESSION = int(os.getenv("MAX_FILES_PER_SESSION", "5"))
SESSION_TTL_HOURS = float(os.getenv("SESSION_TTL_HOURS", "6"))


def as_dict() -> dict:
    """The knobs that affect results — dumped into every eval results file."""
    return {
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }






