import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieve import retrieve, get_collection
from session import get_session_collection

from session import get_session_collection, new_session_id, clear_session
from ingest import embed

a, b = new_session_id(), new_session_id()

# put something in A only
text = "The maximum penalty is 42 million euro."
get_session_collection(a).add(
    ids=["t1"], documents=[text], embeddings=embed([text]),
    metadatas=[{"source": "test.pdf", "page": 1, "chunk_index": 0}],
)

print(retrieve("penalty", collection=get_session_collection(a)))  # finds it
print(retrieve("penalty", collection=get_session_collection(b)))  # empty

clear_session(a); clear_session(b)