import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieve import retrieve, get_collection
from session import get_session_collection

# default path — existing behaviour must be unchanged
print(len(retrieve("Which AI practices are prohibited?")))

# explicit demo collection — same result
print(len(retrieve("Which AI practices are prohibited?", collection=get_collection())))

# empty session collection — should return nothing, not crash
print(retrieve("anything", collection=get_session_collection("deadbeef" * 4)))