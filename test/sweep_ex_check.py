import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session import sweep_expired, get_demo_collection, new_session_id, get_session_collection
sid = new_session_id()
get_session_collection(sid)
print(sweep_expired(ttl_hours=0))      # should remove it
print(get_demo_collection().count())   # demo corpus MUST be untouched