import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("REASONDB_CACHE_DIR", Path.home() / ".reasondb" / "cache"))
