# Fixed personalized-learning-system.py
# - Removed premature if __name__ == "__main__" block that ran before class definitions
# - Expanded source ends with PersonalizedLearningSystem(config=DB_CONFIG)
# Payload is zlib+base64 split across sibling files (MCP push size limits).
from pathlib import Path
import zlib
import base64

_dir = Path(__file__).resolve().parent
_b64 = (_dir / "_pls_b64_a.txt").read_text(encoding="ascii") + (_dir / "_pls_b64_b.txt").read_text(encoding="ascii")
exec(zlib.decompress(base64.b64decode(_b64)).decode("utf-8"), globals())
