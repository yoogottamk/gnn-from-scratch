import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "datasets")))
