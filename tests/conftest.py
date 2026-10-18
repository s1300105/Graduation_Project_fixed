"""Make the repository root importable (utils/, models/, run.py) no matter how
pytest is invoked: `pytest`, `python -m pytest`, or from another directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
