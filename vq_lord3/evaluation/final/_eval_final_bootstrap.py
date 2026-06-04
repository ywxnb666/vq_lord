from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
parent_str = str(_PARENT_DIR)
if parent_str not in sys.path:
    sys.path.insert(0, parent_str)
