"""Discoverable Step 3 and tooling tests."""

import sys
from pathlib import Path


REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if "__path__" in globals():
    __path__.append(str(Path(REPO_ROOT) / "tools"))
