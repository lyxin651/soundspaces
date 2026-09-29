#!/usr/bin/env python3
"""Compatibility entry point for the audited A4 smoke re-freeze.

The former A4-4P implementation generated placeholder cache keys and the
pre-audit noise parents.  The only supported preparation path now delegates
to the strict manual-audit re-freeze command, which produces a v2 manifest
and a metadata-only cache plan.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.refreeze_a4_smoke_with_audit import main  # noqa: E402


if __name__ == "__main__":
    main()
