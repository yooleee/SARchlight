# =============================================================================
# conftest.py (repo root)
# -----------------------------------------------------------------------------
# Responsible for: Putting the repo root on sys.path so tests can import the
#                  `src.common` / `src.search` packages without an install step.
# Role in project: Test-only convenience. Keeps `from src.common... import ...`
#                  working when running `pytest` from the repo root, which is the
#                  same import path the demo and the rest of the system use.
# =============================================================================

import pathlib
import sys

# Insert the directory containing this file (the repo root) at the front of the
# import path so the top-level `src` package resolves during test collection.
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
