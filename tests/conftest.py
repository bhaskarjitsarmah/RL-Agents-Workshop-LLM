"""Shared pytest fixtures.

Why the DB is built ONCE per session
------------------------------------
The vendored `db.run_sql` does not close its connection on the error path:

    try:
        con = sqlite3.connect(path); ...; con.close(); return rows, None
    except Exception as e:
        return None, f"..."          # <- con is never closed here

Under CPython that connection is normally collected as soon as the frame dies,
so it is invisible in practice -- but when a test framework (or a notebook's
`_` / `__` history) keeps an exception's traceback alive, the frame survives,
the connection survives, and on **Windows** the open handle makes the next
`build_db()` fail with `PermissionError: [WinError 32]`.

We cannot fix that: `db.py` is a byte-identical vendored file and its hash is
the fairness contract (see `test_vendored_parity.py`). So we build the database
exactly once per session and never delete it mid-run. The DB is deterministic
(`seed=42`), so one build is all any test needs anyway.

For the training path -- where millions of failing queries run -- use
`llm_utils.sqlio.safe_run_sql`, which guarantees closure.
"""

from __future__ import annotations

import gc
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def _shop_db():
    """Build the toy shop DB once for the whole test session."""
    from llm_utils.db import DB_PATH, build_db

    gc.collect()  # release any handle left over from a previous run
    build_db()
    yield DB_PATH
    gc.collect()
