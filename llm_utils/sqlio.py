"""A safe, fast SQL execution + scoring layer for the training loop.

Two problems with using the vendored `db.run_sql` / `db.score_sql` directly as a
training reward, neither of which matters at repo 1's scale (a few hundred calls)
and both of which matter here (millions):

1. **A leaked connection on the error path.** `db.run_sql` closes its connection
   on success but not in its `except` branch. CPython usually collects it when
   the frame dies, but anything that keeps a traceback alive (a debugger, a
   notebook's `_` history, pytest's report) keeps the handle open too -- and on
   Windows that makes the next `build_db()` fail outright. During GRPO the model
   emits a *lot* of invalid SQL, so this is the hot path, not the edge case.

2. **The gold query is re-executed on every single scoring call.** `score_sql`
   runs gold *and* prediction every time. In GRPO with G=8 rollouts over 800
   prompts, that is ~6,400 redundant executions of a query whose answer cannot
   change -- the database is immutable during training.

We cannot fix either in `db.py`: it is a byte-identical vendored file and its
hash is the fairness contract (`tests/test_vendored_parity.py`).

So this module reimplements them safely, and `tests/test_rewards.py` asserts
that `fast_score_sql` agrees with the vendored `score_sql` on every task and on
a corpus of deliberately-wrong queries. **The vendored `score_sql` remains the
authority for every headline number**; this is strictly a training-loop
accelerator that is proven to agree with it.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from .db import DB_PATH, _normalize

# Read-only queries only. Anything that could mutate the DB would corrupt the
# environment for every subsequent rollout in the run.
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "begin", "commit",
)


def safe_run_sql(query: str, path: str = DB_PATH, timeout: float = 5.0):
    """Execute a read-only query. Returns (rows, error); rows is None on error.

    Same contract as `db.run_sql`, but the connection is closed in a `finally`
    so it cannot leak, and a per-query timeout stops a pathological generation
    (a cross join of every table, say) from stalling a training step.
    """
    con = None
    try:
        con = sqlite3.connect(path, timeout=timeout)
        con.execute("PRAGMA query_only = ON;")
        cur = con.execute(query)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:  # noqa: BLE001 - surface any SQL error verbatim
        return None, f"{type(e).__name__}: {e}"
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass


def is_read_only(query: str) -> bool:
    """True if `query` is a bare SELECT (or a WITH ... SELECT).

    Used by the safety reward and by the multi-turn env's `run_query` tool. The
    DB is opened `query_only` anyway, so this is defence in depth -- but it also
    gives the policy a clean 0 for trying, which is the signal we want.
    """
    q = query.strip().lstrip("(").lower()
    if not (q.startswith("select") or q.startswith("with")):
        return False
    # Reject stacked statements: "SELECT 1; DROP TABLE customers"
    body = query.split("--")[0]
    if body.count(";") > 1 or (body.count(";") == 1 and not body.rstrip().endswith(";")):
        return False
    return not any(f" {kw} " in f" {q} " for kw in _FORBIDDEN)


@lru_cache(maxsize=4096)
def _gold_key(gold_sql: str, path: str):
    """Execute a gold query once and cache its normalized result.

    Returns (normalized_rows, ordered). Raises ValueError if the gold is broken,
    matching `db.score_sql`'s behaviour -- a bad gold must be loud, because
    silently scoring 0 would look like a model failure.

    Safe to cache: the DB is immutable for the lifetime of a training run. Call
    `clear_gold_cache()` if you ever rebuild it with a different seed.
    """
    rows, err = safe_run_sql(gold_sql, path)
    if err is not None:
        raise ValueError(f"Gold SQL failed (fix the dataset): {err}\n{gold_sql}")
    ordered = "order by" in gold_sql.lower()
    return _normalize(rows, ordered), ordered


def clear_gold_cache() -> None:
    _gold_key.cache_clear()


def gold_cache_info():
    return _gold_key.cache_info()


def fast_score_sql(pred_sql: str, gold_sql: str, path: str = DB_PATH) -> bool:
    """Execution match, with the gold side cached. Agrees with `db.score_sql`.

    This is the reward the training loop calls. It is NOT used for any headline
    number -- those go through the vendored `evaluate()` / `score_sql` so the
    comparison with repo 1 runs on repo 1's exact code.
    """
    gold_rows, ordered = _gold_key(gold_sql, path)
    pred_rows, pred_err = safe_run_sql(pred_sql, path)
    if pred_err is not None:
        return False
    return _normalize(pred_rows, ordered) == gold_rows


def exec_info(pred_sql: str, gold_sql: str, path: str = DB_PATH) -> dict:
    """One execution, every fact the reward components need.

    Calling `r_executes`, `r_nonempty`, and `r_exec_match` separately would run
    the same query three times. The composite reward calls this once instead.
    """
    if not pred_sql.strip():
        return {"executes": False, "error": "empty", "n_rows": 0, "n_cols": 0,
                "correct": False, "read_only": False}
    read_only = is_read_only(pred_sql)
    rows, err = safe_run_sql(pred_sql, path)
    if err is not None:
        return {"executes": False, "error": err, "n_rows": 0, "n_cols": 0,
                "correct": False, "read_only": read_only}
    gold_rows, ordered = _gold_key(gold_sql, path)
    correct = _normalize(rows, ordered) == gold_rows
    return {
        "executes": True,
        "error": None,
        "n_rows": len(rows),
        "n_cols": len(rows[0]) if rows else 0,
        "correct": correct,
        "read_only": read_only,
        "gold_rows": len(gold_rows),
        "gold_cols": len(gold_rows[0]) if gold_rows else 0,
    }
