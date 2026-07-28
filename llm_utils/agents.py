"""Small reusable agent pieces shared across notebooks.

VENDORED from RL-Agents-Workshop/workshop_utils/agents.py with exactly ONE
additive change: `make_agent` and `make_baseline_agent` accept an `llm_fn` hook
so the SAME agent can be driven by a local HF/vLLM policy instead of the OpenAI
client. Everything else -- `extract_sql`, `BASELINE_SYSTEM`, `baseline_prompt`,
`REPAIR_SYSTEM`, `repair_prompt` -- is byte-identical to the original.

That byte-identity is the fairness contract of this workshop. It means the
fine-tuned Qwen policy and gpt-4o-mini see the *same prompt string* and are
parsed by the *same parser*, so the head-to-head in NB8 isolates exactly one
variable: the weights.

A direct consequence for reward design (NB3): the format reward must reward a
```sql fenced block, because `extract_sql` below is what will parse the model's
output at eval time. Train to the parser you will be scored by.
"""

from __future__ import annotations

import re

from .db import SCHEMA_TEXT, run_sql
from .llm import llm, observe

# Matches a ```sql ... ``` or ``` ... ``` fenced block.
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model response.

    Handles fenced code blocks, a leading 'SQL:' label, and trailing prose.
    Returns the first statement (up to the first ';' if present).
    """
    text = (text or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # Drop a leading "SQL:" style label if present.
    text = re.sub(r"^\s*sql\s*:\s*", "", text, flags=re.IGNORECASE)
    # Keep only up to the first complete statement.
    if ";" in text:
        text = text.split(";")[0] + ";"
    return text.strip()


BASELINE_SYSTEM = (
    "You are a precise text-to-SQL assistant for a SQLite database. "
    "Return ONE valid SQLite SELECT query that answers the user's question. "
    "Output ONLY the SQL inside a ```sql code block -- no explanation."
)


def baseline_prompt(question: str, extra: str = "") -> list:
    """Build the zero-shot baseline messages for a question.

    `extra` is an optional block (memory / skills) injected by later notebooks;
    the baseline passes "".
    """
    user = f"Database schema:\n{SCHEMA_TEXT}\n"
    if extra:
        user += f"\n{extra}\n"
    user += f"\nQuestion: {question}\nSQL:"
    return [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_baseline_agent(model: str | None = None, llm_fn=None):
    """Return an agent_fn(question) -> sql: a single zero-shot call, no loop.

    One shot, text in, text out, no ability to notice it failed. `llm_fn` lets a
    local policy stand in for the OpenAI client (see `local_llm.LocalLM.as_llm_fn`).
    """
    _llm = llm_fn or llm

    def agent_fn(question: str) -> str:
        raw = _llm(baseline_prompt(question), model=model)
        return extract_sql(raw)

    return agent_fn


REPAIR_SYSTEM = (
    "You are a meticulous SQLite debugging expert. A query failed to execute. "
    "Diagnose the likely cause and return a corrected query. "
    "Output ONLY the corrected SQL inside a ```sql code block -- no explanation."
)


def repair_prompt(question: str, sql: str, error: str, extra: str = "") -> list:
    """Messages asking the model to fix a query that raised a database error."""
    user = f"Database schema:\n{SCHEMA_TEXT}\n"
    if extra:
        user += f"\n{extra}\n"
    user += (
        f"\nQuestion: {question}"
        f"\n\nThis SQL was attempted:\n{sql}"
        f"\n\nIt failed with this database error:\n{error}"
        f"\n\nReturn a corrected SQLite query."
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": user},
    ]


def make_agent(model: str | None = None, extra: str = "", max_repairs: int = 2, llm_fn=None):
    """The text-to-SQL agent: brain + tool + execution loop.

    Per question:
      1. ask the brain for SQL, given the schema and any injected `extra` block;
      2. run it with `run_sql` (the tool);
      3. if it raises a database error, feed the error back and retry, up to
         `max_repairs` times. It does NOT retry on a clean run that simply
         returns the "wrong" rows -- catching subtly wrong answers needs the
         reward signal (`score_sql`), which is what we optimize against here.

    `llm_fn` is the ONE addition over the original: pass
    `LocalLM(...).as_llm_fn()` to drive this exact agent with a local policy, or
    an ART-served OpenAI-compatible client. The harness is held fixed across all
    of them -- only the weights behind `llm_fn` change.

    Returns agent_fn(question) -> sql.
    """
    _llm = llm_fn or llm

    @observe(name="sql_agent")
    def agent_fn(question: str) -> str:
        sql = extract_sql(_llm(baseline_prompt(question, extra=extra), model=model))
        for _ in range(max_repairs):
            _, err = run_sql(sql)
            if err is None:
                return sql  # executed cleanly -> done
            sql = extract_sql(_llm(repair_prompt(question, sql, err, extra), model=model))
        return sql

    return agent_fn
