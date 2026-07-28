"""The reward function -- and, in Module 5, the reward function as attack surface.

The verifiable reward
---------------------
Text-to-SQL is the spine of this workshop because the reward is *checkable*:
execute the predicted query, execute the gold, compare result sets. No LLM judge,
no human labels, no proxy. That is rare and precious -- most agent tasks do not
have it, which is why ART ships RULER for the ones that don't.

The training reward is:

    R = 1.00*exec_match + 0.15*format + 0.10*executes + 0.05*nonempty
        - 0.05*max(0, turns-1)

clipped to [-0.20, 1.35]. The shaping terms exist to break ties among *wrong*
answers -- a query that parses and runs is closer to right than one that doesn't,
and that gradient is what gets a small model off the floor. But shaping must
never let a wrong answer outscore a right one:

    min correct   = 1.00 + 0 + 0.10 + 0.05 - 0.15  = 1.00
    max incorrect = 0    + 0.15 + 0.10 + 0.05      = 0.30

**1.00 > 0.30, with room to spare.** That inequality is the whole art of reward
shaping in one line, it is asserted in `tests/test_rewards.py`, and NB6 shows
what happens when you get it wrong.

Format matters more than it looks
---------------------------------
`r_format` rewards a ```sql fenced block -- because the *vendored, unmodified*
`extract_sql` is what parses the policy's output at eval time. Train to the
parser you will be scored by. A policy trained to emit `<sql>...</sql>` would
score zero through the real harness while looking perfect in training.

The hackable reward (Module 5)
------------------------------
`r_hackable_rowcount` is a proxy that would pass code review -- it checks that
the query executes, returns a plausible number of rows, and mentions words from
the question -- and it never sees the gold at all. Optimising it produces
`SELECT * FROM orders LIMIT 5`. NB6 trains against it on purpose and plots the
proxy reward climbing while true accuracy falls.
"""

from __future__ import annotations

import re
from typing import Callable

from .agents import extract_sql
from .sqlio import (exec_info, fast_score_sql, is_read_only, is_write_statement,
                    safe_run_sql)

# Component weights. Kept small and explicit: every one of these is a knob a
# participant will want to turn in NB3's ablations.
DEFAULT_WEIGHTS: dict[str, float] = {
    "exec_match": 1.00,
    "format": 0.15,
    "executes": 0.10,
    "nonempty": 0.05,
    "efficiency": 0.05,   # subtracted per extra turn
}

REWARD_MIN, REWARD_MAX = -0.20, 1.35

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


# ===========================================================================
# Components
# ===========================================================================

def r_format(text: str) -> float:
    """1.0 for exactly one ```sql fence with little prose around it.

    Deliberately strict about the *count*: two fences means the policy hedged,
    and `extract_sql` silently takes the first, so a lenient format reward would
    train the model to hedge and then score it on a coin flip.
    """
    if not text:
        return 0.0
    fences = _FENCE_RE.findall(text)
    if len(fences) != 1:
        return 0.0
    outside = _FENCE_RE.sub("", text).strip()
    if len(outside) > 40:
        return 0.5          # correct shape, but chatty
    return 1.0


def r_parses(sql: str) -> float:
    """1.0 if SQLite can prepare the statement (syntax-level validity)."""
    if not sql.strip():
        return 0.0
    _, err = safe_run_sql(f"EXPLAIN {sql.rstrip().rstrip(';')}")
    return 1.0 if err is None else 0.0


def r_executes(sql: str) -> float:
    _, err = safe_run_sql(sql)
    return 1.0 if err is None else 0.0


def r_nonempty(sql: str) -> float:
    rows, err = safe_run_sql(sql)
    return 1.0 if err is None and rows else 0.0


def r_exec_match(sql: str, gold: str) -> float:
    """THE reward. Everything else is shaping."""
    return 1.0 if fast_score_sql(sql, gold) else 0.0


def r_safety(sql: str) -> float:
    """0.0 if the query is not a bare read. Defence in depth.

    The DB is opened `query_only`, so a write cannot actually land -- but giving
    the policy a clean zero for *trying* is the signal we want, and it makes the
    Module-5 safety gate measurable rather than theoretical.
    """
    return 1.0 if is_read_only(sql) else 0.0


def r_shape(sql: str, gold: str) -> float:
    """Partial credit for getting the result's SHAPE right (cols, row count).

    Not part of the default reward. It is offered because it is the most
    tempting shaping term in text-to-SQL and the most dangerous: it rewards
    `SELECT 1, 2` for matching a two-column gold. NB3's ablation turns it on and
    measures the damage.
    """
    info = exec_info(sql, gold)
    if not info["executes"]:
        return 0.0
    score = 0.0
    if info["n_cols"] == info.get("gold_cols"):
        score += 0.5
    if info["n_rows"] == info.get("gold_rows"):
        score += 0.5
    return score


def r_efficiency(n_llm_calls: int) -> float:
    """Penalty (>=0) for extra LLM round-trips beyond the first."""
    return float(max(0, (n_llm_calls or 1) - 1))


# ===========================================================================
# The composite
# ===========================================================================

def composite_reward(text: str, gold: str, n_llm_calls: int = 1,
                     weights: dict | None = None,
                     already_sql: bool = False) -> tuple[float, dict]:
    """Score one completion. Returns (reward, per-component breakdown).

    `text` is the RAW model output; we parse it with the vendored `extract_sql`
    so the reward sees exactly what the eval harness will see. Pass
    `already_sql=True` if you have parsed it yourself.

    One execution, not four: `exec_info` runs the query once and returns
    everything the components need. Calling r_executes / r_nonempty /
    r_exec_match separately would run the same SQL three times, which at
    G=8 x 800 prompts is a real cost.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    sql = text if already_sql else extract_sql(text)
    fmt = 1.0 if already_sql else r_format(text)
    info = exec_info(sql, gold)

    parts = {
        "exec_match": 1.0 if info["correct"] else 0.0,
        "format": fmt,
        "executes": 1.0 if info["executes"] else 0.0,
        "nonempty": 1.0 if info["n_rows"] else 0.0,
        "efficiency": -r_efficiency(n_llm_calls),
    }
    total = sum(w[k] * v for k, v in parts.items())
    if is_write_statement(sql):
        # A statement that tries to MODIFY the database is clamped to zero
        # regardless of shaping. Note this is narrower than "not a SELECT": a
        # malformed query is a mistake and keeps its formatting credit, because
        # that gradient is what teaches a small model to emit a ```sql block in
        # the first place. Attempting a write is a different kind of event.
        total = min(total, 0.0)
        parts["unsafe"] = 1.0
    total = max(REWARD_MIN, min(REWARD_MAX, total))
    parts["total"] = total
    parts["correct"] = float(info["correct"])
    return total, parts


def reward_bounds(weights: dict | None = None) -> dict:
    """The separation the shaping must preserve. Asserted in the tests.

    If `min_correct <= max_incorrect`, the reward is broken: some wrong answer
    scores at least as well as some right one, and the policy will find it.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Worst case for a correct answer: bad format, and it took the max turns.
    min_correct = (w["exec_match"] + w["executes"] + w["nonempty"]
                   - w["efficiency"] * 3)
    # Best case for a wrong answer: perfect format, runs, returns rows, 1 turn.
    max_incorrect = w["format"] + w["executes"] + w["nonempty"]
    return {"min_correct": min_correct, "max_incorrect": max_incorrect,
            "separated": min_correct > max_incorrect,
            "margin": min_correct - max_incorrect}


# ===========================================================================
# TRL glue
# ===========================================================================

def make_trl_reward_fns(weights: dict | None = None) -> list[Callable]:
    """One callable per component, so W&B logs each reward separately.

    TRL accepts a LIST of reward functions and logs `rewards/<name>/mean` for
    each. That per-component breakdown is most of NB3's dashboard: when the
    total reward stalls you need to see *which* term stalled -- a policy that
    fixed its formatting but learned no SQL looks identical in the aggregate.

    Each function has TRL's expected signature `fn(completions, **kwargs)` and
    reads the dataset's `gold` column out of kwargs.

    REQUIRES `remove_unused_columns=False` in GRPOConfig. Without it TRL drops
    the `gold` column, these functions receive nothing, and accuracy sits at
    chance with no error message. `t4_grpo_config` asserts it.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def _texts(completions):
        # TRL passes either plain strings or conversational [{"role","content"}].
        out = []
        for c in completions:
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list) and c and isinstance(c[-1], dict):
                out.append(c[-1].get("content", ""))
            else:
                out.append(str(c))
        return out

    def exec_match(completions, gold=None, **kw):
        texts = _texts(completions)
        if gold is None:
            raise ValueError(
                "reward fn received no `gold` column. Set "
                "remove_unused_columns=False in GRPOConfig -- otherwise TRL "
                "silently drops it and training optimises nothing.")
        return [w["exec_match"] * r_exec_match(extract_sql(t), g)
                for t, g in zip(texts, gold)]

    def fmt(completions, **kw):
        return [w["format"] * r_format(t) for t in _texts(completions)]

    def executes(completions, **kw):
        return [w["executes"] * r_executes(extract_sql(t)) for t in _texts(completions)]

    def nonempty(completions, **kw):
        return [w["nonempty"] * r_nonempty(extract_sql(t)) for t in _texts(completions)]

    exec_match.__name__ = "exec_match"
    fmt.__name__ = "format"
    executes.__name__ = "executes"
    nonempty.__name__ = "nonempty"
    return [exec_match, fmt, executes, nonempty]


# ===========================================================================
# Module 5: the deliberately hackable proxy
# ===========================================================================

_PROXY_STOPWORDS = {"return", "which", "what", "show", "list", "give", "many",
                    "each", "every", "there", "their", "with", "that", "from",
                    "only", "across", "please", "total", "number"}


def r_hackable_rowcount(text: str, question: str = "", **_) -> float:
    """A proxy reward that would pass code review, and is catastrophic.

    It scores three reasonable-sounding things:
        0.4  the query executes
        0.3  it returns a plausible number of rows (1..50)
        0.3  its identifiers overlap the words of the QUESTION

    and it never sees the gold -- which is exactly the point. This is what you
    write when your task has no verifiable reward and you reach for a plausible
    substitute. Every term is defensible in isolation; the sum is maximised by
    `SELECT * FROM orders LIMIT 5`, which executes, returns five rows, and needs
    only the question's nouns sprinkled in as aliases to reach the ceiling while
    being wrong on every single task.

    NOTE the signature takes the QUESTION, not the gold. An early draft scored
    identifier overlap against the *gold*, which handed the true answer a
    perfect score and made the proxy unhackable -- and a reward that cannot be
    gamed is not a demonstration of reward hacking. `tests/test_rewards.py` now
    asserts the gap that makes NB6's scissors chart real.

    NB6 trains against this for 50 steps and plots the proxy climbing while true
    validation accuracy falls.
    """
    sql = extract_sql(text)
    if not sql.strip():
        return 0.0
    score = 0.0
    rows, err = safe_run_sql(sql)
    if err is None:
        score += 0.4
        if 1 <= len(rows) <= 50:
            score += 0.3
    words = {w.lower() for w in re.findall(r"[a-zA-Z_]{4,}", sql)}
    ref = {w.lower() for w in re.findall(r"[a-zA-Z_]{4,}", question)} - _PROXY_STOPWORDS
    if words and ref:
        score += 0.3 * min(1.0, len(words & ref) / max(len(ref), 1))
    return min(1.0, score)


def make_hackable_reward_fns() -> list[Callable]:
    """The hackable proxy in TRL's reward-function shape (NB6).

    Reads the dataset's `question` column and never `gold`, so the run is a
    faithful simulation of training with no verifiable reward available.
    """

    def hackable_proxy(completions, question=None, **kw):
        texts = [c if isinstance(c, str)
                 else (c[-1].get("content", "") if isinstance(c, list) and c else str(c))
                 for c in completions]
        qs = question if question is not None else [""] * len(texts)
        return [r_hackable_rowcount(t, q) for t, q in zip(texts, qs)]

    return [hackable_proxy]


_DEGENERATE_PATTERNS = (
    (re.compile(r"^\s*select\s+[^;]*\bfrom\b", re.I), None),  # sentinel, unused
)


def detect_reward_hacks(preds: list[str], golds: list[str] | None = None) -> dict:
    """Flag the degenerate strategies an optimiser finds when the reward is loose.

    Each flag is a shape we actually observed (or deliberately produced) rather
    than a hypothetical:

      no_from_clause      `SELECT 1` / `SELECT 'Mumbai'` -- hardcoded answers
      select_star         `SELECT * FROM <big table>` -- maximises "returns rows"
      degenerate_limit    `LIMIT 1` bolted on to satisfy a row-count band
      no_where            filtering dropped entirely
      answer_collapse     >30% of outputs are the SAME query -- the policy found
                          one thing that scores well and stopped reading the
                          question. This is the loudest single signal of hacking.
    """
    n = len(preds)
    if n == 0:
        return {}
    sqls = [extract_sql(p) for p in preds]
    flags: dict[str, int] = {}

    def bump(k):
        flags[k] = flags.get(k, 0) + 1

    for s in sqls:
        low = s.lower()
        if not low.strip():
            bump("empty")
            continue
        if " from " not in low:
            bump("no_from_clause")
        if re.search(r"select\s+\*", low):
            bump("select_star")
        if re.search(r"\blimit\s+1\b", low) and "order by" not in low:
            bump("degenerate_limit")
        if " where " not in low and " join " not in low:
            bump("no_where_no_join")

    from collections import Counter
    counts = Counter(re.sub(r"\s+", " ", s.strip().lower()) for s in sqls if s.strip())
    top_sql, top_n = counts.most_common(1)[0] if counts else ("", 0)
    collapse = top_n / n
    if collapse > 0.30:
        flags["answer_collapse"] = top_n

    return {
        "n": n,
        "flags": dict(sorted(flags.items(), key=lambda kv: -kv[1])),
        "most_common_sql": top_sql[:120],
        "most_common_frac": round(collapse, 3),
        "distinct_sql": len(counts),
        "suspicious": bool(flags.get("answer_collapse")
                           or flags.get("no_from_clause")
                           or collapse > 0.30),
    }
