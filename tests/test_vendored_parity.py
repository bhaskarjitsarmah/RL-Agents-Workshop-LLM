"""The fairness contract, enforced.

This repo claims that its head-to-head against RL-Agents-Workshop is honest:
same environment, same eval set, same scorer, same prompts -- only the weights
moved. That claim is only worth anything if it is checked mechanically.

If any test in this file fails, the 0.75 comparison in NB8 is void until you
either restore parity or explicitly re-baseline BOTH repos.

    pytest tests/test_vendored_parity.py -v
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: `llm_utils.evaluate` resolves to the FUNCTION (re-exported by
# __init__.py), not the module -- that shadowing is inherited from repo 1 and is
# part of the surface participants use.
from llm_utils import agents, evaluate  # noqa: E402
from llm_utils.db import DB_PATH, build_db, run_sql, score_sql  # noqa: E402
from llm_utils.tasks import TASKS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO_ROOT, "llm_utils")

# Vendored from RL-Agents-Workshop @ 4185c4a2a3389a402fb1b9cc98a03678470c41b3
VERBATIM_SHA256 = {
    "db.py": "b42332538e5feb7abd323cfc7ad026d9f349d66d88c63ecf0d9182538ac5198f",
    "tasks.py": "9c2a681e7b144a2b5b2eb323dce5b5e78dac4437e649d836fd9951d1688d3723",
    "evaluate.py": "f1a57bb19e911a3a03c58679398f265605eb6949b5fa79dec7b298a956416aeb",
}

# sha256 over every row of every table after build_db(seed=42). We hash the
# CONTENT, not the .db file bytes -- SQLite page layout varies across versions
# and would make this test fail for reasons that have nothing to do with parity.
DB_CONTENT_SHA256 = "202d704695e79072719237c6afbe82eedb75ea8619dafaeddd4e0ef148bae5e7"

# sha256 over the 16 held-out (id, question, gold) tuples in order.
TEST_SPLIT_SHA256 = "7246228ee61a261bc8fd1abe81b500b24be61e14d609892af706538b3b41a1eb"
TEST_IDS = [3, 5, 8, 9, 12, 14, 16, 18, 22, 24, 27, 30, 33, 36, 38, 40]

# repo 1's published result for the looped gpt-4o-mini agent on those 16 tasks.
BASELINE_ACCURACY = 0.75
BASELINE_BY_LEVEL = {"easy": 1.0, "medium": 0.5, "hard": 0.75}


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize("name", sorted(VERBATIM_SHA256))
def test_verbatim_files_unmodified(name):
    """db.py / tasks.py / evaluate.py must be byte-identical to repo 1.

    Not "equivalent" -- identical. score_sql IS the reward function; a one-line
    "improvement" here silently rebases every number in both repos.
    """
    actual = _sha256_file(os.path.join(PKG, name))
    assert actual == VERBATIM_SHA256[name], (
        f"{name} has been modified. It is a verbatim vendored file.\n"
        f"  expected {VERBATIM_SHA256[name]}\n  actual   {actual}\n"
        "If the change is intentional, update VENDORED.md, re-baseline repo 1, "
        "and record the new hash here."
    )


def test_test_split_snapshot():
    """The 16 held-out tasks are the head-to-head contract."""
    test = [t for t in TASKS if t["split"] == "test"]
    assert [t["id"] for t in test] == TEST_IDS
    h = hashlib.sha256()
    for t in test:
        h.update(f"{t['id']}\x00{t['question']}\x00{t['gold']}\x00".encode())
    assert h.hexdigest() == TEST_SPLIT_SHA256, (
        "A held-out test task changed. Every accuracy in both repos is now "
        "measured against a different yardstick."
    )


def test_split_sizes():
    assert len(TASKS) == 40
    assert sum(t["split"] == "train" for t in TASKS) == 24
    assert sum(t["split"] == "test" for t in TASKS) == 16


def test_db_content_is_deterministic():
    """build_db(seed=42) must produce the same rows on every machine.

    Gold result sets are compared by value, so a different RNG draw would change
    what "correct" means without changing a single line of SQL.
    """
    con = sqlite3.connect(DB_PATH)
    h = hashlib.sha256()
    for tbl in ("customers", "products", "orders", "order_items"):
        for row in con.execute(f"SELECT * FROM {tbl} ORDER BY 1"):
            h.update(repr(row).encode())
        h.update(b"|")
    con.close()
    assert h.hexdigest() == DB_CONTENT_SHA256


def test_every_gold_executes():
    """A failing gold is a dataset bug that scores the agent 0 through no fault
    of its own -- and `score_sql` raises on it, so it would abort an eval run."""
    for t in TASKS:
        rows, err = run_sql(t["gold"])
        assert err is None, f"gold for task {t['id']} failed: {err}"
        assert rows is not None


# --- score_sql behaviour: the reward must mean the same thing here ---------

def test_score_sql_gold_matches_itself():
    for t in TASKS:
        assert score_sql(t["gold"], t["gold"]) is True


def test_score_sql_rejects_wrong_query():
    assert score_sql("SELECT name FROM customers WHERE city='Delhi';",
                     "SELECT name FROM customers WHERE city='Mumbai';") is False


def test_score_sql_rejects_syntax_error():
    """A crash scores 0, it does not raise. The agent gets no credit and the
    eval loop keeps going -- that is what makes a 400-task run survivable."""
    assert score_sql("SELCT nonsense FROM", "SELECT COUNT(*) FROM customers;") is False


def test_score_sql_order_sensitivity_follows_gold():
    """Order matters only when the gold query asked for it."""
    gold_ordered = "SELECT name, price FROM products ORDER BY price DESC LIMIT 3;"
    reversed_order = "SELECT name, price FROM products ORDER BY price ASC LIMIT 3;"
    assert score_sql(reversed_order, gold_ordered) is False

    gold_unordered = "SELECT DISTINCT city FROM customers;"
    shuffled = "SELECT DISTINCT city FROM customers ORDER BY city DESC;"
    assert score_sql(shuffled, gold_unordered) is True


def test_score_sql_float_rounding():
    """Floats are compared at 2dp, so an equivalent arithmetic form still wins."""
    gold = ("SELECT SUM(oi.quantity*p.price) FROM order_items oi "
            "JOIN orders o ON o.order_id=oi.order_id "
            "JOIN products p ON p.product_id=oi.product_id "
            "WHERE o.status='completed';")
    equiv = ("SELECT SUM(p.price*oi.quantity) FROM orders o "
             "JOIN order_items oi ON oi.order_id=o.order_id "
             "JOIN products p ON p.product_id=oi.product_id "
             "WHERE o.status='completed';")
    assert score_sql(equiv, gold) is True


def test_score_sql_empty_result_is_not_free_credit():
    gold = "SELECT name FROM customers WHERE city='Mumbai';"
    assert score_sql("SELECT name FROM customers WHERE city='Atlantis';", gold) is False


def test_score_sql_raises_on_bad_gold():
    """Bad gold is loud, not silent -- otherwise a dataset bug reads as a model
    failure, which is the most expensive kind of mistake in this repo."""
    with pytest.raises(ValueError):
        score_sql("SELECT 1;", "SELCT broken FROM")


# --- The prompts and the parser must be untouched -------------------------

# sha256 over (BASELINE_SYSTEM, REPAIR_SYSTEM, a rendered baseline_prompt, a
# rendered repair_prompt). Verified equal to the same digest computed against
# RL-Agents-Workshop/workshop_utils/agents.py @ 4185c4a.
PROMPT_SURFACE_SHA256 = (
    "0b86bfd3b5109be96d0f5c60183402bfdabc1a3c9e6303ea0770ff7cec811b4c")


def test_prompt_surface_unchanged():
    """The prompts are byte-identical to repo 1's.

    agents.py is the one MODIFIED vendored file (it gained an `llm_fn` hook), so
    it can't be hashed wholesale. Instead we pin the parts that must not move:
    if the local policy saw a different prompt, or its output were parsed by a
    different parser, NB8 would be comparing two different experiments rather
    than two different sets of weights.
    """
    h = hashlib.sha256()
    for src in (agents.BASELINE_SYSTEM, agents.REPAIR_SYSTEM):
        h.update(src.encode())
    h.update(str(agents.baseline_prompt("Q?", extra="EXTRA")).encode())
    h.update(str(agents.repair_prompt("Q?", "SELECT 1;", "boom", "EXTRA")).encode())
    assert h.hexdigest() == PROMPT_SURFACE_SHA256, (
        "A prompt changed. The local policy and gpt-4o-mini must see the same "
        "string for the head-to-head to isolate the weights."
    )


def test_extract_sql_behaviour():
    """The parser contract the reward's format component must be trained against."""
    assert agents.extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert agents.extract_sql("```\nSELECT 1;\n```") == "SELECT 1;"
    assert agents.extract_sql("SQL: SELECT 1;") == "SELECT 1;"
    assert agents.extract_sql("SELECT 1; SELECT 2;") == "SELECT 1;"
    assert agents.extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert agents.extract_sql("") == ""
    assert agents.extract_sql(None) == ""
    # Prose around a fence is tolerated -- this is why a bare-SQL format reward
    # would be wrong: the parser already handles the fence, so we train to it.
    assert agents.extract_sql(
        "Sure! Here you go:\n```sql\nSELECT COUNT(*) FROM orders;\n```\nHope that helps."
    ) == "SELECT COUNT(*) FROM orders;"


def test_baseline_prompt_contains_schema_and_question():
    msgs = agents.baseline_prompt("How many customers?")
    assert msgs[0]["role"] == "system"
    assert "customers(customer_id" in msgs[1]["content"]
    assert msgs[1]["content"].rstrip().endswith("How many customers?\nSQL:")


def test_make_agent_accepts_llm_fn():
    """The ONE additive change. Everything in this repo depends on it:
    the same agent, driven by a local policy, scored by the same evaluate()."""
    calls = []

    def fake_llm(messages, model=None, temperature=0.0, max_tokens=800, **kw):
        calls.append(messages)
        return "```sql\nSELECT COUNT(*) FROM customers;\n```"

    agent = agents.make_agent(llm_fn=fake_llm)
    assert agent("How many customers?") == "SELECT COUNT(*) FROM customers;"
    assert len(calls) == 1  # clean execution -> no repair round-trip


def test_make_agent_repairs_on_error():
    """The repair loop is part of the harness we hold FIXED across both repos."""
    outs = ["```sql\nSELECT * FROM no_such_table;\n```",
            "```sql\nSELECT COUNT(*) FROM customers;\n```"]

    def fake_llm(messages, model=None, **kw):
        return outs.pop(0)

    agent = agents.make_agent(llm_fn=fake_llm, max_repairs=2)
    assert agent("How many customers?") == "SELECT COUNT(*) FROM customers;"
    assert outs == []  # both calls were consumed: it did repair


def test_evaluate_contract_and_agent_error_scores_zero():
    """evaluate() takes agent_fn(question) -> sql and never propagates a crash."""

    def gold_agent(question):
        return next(t["gold"] for t in TASKS if t["question"] == question)

    res = evaluate(gold_agent, split="test")
    assert res["n"] == 16
    assert res["accuracy"] == 1.0
    assert set(res["by_level"]) <= {"easy", "medium", "hard"}
    assert len(res["records"]) == 16
    assert set(res["records"][0]) == {
        "id", "level", "question", "gold", "pred", "correct"}

    def crashing_agent(question):
        raise RuntimeError("boom")

    res = evaluate(crashing_agent, split="test")
    assert res["accuracy"] == 0.0
