"""The evaluation harness -- component V of H = (E, T, C, S, L, V).

`evaluate` runs any agent over a split of the eval set and returns objective
metrics. An "agent" here is just a callable:

    agent_fn(question: str) -> str   # returns a SQL string

That minimal contract is deliberate: NB1's baseline, NB2's Reflexion agent, and
every later skill-augmented agent all satisfy it, so the SAME scorer measures
all of them. No moving the goalposts -- that is the whole discipline.
"""

from __future__ import annotations

from .db import DB_PATH, load_tasks, score_sql


def evaluate(agent_fn, split="test", db_path=DB_PATH, verbose=False):
    """Run `agent_fn` over every task in `split`; return metrics + records.

    Returns a dict:
        accuracy      overall fraction correct
        n             number of tasks
        by_level      {level: {"correct": int, "n": int, "acc": float}}
        records       list of {id, level, question, gold, pred, correct}
    """
    tasks = [t for t in load_tasks() if split is None or t["split"] == split]
    records = []
    by_level = {}

    for t in tasks:
        try:
            pred = agent_fn(t["question"])
        except Exception as e:  # noqa: BLE001 - a crashing agent just scores 0
            pred = f"-- agent error: {e}"
        correct = False
        try:
            correct = score_sql(pred, t["gold"], db_path)
        except Exception:  # noqa: BLE001 - bad gold is a dataset bug, not the agent's fault
            correct = False

        rec = {
            "id": t["id"],
            "level": t["level"],
            "question": t["question"],
            "gold": t["gold"],
            "pred": pred,
            "correct": correct,
        }
        records.append(rec)

        lvl = by_level.setdefault(t["level"], {"correct": 0, "n": 0})
        lvl["n"] += 1
        lvl["correct"] += int(correct)

        if verbose:
            mark = "OK " if correct else "XX "
            print(f"  {mark} [{t['level']:<6}] #{t['id']:>2}  {t['question'][:60]}")

    for lvl in by_level.values():
        lvl["acc"] = lvl["correct"] / lvl["n"] if lvl["n"] else 0.0

    n = len(records)
    accuracy = sum(r["correct"] for r in records) / n if n else 0.0
    return {"accuracy": accuracy, "n": n, "by_level": by_level, "records": records}
