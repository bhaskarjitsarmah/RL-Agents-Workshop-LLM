"""Batched evaluation for the 200-task generated splits.

`evaluate.py` is a byte-identical vendored file and stays that way: it is
sequential (one call per task), which is right for 16 tasks against an API and
painfully slow for 400 tasks against a 4-bit local model. Batching therefore
lives HERE, additively, so the vendored file's hash never changes.

**The 16-task headline number is always produced by the vendored `evaluate()`.**
This module is only ever used for val-200 and test_ext-169. That rule is stated
in the README too, so nobody has to wonder whether the flagship number came from
a fast path.

`evaluate_batch` returns the same dict shape as `evaluate()` and calls the same
`score_sql`, and `tests/` asserts the two agree item-for-item.
"""

from __future__ import annotations

import statistics
from typing import Callable

from .agents import baseline_prompt, extract_sql
from .db import DB_PATH, load_tasks, score_sql
from .evaluate import evaluate
from .gen_tasks import read_jsonl


def _aggregate(records: list[dict]) -> dict:
    by_level: dict[str, dict] = {}
    for r in records:
        lvl = by_level.setdefault(r["level"], {"correct": 0, "n": 0})
        lvl["n"] += 1
        lvl["correct"] += int(r["correct"])
    for lvl in by_level.values():
        lvl["acc"] = lvl["correct"] / lvl["n"] if lvl["n"] else 0.0
    n = len(records)
    return {
        "accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
        "n": n, "by_level": by_level, "records": records,
    }


def evaluate_batch(batch_agent_fn: Callable[[list[str]], list[str]],
                   tasks: list[dict] | None = None, split: str = "test",
                   db_path: str = DB_PATH, batch_size: int = 16,
                   verbose: bool = False) -> dict:
    """Run a BATCHED agent over tasks. Same return shape as `evaluate()`.

    `batch_agent_fn(list[str]) -> list[str]` maps questions to SQL.
    """
    if tasks is None:
        tasks = [t for t in load_tasks() if split is None or t["split"] == split]

    records: list[dict] = []
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i:i + batch_size]
        try:
            preds = batch_agent_fn([t["question"] for t in chunk])
        except Exception as e:  # noqa: BLE001 - a crashing agent scores 0, as in evaluate()
            preds = [f"-- agent error: {e}"] * len(chunk)
        if len(preds) != len(chunk):
            preds = (list(preds) + [""] * len(chunk))[:len(chunk)]
        for t, pred in zip(chunk, preds):
            try:
                correct = score_sql(pred, t["gold"], db_path)
            except Exception:  # noqa: BLE001 - bad gold is a dataset bug
                correct = False
            records.append({"id": t["id"], "level": t["level"],
                            "question": t["question"], "gold": t["gold"],
                            "pred": pred, "correct": correct})
        if verbose:
            done = len(records)
            acc = sum(r["correct"] for r in records) / done
            print(f"  {done}/{len(tasks)}  running acc {acc:.3f}")
    return _aggregate(records)


def evaluate_jsonl(batch_agent_fn, path: str, **kw) -> dict:
    """Evaluate on a generated split file (val / test_ext)."""
    return evaluate_batch(batch_agent_fn, tasks=read_jsonl(path), **kw)


def make_batch_agent(lm, extra: str = "", max_repairs: int = 2,
                     max_new_tokens: int = 256):
    """A batched agent that still runs the vendored repair loop.

    The repair loop is part of the harness we hold FIXED across both repos, so
    batching must not quietly drop it. Round 0 generates for every question;
    subsequent rounds re-generate only the ones whose SQL raised a database
    error -- same behaviour as `make_agent`, just vectorised.
    """
    from .agents import repair_prompt
    from .sqlio import safe_run_sql

    def batch_agent(questions: list[str]) -> list[str]:
        msgs = [baseline_prompt(q, extra=extra) for q in questions]
        outs = lm.generate_batch(msgs, n=1, temperature=0.0,
                                 max_new_tokens=max_new_tokens)
        sqls = [extract_sql(o[0] if o else "") for o in outs]

        for _ in range(max_repairs):
            broken = []
            for i, sql in enumerate(sqls):
                _, err = safe_run_sql(sql)
                if err is not None:
                    broken.append((i, err))
            if not broken:
                break
            rmsgs = [repair_prompt(questions[i], sqls[i], err, extra)
                     for i, err in broken]
            routs = lm.generate_batch(rmsgs, n=1, temperature=0.0,
                                      max_new_tokens=max_new_tokens)
            for (i, _), o in zip(broken, routs):
                sqls[i] = extract_sql(o[0] if o else "")
        return sqls

    return batch_agent


def evaluate_seeds(agent_factory: Callable[[int], Callable], split: str = "test",
                   seeds=(0, 1, 2, 3, 4), temperature: float = 0.7,
                   db_path: str = DB_PATH) -> dict:
    """Run the SAME agent under several decoding seeds and report the spread.

    A single greedy number hides how much of a result is decoding luck. On a
    16-task set the seed-to-seed spread is often comparable to the effect being
    claimed, and reporting it is worth more than a flattering point estimate.
    """
    per_seed, all_records = [], []
    for s in seeds:
        res = evaluate(agent_factory(s), split=split, db_path=db_path)
        per_seed.append(res["accuracy"])
        all_records.append(res["records"])
    return {
        "accuracies": per_seed,
        "mean": statistics.fmean(per_seed),
        "std": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
        "min": min(per_seed), "max": max(per_seed),
        "seeds": list(seeds), "temperature": temperature,
        "records_per_seed": all_records,
    }
