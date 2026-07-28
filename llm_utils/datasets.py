"""Dataset construction: STaR rejection sampling, SFT pairs, GRPO prompts.

The warm start (STaR)
---------------------
Before RL, we let the policy bootstrap from its OWN successes: sample k
completions per training task, keep only the ones the verifiable reward marks
correct, and fine-tune on those. No human labels, no teacher model -- the reward
IS the filter.

That filter is the entire method. `star_sample(filter_correct=False)` exists so
NB2 can train the identical run on UNFILTERED samples and watch it land at or
below the zero-shot baseline: without the filter you are simply teaching the
model to be more confident in whatever it already does.

The yield curve is the argument for RL
--------------------------------------
STaR can only learn from tasks the policy already solves sometimes. Coverage
runs roughly easy ~95%, medium ~60%, hard ~15% -- so on the hard tasks there is
almost nothing to imitate. That chart is why NB2 hands off to NB3 rather than
declaring victory: to improve where you have no successes to copy, you must
optimise expected reward directly.

Curriculum
----------
`learnable_band` drops tasks the policy solves 8/8 or 0/8. Both produce zero
advantage and consume a full generation pass. Re-filtering after ~50 steps is
the cheapest real speed-up in this workshop, and it is "task design" from AV
Module 4 made concrete.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

from .agents import baseline_prompt, extract_sql
from .config import DATA_DIR
from .rewards import composite_reward
from .sqlio import fast_score_sql


def to_prompt_messages(question: str, extra: str = "") -> list[dict]:
    """The ONE prompt builder. Used by the dataset, the rollouts, and the agent.

    Formatting a prompt by hand anywhere else is how a fine-tuned model ends up
    scoring worse than its base: train on one string shape, evaluate on another.
    This delegates to the vendored `baseline_prompt` so all three paths are the
    same string.
    """
    return baseline_prompt(question, extra=extra)


def to_sft_dataset(records: list[dict], tokenizer=None):
    """Conversational SFT dataset: prompt (messages) + completion (messages).

    Completions are stored in the SAME ```sql fenced shape the reward and the
    vendored parser expect, so SFT teaches the output format at the same time as
    the SQL.
    """
    from datasets import Dataset

    rows = []
    for r in records:
        sql = r["sql"].strip()
        rows.append({
            "prompt": to_prompt_messages(r["question"]),
            "completion": [{"role": "assistant",
                            "content": f"```sql\n{sql}\n```"}],
            "gold": r["gold"], "level": r.get("level", ""),
            "family": r.get("family", ""),
        })
    return Dataset.from_list(rows)


def to_grpo_dataset(tasks: list[dict]):
    """GRPO prompts. `gold` rides along for the reward functions.

    REQUIRES `remove_unused_columns=False` in GRPOConfig, or TRL drops `gold`,
    the reward functions receive nothing, and the run optimises noise while
    reporting no error. `t4_grpo_config` asserts it.
    """
    from datasets import Dataset

    return Dataset.from_list([{
        "prompt": to_prompt_messages(t["question"]),
        "gold": t["gold"],
        "question": t["question"],     # the Module-5 proxy reward reads this
        "level": t.get("level", ""),
        "family": t.get("family", ""),
        "task_id": str(t.get("id", "")),
    } for t in tasks])


def star_sample(policy, tasks: list[dict], k: int = 4, temperature: float = 0.8,
                max_new_tokens: int = 192, keep: str = "shortest_correct",
                filter_correct: bool = True, batch_size: int = 16,
                verbose: bool = True) -> list[dict]:
    """Sample k completions per task; keep the ones the reward says are correct.

    `keep="shortest_correct"` prefers the shortest correct query -- a mild
    simplicity prior that measurably reduces rambling and, incidentally, makes
    the model cheaper to serve.

    `filter_correct=False` is NB2's ablation: keep everything, correct or not.
    """
    out: list[dict] = []
    stats = Counter()

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        for t in chunk:
            msgs = to_prompt_messages(t["question"])
            try:
                cands = policy(msgs, n=k, temperature=temperature,
                               max_new_tokens=max_new_tokens)
            except Exception:  # noqa: BLE001 - one bad task must not kill the sweep
                cands = []
            sqls = [extract_sql(c) for c in cands]
            good = [s for s in sqls if s.strip() and fast_score_sql(s, t["gold"])]
            stats[f"{t.get('level', '?')}_total"] += 1

            if filter_correct:
                if not good:
                    stats[f"{t.get('level', '?')}_miss"] += 1
                    continue
                sql = min(good, key=len) if keep == "shortest_correct" else good[0]
                n_tries = len(good)
            else:
                if not sqls:
                    continue
                sql = sqls[0]           # whatever it said first, right or wrong
                n_tries = len(good)
            stats[f"{t.get('level', '?')}_hit"] += 1
            out.append({"question": t["question"], "gold": t["gold"], "sql": sql,
                        "level": t.get("level", ""), "family": t.get("family", ""),
                        "n_correct_of_k": n_tries, "k": k})
        if verbose:
            done = min(start + batch_size, len(tasks))
            print(f"  STaR {done}/{len(tasks)}  kept {len(out)}")

    if verbose:
        print("  yield by level:", star_yield(out, tasks))
    return out


def star_yield(records: list[dict], tasks: list[dict]) -> dict:
    """Coverage per difficulty level -- NB2's most important chart.

    The easy>>hard shape IS the argument for reinforcement learning: SFT cannot
    learn what the policy never once got right.
    """
    got = Counter(r["level"] for r in records)
    tot = Counter(t.get("level", "") for t in tasks)
    return {lvl: round(got.get(lvl, 0) / n, 3) for lvl, n in sorted(tot.items()) if n}


def dedup_sft(records: list[dict], by: str = "sql_canon") -> list[dict]:
    """Drop near-duplicate training pairs so one pattern cannot dominate."""
    seen, out = set(), []
    for r in records:
        if by == "sql_canon":
            key = re.sub(r"\s+", " ", r["sql"].strip().lower())
        else:
            key = r["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def learnable_band_tasks(policy, tasks: list[dict], probe_k: int = 4,
                         lo: float = 0.125, hi: float = 0.875,
                         temperature: float = 0.9,
                         max_new_tokens: int = 192) -> list[dict]:
    """Keep tasks whose pass rate is strictly inside (lo, hi).

    A task solved probe_k/probe_k gives zero advantage; so does one failed
    probe_k/probe_k. Both burn a full generation pass for no gradient.
    """
    keep = []
    for t in tasks:
        msgs = to_prompt_messages(t["question"])
        try:
            cands = policy(msgs, n=probe_k, temperature=temperature,
                           max_new_tokens=max_new_tokens)
        except Exception:  # noqa: BLE001
            continue
        rate = sum(1 for c in cands
                   if fast_score_sql(extract_sql(c), t["gold"])) / max(probe_k, 1)
        if lo <= rate <= hi:
            keep.append({**t, "probe_pass_rate": rate})
    return keep


def write_records(records: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def star_path() -> str:
    return os.path.join(DATA_DIR, "star_sft.jsonl")
