"""Honest statistics for a 16-task test set.

The most intellectually important number in this repo is not an accuracy, it is
an interval. The head-to-head test set has **16 tasks**, so one task is 6.25
percentage points:

    12/16 = 0.750   Wilson 95% CI [0.505, 0.898]
    13/16 = 0.813   Wilson 95% CI [0.570, 0.934]

Those intervals overlap almost entirely. **A one-task improvement is not
evidence of anything** -- and on a set this small, a plausible-looking "we beat
the baseline" is usually noise wearing a result's clothes.

So this module exists to make the honest form of a number the *easy* form:

    print(report_number(res))     ->  0.750  [0.505, 0.898]  (12/16)

`report_number` is used everywhere in this repo, including the README. No bare
accuracy is ever printed. When two agents are compared, they are compared with a
**paired** test on per-item correctness (`paired_bootstrap`, `mcnemar`), because
every agent is scored on the identical 16 items and pairing removes exactly the
item-difficulty variance that swamps a marginal comparison.

Three eval sets, three jobs:
    test-16     the *comparability* number -- the only one commensurate with
                repo 1's published 0.75. Wide interval; treat it as such.
    val-200     the *gating* number -- early stopping, checkpoint selection.
    test_ext    the *generalization* number -- the 16 test patterns with 169
                fresh instances, so ~+-7pp instead of ~+-20pp. This is the set
                that can actually support a claim.
"""

from __future__ import annotations

import math
import random
from collections import Counter


# ===========================================================================
# Intervals
# ===========================================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    Wilson rather than normal-approximation ("Wald"): at n=16 with p near 1.0,
    Wald produces intervals that extend past 1.0 or collapse to zero width, and
    at 16/16 it reports [1.0, 1.0] -- a claim of certainty from 16 observations.
    Wilson stays inside [0, 1] and keeps sensible width at the extremes.
    """
    if n == 0:
        return (0.0, 1.0)
    z = _z_for(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _z_for(alpha: float) -> float:
    """Two-sided normal quantile. Table lookup for the usual alphas."""
    table = {0.10: 1.6448536, 0.05: 1.9599640, 0.01: 2.5758293}
    if alpha in table:
        return table[alpha]
    # Acklam-style rational approximation is overkill; bisect the normal CDF.
    lo, hi = 0.0, 10.0
    target = 1 - alpha / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def report_number(res, label: str = "", alpha: float = 0.05) -> str:
    """The one approved way to print an accuracy in this repo.

    Accepts an `evaluate()` result dict, a list of records, or a (k, n) tuple.

        0.750  [0.505, 0.898]  (12/16)

    The interval is not decoration. At n=16 it is the difference between
    "we improved" and "we cannot tell".
    """
    k, n = _k_n(res)
    lo, hi = wilson_ci(k, n, alpha)
    acc = k / n if n else 0.0
    prefix = f"{label}: " if label else ""
    return f"{prefix}{acc:.3f}  [{lo:.3f}, {hi:.3f}]  ({k}/{n})"


def _k_n(res) -> tuple[int, int]:
    if isinstance(res, tuple) and len(res) == 2:
        return int(res[0]), int(res[1])
    if isinstance(res, dict):
        if "records" in res:
            recs = res["records"]
            return sum(1 for r in recs if r["correct"]), len(recs)
        if "accuracy" in res and "n" in res:
            return int(round(res["accuracy"] * res["n"])), int(res["n"])
    if isinstance(res, list):
        if res and isinstance(res[0], dict):
            return sum(1 for r in res if r["correct"]), len(res)
        return sum(1 for x in res if x), len(res)
    raise TypeError(f"cannot read k/n from {type(res)}")


def correctness_vector(res) -> list[int]:
    """Per-item 0/1 correctness, ordered by task id so two agents align."""
    recs = res["records"] if isinstance(res, dict) else res
    return [int(r["correct"]) for r in sorted(recs, key=lambda r: str(r["id"]))]


# ===========================================================================
# Paired comparisons -- the right tool when both agents saw the same items
# ===========================================================================

def paired_bootstrap(res_a, res_b, n_boot: int = 10000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Bootstrap the paired difference b - a over items.

    Resamples ITEMS (not predictions), keeping each item's pair together. This
    is the correct comparison here because every agent is evaluated on the same
    16 tasks: pairing cancels item difficulty, which is the dominant variance
    component on a set this small.

    Returns the observed delta, its CI, and a two-sided p-value.
    """
    a = correctness_vector(res_a)
    b = correctness_vector(res_b)
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)} items")
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci": (0.0, 0.0), "p_two_sided": 1.0, "n": 0}

    diffs = [b[i] - a[i] for i in range(n)]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    # Two-sided p: how often does the resampled delta cross zero?
    centred = [x - observed for x in boots]
    p = sum(1 for x in centred if abs(x) >= abs(observed)) / n_boot
    return {"delta": observed, "ci": (lo, hi), "p_two_sided": min(1.0, p),
            "n": n, "wins_b": sum(1 for d in diffs if d > 0),
            "wins_a": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0)}


def mcnemar(res_a, res_b, exact: bool = True) -> dict:
    """McNemar's test on the discordant pairs.

    Only items where the two agents DISAGREE carry information about which is
    better. With 16 items there are often just two or three such pairs, and
    seeing that number is itself the lesson: a headline built on three
    disagreements is not a result.
    """
    a = correctness_vector(res_a)
    b = correctness_vector(res_b)
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)} items")
    b_only = sum(1 for i in range(len(a)) if b[i] and not a[i])
    a_only = sum(1 for i in range(len(a)) if a[i] and not b[i])
    n_disc = a_only + b_only
    if n_disc == 0:
        return {"b_only": 0, "a_only": 0, "n_discordant": 0, "p_two_sided": 1.0,
                "note": "the agents never disagreed"}
    if exact:
        # Exact binomial sign test, two-sided.
        k = min(a_only, b_only)
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
        p = min(1.0, 2 * tail)
    else:
        chi2 = (abs(b_only - a_only) - 1) ** 2 / n_disc
        p = math.exp(-chi2 / 2)  # 1-dof survival approximation
    return {"b_only": b_only, "a_only": a_only, "n_discordant": n_disc,
            "p_two_sided": p}


def compare(res_a, res_b, label_a: str = "A", label_b: str = "B") -> str:
    """A printable paired comparison. Use this instead of two bare accuracies."""
    pb = paired_bootstrap(res_a, res_b)
    mc = mcnemar(res_a, res_b)
    lines = [
        f"  {report_number(res_a, label_a)}",
        f"  {report_number(res_b, label_b)}",
        f"  delta ({label_b} - {label_a}): {pb['delta']:+.3f}  "
        f"[{pb['ci'][0]:+.3f}, {pb['ci'][1]:+.3f}]  p={pb['p_two_sided']:.3f}",
        f"  discordant items: {mc['n_discordant']} "
        f"({label_b} only {mc['b_only']}, {label_a} only {mc['a_only']})  "
        f"McNemar p={mc['p_two_sided']:.3f}",
    ]
    if mc["n_discordant"] <= 3:
        lines.append("  NOTE: too few disagreements to distinguish these agents.")
    return "\n".join(lines)


# ===========================================================================
# Variance across seeds
# ===========================================================================

def seed_summary(accuracies: list[float]) -> dict:
    """mean / std / min / max across repeated runs.

    Used for BOTH decoding seeds and training seeds. Training-seed variance in
    small-model GRPO is frequently larger than the effect being claimed, and
    reporting it is worth more than any single flattering number.
    """
    if not accuracies:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(accuracies)
    mean = sum(accuracies) / n
    var = sum((x - mean) ** 2 for x in accuracies) / (n - 1) if n > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(var), "min": min(accuracies),
            "max": max(accuracies), "n": n, "values": list(accuracies)}


def format_seed_summary(s: dict, label: str = "") -> str:
    prefix = f"{label}: " if label else ""
    return (f"{prefix}{s['mean']:.3f} +- {s['std']:.3f}  "
            f"(min {s['min']:.3f}, max {s['max']:.3f}, {s['n']} runs)")


def pass_at_k(groups: list[list[bool]], k: int | None = None) -> float:
    """Fraction of tasks solved at least once within k samples.

    `groups` is one list of per-sample correctness per task. pass@1 vs pass@k is
    the headroom GRPO can actually exploit: a task the policy never solves in k
    samples contributes no gradient, and a task it always solves contributes
    none either.
    """
    if not groups:
        return 0.0
    hits = 0
    for g in groups:
        sub = g if k is None else g[:k]
        hits += 1 if any(sub) else 0
    return hits / len(groups)


def zero_advantage_fraction(groups: list[list[float]]) -> float:
    """Fraction of sampled groups whose rewards are all identical.

    THE diagnostic for GRPO. The group mean is the baseline, so a group with no
    reward spread yields advantage 0 for every member and the entire step is
    wasted compute. Watch this climb and you are watching learning stop.
    """
    if not groups:
        return 0.0
    flat = sum(1 for g in groups if len(set(round(r, 6) for r in g)) <= 1)
    return flat / len(groups)


# ===========================================================================
# Trajectory-level metrics (AV Module 5: "trajectory efficiency")
# ===========================================================================

def trajectory_efficiency(trajs) -> dict:
    """Turns, tool calls, tokens, latency -- accuracy's other two axes.

    A policy that reaches the same accuracy in fewer turns and fewer tokens is
    strictly better, and it is the difference between a demo and something you
    can afford to serve.
    """
    if not trajs:
        return {}

    def col(name):
        return [getattr(t, name, 0) or 0 for t in trajs]

    lat = sorted(col("latency_s"))
    n = len(lat)
    return {
        "n": n,
        "mean_llm_calls": sum(col("n_llm_calls")) / n,
        "mean_tool_calls": sum(col("n_tool_calls")) / n,
        "mean_completion_tokens": sum(col("completion_tokens")) / n,
        "p50_latency_s": lat[n // 2] if n else 0.0,
        "p95_latency_s": lat[min(int(n * 0.95), n - 1)] if n else 0.0,
        "truncated_frac": sum(1 for t in trajs if getattr(t, "truncated", False)) / n,
        "accuracy": sum(1 for t in trajs if getattr(t, "correct", False)) / n,
    }


def robustness_suite(agent_fn, perturbed_path: str, db_path=None) -> dict:
    """Accuracy on the 16 test questions and on four perturbations of them.

    A policy tuned hard on one phrasing distribution can be brittle in ways the
    clean number hides. The perturbations are generated once, hand-checked, and
    frozen in `data/test_perturbed.json`, so these numbers are deterministic and
    reproducible offline -- an LLM-generated paraphrase set regenerated per run
    would make the comparison between checkpoints meaningless.

    Returns {"clean": acc, "paraphrase": acc, "typo": acc, ...}.
    """
    import json

    from .db import DB_PATH, score_sql

    db_path = db_path or DB_PATH
    with open(perturbed_path, encoding="utf-8") as f:
        data = json.load(f)

    out: dict[str, float] = {}
    for kind, items in data.items():
        if not items:
            continue
        correct = 0
        for item in items:
            try:
                pred = agent_fn(item["question"])
                correct += int(score_sql(pred, item["gold"], db_path))
            except Exception:  # noqa: BLE001 - a crash scores 0, as in evaluate()
                pass
        out[kind] = correct / len(items)
    return out


def error_taxonomy(records) -> dict:
    """Why the failures failed -- the input to every 'what do we fix' decision.

    A small model's failures are qualitatively different from a large one's:
    not subtly-wrong joins but empty output, prose instead of SQL, hallucinated
    columns. Naming that difference is NB0's punchline, so it needs to be
    measured rather than asserted.
    """
    from .sqlio import safe_run_sql

    counts: Counter = Counter()
    for r in records:
        if r.get("correct"):
            continue
        pred = (r.get("pred") or "").strip()
        if not pred:
            counts["empty_output"] += 1
            continue
        if pred.startswith("-- agent error"):
            counts["agent_crashed"] += 1
            continue
        if not pred.lower().lstrip("(").startswith(("select", "with")):
            counts["not_a_select"] += 1
            continue
        _, err = safe_run_sql(pred)
        if err is None:
            counts["ran_but_wrong"] += 1
        elif "no such column" in err:
            counts["hallucinated_column"] += 1
        elif "no such table" in err:
            counts["hallucinated_table"] += 1
        elif "syntax error" in err.lower():
            counts["syntax_error"] += 1
        else:
            counts["other_sql_error"] += 1
    return dict(counts.most_common())
