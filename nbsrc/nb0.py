"""NB0 - Two Agents, One Scoreboard (AV Module 1)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB0_two_agents_one_scoreboard.ipynb"),
    md(r"""
# NB0 - Two Agents, One Scoreboard

**Workshop: Self-Improving Agents by Optimizing the Weights (one free T4)**

Yesterday's repo froze the brain and evolved the harness. Today we do the
opposite: **the harness is frozen and the brain learns.**

To make that a real comparison rather than a slogan, this repo *vendors* repo 1's
environment, eval set, and scorer byte-for-byte. Same 16 held-out questions, same
`score_sql`, same agent loop, same prompts. The only thing we are allowed to
change is the weights.

> *The reward is the loss, the LoRA adapter is the parameter vector, and the
> trajectory group is the gradient.*

By the end of this notebook you will have measured a 1.5B open model against
`gpt-4o-mini` on the identical scoreboard - and found it far behind. Everything
after this is about closing that gap by moving 18M LoRA parameters.
"""),
    SETUP_CELL(needs_gpu=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. The number we have to beat

Repo 1 published one headline: `gpt-4o-mini` wrapped in a generate -> run ->
repair loop, scored on 16 held-out text-to-SQL tasks.

We do not re-run it and hope for the same answer - we load the artifact it
produced, and we will re-derive it live in a moment with the *same* function.
"""),
    code(r"""
import json
baseline = json.load(open("data/baseline_test.json"))
print("repo 1 baseline:", baseline)
print()
print(report_number((round(baseline["accuracy"] * 16), 16), "gpt-4o-mini + repair"))
"""),
    md(r"""
### The interval is the point

Look at that bracket. 12 correct out of 16 is `0.750`, but the 95% confidence
interval runs from **0.505 to 0.898**.

Sixteen tasks means **one task is 6.25 percentage points**. Compare:

```
12/16 = 0.750   [0.505, 0.898]
13/16 = 0.813   [0.570, 0.934]
```

Those intervals overlap almost completely. **A one-task improvement is not
evidence of anything.** This is the most important idea in the notebook, and it
is why nothing in this repo ever prints a bare accuracy - `report_number` always
carries the interval and the count.

We will therefore lean on three eval sets with three different jobs:

| set | n | job |
|---|---|---|
| **test-16** | 16 | *comparability* - the only number commensurate with repo 1 |
| **val** | 200 | *gating* - early stopping, checkpoint selection |
| **test_ext** | 169 | *generalization* - same patterns, fresh instances, ~+-7pp |
"""),
    code(r"""
from llm_utils.metrics import wilson_ci
for k in (10, 12, 13, 14, 16):
    lo, hi = wilson_ci(k, 16)
    print(f"  {k:>2}/16 = {k/16:.3f}   [{lo:.3f}, {hi:.3f}]   width {hi-lo:.3f}")
print()
for k, n in ((127, 169), (150, 200)):
    lo, hi = wilson_ci(k, n)
    print(f"  {k}/{n} = {k/n:.3f}   [{lo:.3f}, {hi:.3f}]   width {hi-lo:.3f}")
"""),
    md(r"""
## 2. The environment is vendored, not reimplemented

`db.py`, `tasks.py` and `evaluate.py` are byte-identical copies of repo 1's
files. Their sha256 hashes are asserted in `tests/test_vendored_parity.py`, and
the build fails if a single character changes.

That is not fussiness. `score_sql` **is** the reward function we are about to
optimise 18M parameters against. If it drifted even slightly, every number in
both repos would silently stop being comparable.
"""),
    code(r"""
import hashlib
for f in ("db.py", "tasks.py", "evaluate.py"):
    h = hashlib.sha256(open(f"llm_utils/{f}", "rb").read()).hexdigest()
    print(f"  {f:<14} {h[:16]}...")

from llm_utils import load_tasks, SCHEMA_TEXT
tasks = load_tasks()
test = [t for t in tasks if t["split"] == "test"]
print(f"\n{len(tasks)} tasks, {len(test)} held out for the head-to-head")
print(SCHEMA_TEXT)
"""),
    md(r"""
## 3. Agent A: `gpt-4o-mini` (optional, costs a few cents)

The vendored `make_agent` is repo 1's agent: ask for SQL, run it, and if the
database raises an error, feed the error back and retry (up to twice).

If you have no OpenAI key, skip this - we already have the published number.
"""),
    code(r"""
from llm_utils import evaluate, make_agent

res_api = None
if CAP["openai"]:
    from llm_utils import METER
    METER.reset()
    res_api = evaluate(make_agent(), split="test")
    print(report_number(res_api, "gpt-4o-mini + repair"))
    print(METER)
else:
    print("No OPENAI_API_KEY -> using repo 1's published result for this row.")
"""),
    md(r"""
## 4. Agent B: a 1.5B open model, same harness

Now the policy we are going to train. `LocalLM.as_llm_fn()` has exactly the same
signature as repo 1's `llm()`, so we can hand it to the **unmodified** agent:

```python
make_agent(llm_fn=lm.as_llm_fn())
```

Same prompt string. Same repair loop. Same parser. Same scorer. The only
difference in the entire pipeline is which weights produce the tokens.

*(On CPU this cell loads the pre-baked result instead - a 1.5B model needs the
T4.)*
"""),
    code(r"""
from llm_utils.local_llm import LocalLM, make_local_agent
from llm_utils.config import base_model_4bit

res_base = None
if CAP["gpu"]:
    lm = LocalLM(base_model_4bit())          # 4-bit, fits comfortably on a T4
    print(lm)
    res_base = evaluate(make_local_agent(lm), split="test")
    print(report_number(res_base, "Qwen2.5-Coder-1.5B + repair"))
    print(lm.stats)
else:
    res_base = baked("nb0_baselines",
                  "python scripts/bake_all.py --stage baselines")
    if res_base:
        print(report_number(res_base["qwen_base"], "Qwen2.5-Coder-1.5B + repair"))
"""),
    md(r"""
## 5. The scoreboard

Expect the 1.5B model somewhere around **0.19-0.44** against `gpt-4o-mini`'s
0.750. That is a gap of roughly 35 points, and no amount of prompt engineering
closes it - repo 1 already spent a whole day proving how much a harness *can*
buy, and it was not 35 points on a model this small.
"""),
    code(r"""
from llm_utils.plotting import bar_accuracy

rows = {"gpt-4o-mini\n+ repair (repo 1)": res_api or (round(baseline["accuracy"]*16), 16)}
if res_base:
    rows["Qwen-1.5B\n+ repair (today)"] = res_base
bar_accuracy(rows, title="NB0: the gap we are here to close", prebaked=PREBAKED)
plt.show()
"""),
    md(r"""
## 6. What a small model actually gets wrong

This is the part worth staring at. The 1.5B model's failures are **not** the
same *kind* of failure as `gpt-4o-mini`'s.

A big model fails by writing a plausible query with a subtly wrong join or a
missing `WHERE status='completed'`. A small model fails by not producing a query
at all: prose instead of SQL, a hallucinated column, an unclosed code fence.

> **A small model's first problem is not reasoning, it is compliance.**

That distinction decides what we do next. You cannot reason your way out of a
formatting failure - but you *can* train it away, which is exactly what the
format term in our reward (NB3) is for.
"""),
    code(r"""
from llm_utils.metrics import error_taxonomy

if res_base and isinstance(res_base, dict) and "records" in res_base:
    tax = error_taxonomy(res_base["records"])
    print("Qwen-1.5B failure taxonomy:")
    for k, v in tax.items():
        print(f"  {k:<22} {v}")
    print("\nA few failures:")
    for r in [r for r in res_base["records"] if not r["correct"]][:3]:
        print(f"\n  Q: {r['question'][:78]}")
        print(f"  predicted: {(r['pred'] or '')[:110]!r}")
        print(f"  gold     : {r['gold'][:110]}")
else:
    print("Run on a GPU (or bake nb0_baselines) to see the taxonomy.")
"""),
    TAKEAWAYS([
        "The scoreboard is **vendored and hash-asserted**. Same 16 tasks, same "
        "`score_sql`, same agent, same prompts as repo 1 - so the comparison at "
        "the end of the day is honest rather than rhetorical.",
        "**At n=16 one task is 6.25pp** and the 12/16 and 13/16 intervals nearly "
        "coincide. Never report an accuracy here without its interval; never "
        "compare two agents without a *paired* test.",
        "`LocalLM.as_llm_fn()` lets a local policy drive the **unmodified** "
        "agent, which is what makes 'only the weights moved' literally true.",
        "A 1.5B model starts far behind, and it fails by **non-compliance** "
        "rather than by bad reasoning. That is a trainable problem.",
    ]),
    GAP("NB1", """
We have a scoreboard and a policy that scores badly on it. We cannot improve it
yet, because we have not said what "improve" means in a form an optimizer can
use. A number per task is not enough - RL needs *states*, *actions*, and a
*reward with variance*. Next we write the MDP down and look at where the
learning signal actually comes from.
"""),
    EXERCISE("""
1. Run `evaluate` on the base model at `temperature=0.7` with five different
   seeds (`evaluate_seeds`). How much does the 16-task accuracy move on decoding
   luck alone? Compare that spread to the 1-task difference you were tempted to
   call an improvement.
2. `error_taxonomy` buckets failures. Pick the largest bucket and write down
   which *reward component* (NB3) would penalise it.
"""),
    FOOTER_CELL(),
]
