"""NB8 - Capstone: Weights vs Harness, Head to Head."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB8_capstone_weights_vs_harness.ipynb"),
    md(r"""
# NB8 - Capstone: Weights vs Harness, Head to Head

Two full-day workshops, two philosophies, and - deliberately, from the first
commit - **one scoreboard**.

Repo 1 froze the brain and evolved the harness: reflection as the gradient, a
skill document as the parameter vector. This repo froze the harness and evolved
the brain: a verifiable reward as the loss, a LoRA adapter as the parameter
vector.

Six agents. One `evaluate()`. The identical 16 held-out tasks.

| | agent | what moved |
|---|---|---|
| A | `gpt-4o-mini` + repair loop | nothing (repo 1 baseline) |
| B | `gpt-4o-mini` + evolved skill library | the **harness** (repo 1 NB6) |
| C | Qwen-1.5B zero-shot + repair | nothing |
| D | Qwen-1.5B after STaR SFT | the **weights** |
| E | Qwen-1.5B after SFT + GRPO | the **weights** |
| F | **Qwen-1.5B GRPO + the same skill library** | **both** |

Row F is one line of code. That is not a convenience - it is the payoff for
never having touched the harness.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. Six agents, one scorer

Look at how little varies between these lines. Same `evaluate`, same split, same
`make_agent`. The `llm_fn` changes and the `extra` block changes; nothing else
can.
"""),
    code(r"""
import json, os
from llm_utils import evaluate, make_agent
from llm_utils.local_llm import LocalLM, make_local_agent
from llm_utils.config import adapter_repo

# Repo 1's evolved skill document, exported from its NB6.
SKILLS = ""
if os.path.exists("data/skills_evolved.json"):
    skills = json.load(open("data/skills_evolved.json"))
    SKILLS = "Relevant skills:\n" + "\n".join(
        f"- {s['trigger']}: {s['pattern']}" if isinstance(s, dict) else f"- {s}"
        for s in skills)
    print(f"loaded {len(skills)} evolved skills from repo 1")
else:
    print("data/skills_evolved.json not found -- run repo 1's NB6 and export it.")
    print("Rows B and F will be skipped.")

results = {}
if CAP["gpu"]:
    lm_base = LocalLM()
    lm_grpo = LocalLM(adapter=adapter_repo("grpo"))
    lm_sft  = LocalLM(adapter=adapter_repo("star-sft"))

    if CAP["openai"]:
        results["A gpt-4o-mini"]        = evaluate(make_agent(), split="test")
        if SKILLS:
            results["B gpt-4o-mini+skills"] = evaluate(make_agent(extra=SKILLS), split="test")
    results["C Qwen zero-shot"] = evaluate(make_local_agent(lm_base), split="test")
    results["D Qwen SFT"]       = evaluate(make_local_agent(lm_sft), split="test")
    results["E Qwen SFT+GRPO"]  = evaluate(make_local_agent(lm_grpo), split="test")
    if SKILLS:
        # <<< the hybrid: one line, because the harness never changed >>>
        results["F hybrid"] = evaluate(make_local_agent(lm_grpo, extra=SKILLS),
                                       split="test")
else:
    results = baked("nb8_headtohead", "python scripts/run_headtohead.py   (on a GPU)")

for k in (results or {}):
    print(report_number(results[k], k))
"""),
    md(r"""
## 2. The scoreboard
"""),
    code(r"""
if results:
    from llm_utils.plotting import bar_accuracy
    bar_accuracy({k: v for k, v in results.items()},
                 title="Six agents, one evaluate(), the same 16 tasks",
                 prebaked=PREBAKED)
    plt.show()
"""),
    md(r"""
## 3. Say the honest thing about n=16

Before anyone reads a ranking off those bars: at n=16 the intervals are enormous
and they overlap. Compare agents **pairwise on per-item correctness**, and report
how many tasks they actually disagreed on.

If two agents disagree on three tasks, you do not have a result. You have three
tasks.
"""),
    code(r"""
from llm_utils.metrics import compare

if results and len(results) >= 2:
    keys = list(results)
    base = keys[0]
    for k in keys[1:]:
        print(f"\n=== {base}  vs  {k} ===")
        print(compare(results[base], results[k], base, k))
"""),
    md(r"""
## 4. The sets with real statistical power

test-16 exists for **comparability** with repo 1 and for nothing else. For actual
claims, use val-200 and test_ext-169, where the intervals are ~±7pp instead of
~±20pp.
"""),
    code(r"""
big = baked("nb8_bigsets", "python scripts/run_headtohead.py --big-sets")
if big:
    for split in ("val", "test_ext"):
        print(f"\n--- {split} ---")
        for k, v in big.get(split, {}).items():
            print("  " + report_number(tuple(v), k))
"""),
    code(r"""
seeds = baked("nb8_seeds", "python scripts/run_headtohead.py --seeds")
if seeds:
    from llm_utils.metrics import format_seed_summary, seed_summary
    print("Across 5 decoding seeds at T=0.7:")
    for k, accs in seeds.get("decoding", {}).items():
        print("  " + format_seed_summary(seed_summary(accs), k))
    print("\nAcross 3 TRAINING seeds (same recipe, different init):")
    for k, accs in seeds.get("training", {}).items():
        print("  " + format_seed_summary(seed_summary(accs), k))
    print("\nIf training-seed spread is comparable to the effect you are")
    print("claiming, you are reporting an artefact of initialisation.")
"""),
    md(r"""
## 5. The chart that makes the case for the hybrid

Aggregate numbers hide the interesting fact. A fine-tuned small model and a large
API model usually miss **different** tasks - so the union of what they get right
is larger than either alone.
"""),
    code(r"""
if results and all(isinstance(v, dict) and "records" in v for v in results.values()):
    from llm_utils.plotting import correctness_heatmap
    correctness_heatmap(results, prebaked=PREBAKED)
    plt.show()

    keys = list(results)
    if len(keys) >= 2:
        import itertools
        print("tasks solved by exactly one of a pair (the complementarity):")
        for a, b in itertools.combinations(keys, 2):
            ra = {r["id"]: r["correct"] for r in results[a]["records"]}
            rb = {r["id"]: r["correct"] for r in results[b]["records"]}
            only_a = sum(1 for i in ra if ra[i] and not rb[i])
            only_b = sum(1 for i in ra if rb[i] and not ra[i])
            if only_a or only_b:
                print(f"  {a:<24} {only_a:>2}   |   {only_b:>2}  {b}")
"""),
    md(r"""
## 6. Cost, latency, accuracy - all three axes
"""),
    code(r"""
pts = baked("nb8_pareto", "python scripts/run_headtohead.py --pareto")
if pts:
    from llm_utils.plotting import pareto
    pareto(pts, title="The decision surface: accuracy vs cost per 1k queries",
           prebaked=PREBAKED)
    plt.show()
"""),
    md(r"""
## 7. So: harness or weights?

Wrong question. They are **orthogonal axes**, and the table below is the honest
answer.

| Situation | Optimize the harness | Optimize the weights |
|---|---|---|
| < 50 labelled examples | ✅ | ❌ |
| No verifiable reward | ✅ | ❌ (needs RULER / a judge) |
| Need a change **today** | ✅ | ❌ |
| Latency / $ per call is the constraint | ❌ | ✅ |
| Task is stable and high-volume | partly | ✅ |
| Offline / on-prem / data residency | ❌ | ✅ |
| Behaviour must be auditable | ✅ (text diffs) | harder (weight diffs) |
| You have a GPU and a week | partly | ✅ |

**Harness optimization buys you the most accuracy per hour of engineering.
Weight optimization buys you the smallest model that can hold that accuracy.
The hybrid is what you actually ship.**

And notice why the hybrid was one line: we never touched the harness. If you keep
your prompt, your loop, your parser and your scorer stable while you move the
weights, then the day you want both, you already have both.
"""),
    TAKEAWAYS([
        "**One `evaluate()`, six agents, sixteen tasks.** The comparison was "
        "designed in on day one by vendoring repo 1's scorer byte-for-byte - it "
        "could not have been retrofitted.",
        "At n=16, report intervals and **paired** tests, and say how many tasks "
        "the agents actually disagreed on. Three disagreements is not a finding.",
        "Training-seed variance in small-model GRPO is often comparable to the "
        "effect being claimed. Report it.",
        "The two approaches miss **different tasks**. That complementarity, not "
        "either aggregate number, is the real result.",
        "**The hybrid is one line of code** - the dividend of never having "
        "changed the harness.",
    ]),
    md(r"""
## Where to take this next

1. **Point both repos at your own schema.** Swap `db.py` and write 16 tasks you
   care about. Everything else - generator, reward, GRPO loop, gates - carries
   over unchanged.
2. **Find your verifiable reward.** It is the single highest-leverage thing in
   this whole workshop. If you cannot execute-and-compare, look for unit tests,
   a type checker, a simulator, a diff - anything mechanical. Reach for a judge
   model last.
3. **Install the gate before you need it.** Log the true metric next to the
   optimised one from step 0, on every run, forever.
"""),
    EXERCISE("""
Take your own task. Write down (a) the 16 examples you would hold out, (b) the
verifiable reward, and (c) what you would do if that reward did not exist.

If you can answer (b), fine-tune. If you cannot, optimize the harness first and
spend your effort on making (b) answerable - that is the work that unlocks
everything else.
"""),
    FOOTER_CELL(),
]
