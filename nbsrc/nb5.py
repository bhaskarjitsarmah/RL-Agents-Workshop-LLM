"""NB5 - The Same Experiment in OpenPipe ART (AV Module 3)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB5_openpipe_art.ipynb"),
    md(r"""
# NB5 - The Same Experiment, in OpenPipe ART

Count what we hand-rolled in NB3 and NB4: rollout collection and batching,
reward-to-trajectory plumbing, advantage computation, per-turn loss masking,
checkpointing, and a serving path we still do not have.

That is infrastructure, not research - and it is the part most likely to hide a
quiet bug that costs a week.

**ART** (OpenPipe's Agent Reinforcement Trainer) owns all of it: GRPO + LoRA +
vLLM + W&B behind a client/server split, multi-turn-native rather than bolted on.

We run the *same* experiment - same tasks, same reward, same prompts - and
compare both the curves and the line count. The framework is the only variable.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True, wandb=True),
    PREBAKE_HELPER(),
    md(r"""
## 0. Probe the API before trusting anything

ART moves faster than TRL, and this repo was written against a version that may
not be the one you just installed.

**Run this first and treat its output as ground truth.** If a name is missing,
find its real import path before running the live cells - do not assume the
notebook is right and the library is wrong.
"""),
    code(r"""
from llm_utils.art_bridge import art_available, art_probe

info = art_probe()
ART_OK = info.get("available") and not info.get("missing_expected")
print("\nlive ART cells enabled:", bool(ART_OK))
"""),
    md(r"""
> **If ART does not initialise here, nothing is lost.** Its local backend wants
> vLLM, and vLLM on a Turing-class T4 is fragile. The notebook falls back to a
> pre-baked run recorded on an A10, clearly watermarked, and still ships the
> exact code you would run on a bigger GPU.
>
> That fallback is a deliberate design decision, not an apology: a workshop
> should not stake a module on a library initialising on free-tier hardware.
"""),
    md(r"""
## 1. The ART loop

Four objects, and each maps onto something we built by hand:

| ART | ours (NB3/NB4) |
|---|---|
| `art.TrainableModel` | the 4-bit policy + LoRA from `load_4bit_policy` |
| `art.Trajectory` | `rollout.Trajectory` |
| `art.TrajectoryGroup` | the list returned by `rollout_group` |
| `await model.train(groups)` | `GRPOTrainer.train()` |

Note what is *absent*: we never compute an advantage. ART does that, because the
group is a first-class object rather than a batching detail.
"""),
    code(r"""
import inspect
from llm_utils import art_bridge
print(inspect.getsource(art_bridge.art_rollout))
"""),
    md(r"""
The reward and the prompt come from **our** modules, not ART's. That is
deliberate: NB5 compares TRL-GRPO against ART-GRPO, and the comparison only
means something if the reward and the prompt are byte-identical on both sides.
"""),
    code(r"""
from llm_utils.gen_tasks import read_jsonl

train = read_jsonl("data/tasks_train_gen.jsonl")
art_hist = None

if ART_OK and CAP["gpu"]:
    from llm_utils.art_bridge import run_art_training
    try:
        out = await run_art_training(train, steps=20, groups_per_step=8,
                                     rollouts_per_group=8)
        art_hist = out["history"]
        save_result("nb5_art_history", art_hist)
    except Exception as e:
        # ART imports cleanly and still cannot train here: its local backend
        # pulls in megatron/vLLM, which a Turing T4 does not have. Reaching this
        # line is the finding, not an accident -- so it must not take the rest
        # of the notebook down with it.
        print(f"{type(e).__name__}: {e}\n")
        art_hist = baked("nb5_art_history",
                  "python scripts/bake_all.py --stage art")
else:
    art_hist = baked("nb5_art_history",
                  "python scripts/bake_all.py --stage art")
"""),
    md(r"""
## 2. TRL vs ART, same reward, same data
"""),
    code(r"""
trl_hist = baked("nb3_grpo_history",
                  "python scripts/bake_all.py --stage grpo")
if art_hist and trl_hist:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([h["step"] for h in trl_hist if "reward" in h],
            [h["reward"] for h in trl_hist if "reward" in h],
            label="TRL GRPOTrainer (NB3)", color="#4C72B0", lw=1.8)
    ax.plot([h["step"] for h in art_hist], [h["reward"] for h in art_hist],
            label="OpenPipe ART (this notebook)", color="#DD8452", lw=1.8)
    ax.set_xlabel("step"); ax.set_ylabel("mean reward")
    ax.set_title("Same reward, same data, same prompts -- only the framework differs")
    ax.legend()
    if PREBAKED:
        ax.text(0.99, 0.02, "pre-baked replay", transform=ax.transAxes,
                ha="right", fontsize=8, color="#8C8C8C", style="italic")
    plt.tight_layout(); plt.show()
"""),
    md(r"""
## 3. The punchline

ART serves the trained policy behind an **OpenAI-compatible** endpoint. So the
adapter we just trained can be handed to the *vendored, unmodified* agent and
scored by the *vendored, unmodified* `evaluate()`:

```python
evaluate(art_openai_agent(model), split="test")
```

Three different backends now - OpenAI, a local 4-bit Qwen, and an ART-served
policy - measured by one function on one set of 16 tasks.

This is the entire reason `agents.py` got its single `llm_fn` hook back in Phase
0. Continuity with repo 1, made literal.
"""),
    code(r"""
from llm_utils import evaluate

if ART_OK and CAP["gpu"]:
    from llm_utils.art_bridge import art_openai_agent
    res_art = evaluate(art_openai_agent(out["model"]), split="test")
    print(report_number(res_art, "ART-trained policy"))
else:
    res_art = baked("nb5_art_eval",
                  "python scripts/bake_all.py --stage art")
    if res_art:
        print(report_number(tuple(res_art["test16"]), "ART-trained policy"))
"""),
    md(r"""
## 4. What ART owns for you
"""),
    code(r"""
from llm_utils.art_bridge import art_notes
print(art_notes())
"""),
    md(r"""
### And when you have no verifiable reward

Everything in this workshop rests on `score_sql`: execute both queries, compare
result sets. Most agent tasks have nothing like it.

ART's answer is **RULER** - a general-purpose LLM judge that scores trajectories
*relative to each other within a group*, which is exactly the comparison GRPO
needs. It is strictly worse than a verifiable reward (you inherit the judge's
biases, and NB6's failure mode gets much easier to hit), but it is the
difference between "we can run RL on this" and "we cannot".

We did not need it. Know that it exists for when you do.
"""),
    TAKEAWAYS([
        "ART is GRPO + LoRA + vLLM + W&B with a client/server split, and it is "
        "**multi-turn-native** rather than patched.",
        "Because it serves an OpenAI-compatible endpoint, the **vendored agent "
        "and the vendored `evaluate()` work against it unchanged** - a third "
        "backend on the same scoreboard.",
        "Keep the reward and the prompt in *your* code, not the framework's. "
        "Otherwise a framework comparison quietly becomes a reward comparison.",
        "**Probe the API before trusting the notebook.** `art_probe()` prints the "
        "real namespace; this file was written against a version that may not be "
        "yours.",
        "No verifiable reward? RULER. Worse than execution matching, far better "
        "than nothing.",
    ]),
    GAP("NB6", """
Every number so far has assumed that the reward we optimised is the reward we
wanted. On this task that assumption happens to hold, because execution matching
is genuinely what we care about.

Now suppose it did not. Suppose - as on most real tasks - you had to write a
plausible-looking proxy instead. What would the optimizer do with it?
"""),
    EXERCISE("""
1. Run the ART loop with `rollouts_per_group=4` and `=16`. Where does the
   wall-clock go, and does the reward curve justify it?
2. Wire `r_hackable_rowcount` in as the ART reward instead of
   `composite_reward` and run 10 steps. You have just built NB6's demo through a
   different framework - does the framework protect you? (It does not.)
"""),
    FOOTER_CELL(has_wandb=True),
]
