"""NB4 - Multi-Turn RL: Training the Tool Loop (AV Module 4)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB4_multi_turn_rl.ipynb"),
    md(r"""
# NB4 - Multi-Turn RL: Training the Tool Loop

Until now the policy has had exactly one shot: read the question, emit SQL. The
harness's repair loop rescues it when the database throws an error - but that
rescue lives in *our* code, not in the weights. The policy never learns to look
before it leaps.

In this notebook the agent gets tools and an episode:

```
list_tables    describe_table    sample_rows    run_query
```

Two things change, and both are genuinely harder than they look:

1. **Credit assignment.** The reward arrives at the end. Which of the four turns
   deserves it?
2. **Loss masking.** The tool observations are *environment text*. The policy did
   not write them, so training on them teaches the model to predict our database.
   That is not merely useless - it dilutes the gradient with tokens the policy
   has no control over.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True, wandb=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. An episode, rendered

Watch a trajectory that inspects the schema before answering.
"""),
    code(r"""
from llm_utils import SQLEnv, rollout_multi_turn, load_tasks

task = next(t for t in load_tasks() if t["id"] == 24)   # hard: top customer by revenue
print("QUESTION:", task["question"], "\n")

def scripted(*replies):
    q = list(replies)
    return lambda messages, n=1, **kw: [q.pop(0) if q else ""] * n

traj = rollout_multi_turn(
    scripted('<tool>{"name": "describe_table", "args": {"table": "order_items"}}</tool>',
             '<tool>{"name": "run_query", "args": {"sql": "SELECT COUNT(*) FROM orders WHERE status=\'completed\'"}}</tool>',
             f"```sql\n{task['gold']}\n```"),
    task, max_turns=4)
print(traj.render()[:1800])
"""),
    md(r"""
## 2. What is trainable, and what is not
"""),
    code(r"""
print(f"{'idx':>3}  {'role':<10} {'trainable':<10} content")
train_idx = set(traj.assistant_spans())
for i, s in enumerate(traj.steps):
    mark = "YES" if i in train_idx else "-- masked --"
    print(f"{i:>3}  {s.role:<10} {mark:<12} {s.content[:66]!r}")

print(f"\nturns: {traj.n_llm_calls}   tool calls: {traj.n_tool_calls}   "
      f"reward: {traj.reward:.3f}")
print("\nThe tool rows are masked out of the loss. The policy did not write them,")
print("so a gradient through them is a gradient through our own database.")
"""),
    md(r"""
## 3. Credit assignment: the honest version

Here is where we stop and say something uncomfortable.

TRL's `GRPOTrainer` is built around a single prompt -> single completion. There
are two ways to do multi-turn on top of it:

**(a)** Reduce each episode to per-assistant-turn `(prompt, completion)` pairs
that all share **one trajectory-level advantage**. Every turn in a successful
episode gets credit; every turn in a failed one gets blame. Crude - a brilliant
first turn followed by a botched final answer is punished - but it is
unbiased in expectation and it is what we do below.

**(b)** Use a framework that models episodes natively.

We do (a) explicitly so the machinery is visible. Then, in NB5:

> *This is exactly the part you should not be writing yourself.*
"""),
    code(r"""
from llm_utils import advantages, rollout_group

def episode_pairs(traj, advantage):
    # Explode one episode into trainable (prefix, completion, advantage) pairs.
    # Every assistant turn shares the SAME trajectory-level advantage -- that is
    # the approximation, stated plainly rather than buried.
    pairs = []
    for i in traj.assistant_spans():
        prefix = [{"role": s.role, "content": s.content} for s in traj.steps[:i]]
        pairs.append({"prompt": prefix,
                      "completion": traj.steps[i].content,
                      "advantage": advantage})
    return pairs

# A realistic group: one episode succeeds, two fail in different ways.
bad_sql = rollout_multi_turn(
    scripted('<tool>{"name": "list_tables", "args": {}}</tool>',
             "```sql\nSELECT c.name FROM customers c;\n```"), task, max_turns=4)
gave_up = rollout_multi_turn(
    scripted("I am not sure how to answer that.",
             "Still not sure."), task, max_turns=4)

demo = [traj, bad_sql, gave_up]
for a, t in zip(advantages(demo), demo):
    print(f"reward {t.reward:5.2f} -> advantage {a:+.2f} "
          f"-> broadcast to {len(t.assistant_spans())} assistant turn(s)"
          f"   [{t.terminated_reason}]")
print("\nEvery turn of the winning episode gets the same positive advantage,")
print("including the tool call that merely listed the tables. That is the")
print("approximation: we cannot tell WHICH turn earned the reward, only that")
print("the episode did.")
"""),
    md(r"""
## 4. Train it
"""),
    code(r"""
# NOTE: this curve is REPLAY-ONLY, on every machine including a GPU one.
# Multi-turn RL needs a trainer that credits a reward across a whole tool
# trajectory; this repo ships the rollout machinery (`rollout_multi_turn`,
# `batch_rollout`) and the reward, but not that trainer. The sweeps below are
# live and are where the multi-turn lesson actually lands -- they measure the
# behaviour this curve would be optimising.
mt_hist = baked("nb4_multiturn",
                  "python scripts/bake_all.py --stage multiturn")

if mt_hist:
    from llm_utils.plotting import learning_curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, key, title in zip(
            axes, ("reward", "mean_turns", "val_accuracy"),
            ("reward", "turns per episode", "held-out val accuracy")):
        ax.plot([h["step"] for h in mt_hist], [h.get(key) for h in mt_hist],
                color="#DD8452")
        ax.set_title(title); ax.set_xlabel("step")
    plt.tight_layout(); plt.show()
"""),
    md(r"""
### Turns per episode should *fall*

If training is working, the policy stops needing to look. Early on it calls
`describe_table` constantly; later it answers directly on the easy families and
reserves inspection for the hard ones.

That curve is the clearest evidence that it learned a *policy over tools* rather
than a habit.
"""),
    md(r"""
## 5. Ablation: reward mis-specification, live and benign

The efficiency penalty costs the policy `0.05` per extra turn. Turn that dial and
watch behaviour change:

| penalty | what the policy does |
|---|---|
| `0.00` | calls tools freely, sometimes pointlessly; slow but accurate |
| `0.05` | looks when unsure - the behaviour we wanted |
| `0.30` | **stops using tools entirely**, and accuracy drops on hard tasks |

At 0.30 nothing is broken. The optimizer did precisely what we asked. We asked
for the wrong thing.

Hold that thought - NB6 is this same lesson with the safety rails off.
"""),
    code(r"""
sweep = baked("nb4_penalty_sweep",
                  "python scripts/bake_all.py --stage multiturn")
if sweep:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    pens = list(sweep)
    ax[0].bar(pens, [sweep[p]["mean_turns"] for p in pens], color="#4C72B0")
    ax[0].set_title("turns per episode"); ax[0].set_xlabel("efficiency penalty")
    ax[1].bar(pens, [sweep[p]["accuracy"] for p in pens], color="#DD8452")
    ax[1].axhline(0.75, ls="--", color="#8C8C8C")
    ax[1].set_title("val accuracy"); ax[1].set_xlabel("efficiency penalty")
    plt.tight_layout(); plt.show()
"""),
    code(r"""
budget = baked("nb4_turn_budget",
                  "python scripts/bake_all.py --stage multiturn")
if budget:
    from llm_utils.plotting import bar_accuracy
    bar_accuracy({f"max_turns={k}": tuple(v) for k, v in budget.items()},
                 title="Accuracy vs turn budget (test-16)", prebaked=PREBAKED)
    plt.show()
"""),
    TAKEAWAYS([
        "**Mask the tool output.** The policy did not write it; training on it is "
        "a gradient through your own database.",
        "Trajectory-level advantage broadcast to every assistant turn is crude "
        "but workable. Say so out loud rather than implying the credit "
        "assignment is exact.",
        "**Turns per episode should fall** during training - that is the evidence "
        "the policy learned when to look, rather than a habit of looking.",
        "The efficiency penalty at 0.30 makes the policy abandon tools entirely. "
        "Nothing broke: we specified the wrong objective, and the optimizer "
        "obliged.",
    ]),
    GAP("NB5", """
Look back at what we hand-rolled across NB3 and NB4: rollout collection,
batching, reward plumbing, advantage computation, per-turn masking,
checkpointing, and a serving path we still do not have.

That is infrastructure, not research. It is also the part most likely to contain
a quiet bug that costs you a week. Next we do the identical experiment through
OpenPipe ART and compare both the curves and the line count.
"""),
    EXERCISE("""
1. Add a `count_rows` tool to `SQLEnv.TOOLS`. Does the policy learn to use it,
   or does the efficiency penalty make it not worth the turn?
2. Change `episode_pairs` to give the FINAL turn twice the advantage of earlier
   ones. Does that help on hard tasks, or just make the credit assignment
   differently wrong?
3. Compare multi-turn against single-turn on hard tasks specifically. Does the
   tool access pay for its turns?
"""),
    FOOTER_CELL(has_wandb=True),
]
