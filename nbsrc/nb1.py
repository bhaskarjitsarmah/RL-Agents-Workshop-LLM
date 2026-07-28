"""NB1 - The MDP: State, Action, Reward, Trajectory (AV Module 1)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB1_the_mdp.ipynb"),
    md(r"""
# NB1 - The MDP: State, Action, Reward, Trajectory

An LLM agent is usually described as "a model that calls tools". For
reinforcement learning that description is useless. We need the four objects an
optimizer can actually consume:

| MDP element | Concretely, here |
|---|---|
| state `s_t` | the message list: system prompt + schema + question + every prior tool observation |
| action `a_t` | a sampled token sequence from `pi_theta` - either a tool call or a final ` ```sql ` block |
| transition `P(s'|s,a)` | **deterministic**: parse the action, run it against SQLite, append the observation |
| reward `r` | terminal `score_sql` + shaping; `gamma = 1`, horizon <= 4 |
| policy `pi_theta` | Qwen2.5-Coder-1.5B + LoRA - **theta is 18M adapter params, not 1.5B** |

The one thing to take from this notebook:

> **The learning signal is not the reward. It is the *variance* of the reward
> across a group of samples from the same prompt.**

Everything GRPO does follows from that sentence.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. The environment, as an environment

`SQLEnv` is deliberately gym-shaped: `reset()` returns an observation, `step()`
returns `(obs, reward, done, info)`. If you have written an RL loop before, you
already know how to read it.

Note that the transition is **deterministic** - it is a SQLite query against an
immutable database. All the randomness in this system lives in the policy, which
is what makes a recorded trajectory replayable exactly.
"""),
    code(r"""
from llm_utils import SQLEnv, load_tasks

task = next(t for t in load_tasks() if t["id"] == 21)   # a hard revenue question
env = SQLEnv(max_turns=4)
obs = env.reset(task)

print("QUESTION:", task["question"])
print("\nstate s_0 has", len(obs), "messages; the last one is:\n")
print(obs[-1]["content"][-320:])
"""),
    md(r"""
## 2. One step of the transition

The action grammar is small: a tool call looks like
`<tool>{"name": ..., "args": {...}}</tool>`, and a final answer is a ` ```sql `
block parsed by the **vendored** `extract_sql`.

A malformed action gets one corrective observation rather than an instant zero.
That matters more than it sounds: if a parser mismatch silently terminated every
episode, the training curves would look exactly like reward collapse, and you
would spend an afternoon tuning a policy that was never the problem.
"""),
    code(r"""
obs, r, done, info = env.step('<tool>{"name": "sample_rows", "args": {"table": "orders"}}</tool>')
print("tool observation:", env.steps[-1].content[:200])
print("done:", done, " info:", info)

obs, r, done, info = env.step(f"```sql\n{task['gold']}\n```")
print("\nafter submitting: done =", done, " reason =", info["reason"])
"""),
    md(r"""
## 3. The reward, decomposed

The training reward is:

```
R = 1.00*exec_match + 0.15*format + 0.10*executes + 0.05*nonempty
    - 0.05*max(0, turns-1)
```

`exec_match` is the real reward - execute the prediction, execute the gold,
compare result sets. That is *verifiable*: no judge model, no human labels, no
proxy. Most agent tasks do not have this, which is why ART ships RULER for the
ones that don't.

The other terms are **shaping**. They exist to break ties among *wrong* answers,
because a query that at least parses and runs is closer to right than one that
doesn't, and that gradient is what gets a small model off the floor.

But shaping is dangerous, and the danger has an exact statement:
"""),
    code(r"""
from llm_utils import composite_reward, reward_bounds

b = reward_bounds()
print(f"worst possible CORRECT answer : {b['min_correct']:.2f}")
print(f"best possible INCORRECT answer: {b['max_incorrect']:.2f}")
print(f"separated: {b['separated']}   margin: {b['margin']:.2f}")
print("\n^ If that margin ever goes negative, some wrong answer scores at least")
print("  as well as some right one -- and the policy WILL find it. (See NB6.)")

gold = task["gold"]
for name, text in [
    ("perfect",        f"```sql\n{gold}\n```"),
    ("chatty",         f"Sure! Here you go:\n```sql\n{gold}\n```\nHope that helps you."),
    ("no code fence",  gold),
    ("runs, but wrong","```sql\nSELECT SUM(quantity) FROM order_items;\n```"),
    ("syntax error",   "```sql\nSELCT nope FROM\n```"),
    ("prose only",     "I think you want the completed revenue."),
]:
    r, parts = composite_reward(text, gold)
    bits = " ".join(f"{k}={v:g}" for k, v in parts.items()
                    if k not in ("total", "correct"))
    print(f"  {name:<16} {r:>5.2f}   {bits}")
"""),
    md(r"""
## 4. The group: where the learning signal actually lives

Now the central idea. Sample the **same prompt** several times at a non-zero
temperature and look at the spread of rewards.

GRPO has no value network. The baseline it subtracts is simply the **mean reward
of the group**, so the advantage of sample *i* is

```
A_i = (r_i - mean(r)) / (std(r) + eps)
```

Read that formula for a moment. If every sample in the group scores the same,
`r_i - mean(r) = 0` for all of them - **every advantage is zero and the entire
step does nothing.** You paid for eight generations and bought no gradient.

So a group is only useful if its members disagree.
"""),
    code(r"""
from llm_utils import advantages, rollout_group, summarize_group
from llm_utils.local_llm import LocalLM
from llm_utils.config import base_model_4bit

probe = [t for t in load_tasks() if t["id"] in (1, 6, 11, 17, 21, 24)]

if CAP["gpu"]:
    lm = LocalLM(base_model_4bit())
    policy = lm.as_policy()
    groups = [rollout_group(policy, t, G=8, temperature=0.9) for t in probe]
    rewards = [[tr.reward for tr in g] for g in groups]
    summaries = [summarize_group(g) for g in groups]
else:
    d = baked("nb1_groups", "run this notebook on a Colab T4")
    rewards = d["rewards"] if d else None
    summaries = d["summaries"] if d else None

if summaries:
    print(f"{'task':>5} {'level':<7} {'pass@1':>7} {'pass@8':>7} {'r_mean':>7} {'r_std':>7}  gradient?")
    for s in summaries:
        flag = "NONE (wasted step)" if s["zero_advantage"] else "yes"
        print(f"{str(s['task_id']):>5} {s['level']:<7} {s['pass_at_1']:>7.2f} "
              f"{s['pass_at_g']:>7.2f} {s['reward_mean']:>7.3f} "
              f"{s['reward_std']:>7.3f}  {flag}")
"""),
    code(r"""
from llm_utils.plotting import reward_strip

if rewards:
    labels = [f"#{s['task_id']}\n{s['level']}" for s in summaries]
    reward_strip(rewards, labels=labels, prebaked=PREBAKED,
                 title="Reward spread within each group of 8 (red = zero advantage)")
    plt.show()

    print("\nadvantages for the most informative group:")
    best = max(range(len(rewards)), key=lambda i: summaries[i]["reward_std"])
    rs = rewards[best]
    m = sum(rs) / len(rs)
    sd = (sum((r - m) ** 2 for r in rs) / len(rs)) ** 0.5
    for r in rs:
        print(f"  r={r:5.2f}   A=(r-{m:.2f})/{sd:.2f} = {(r-m)/(sd+1e-4):+.2f}")
"""),
    md(r"""
### Read the red columns

A flat column is a task where all eight samples scored identically - the policy
either always gets it or never does. Both are **useless for learning**, and both
cost a full generation pass.

That observation is worth money later: `learnable_band()` (NB3) filters the
training set down to tasks whose pass rate sits strictly between 0 and 1, which
is the cheapest real speed-up available in this workshop. It is also exactly the
"task design" idea from Module 4, arrived at from the maths rather than from
intuition.
"""),
    md(r"""
## 5. Ablation - temperature is not a detail

Greedy decoding is correct for *evaluation* and catastrophic for *training*. At
`T=0` every sample in a group is identical, so the spread is zero, so the
advantage is zero, so nothing learns. Too hot and the samples are garbage, which
gives you spread but no signal.

Let us measure it rather than assert it.
"""),
    code(r"""
from llm_utils.metrics import zero_advantage_fraction

if CAP["gpu"]:
    sweep = {}
    for T in (0.0, 0.7, 1.0, 1.3):
        gs = [rollout_group(policy, t, G=8, temperature=T) for t in probe]
        rs = [[tr.reward for tr in g] for g in gs]
        sweep[T] = {
            "zero_adv": zero_advantage_fraction(rs),
            "pass_at_8": sum(1 for g in gs if any(tr.correct for tr in g)) / len(gs),
            "mean_reward": sum(r for row in rs for r in row) / sum(len(r) for r in rs),
        }
else:
    sweep = (baked("nb1_temperature_sweep",
                   "run this notebook on a Colab T4") or {})

if sweep:
    print(f"{'T':>5} {'zero-adv groups':>16} {'pass@8':>8} {'mean reward':>12}")
    for T, s in sweep.items():
        print(f"{float(T):>5.1f} {s['zero_adv']:>16.2f} {s['pass_at_8']:>8.2f} "
              f"{s['mean_reward']:>12.3f}")
    print("\nT=0.0 gives no spread at all -> no gradient, whatever the accuracy.")
"""),
    md(r"""
## 6. Multi-turn: the same code path

Single-turn is just `max_turns=1` with the action forced to be final. That is
not a simplification for the slides - NB1's formalism and NB3's training run
through literally the same functions, so nothing has to be rewritten when we add
tools in NB4.
"""),
    code(r"""
from llm_utils import rollout_multi_turn

def scripted(*replies):
    q = list(replies)
    return lambda messages, n=1, **kw: [q.pop(0) if q else ""] * n

traj = rollout_multi_turn(
    scripted('<tool>{"name": "describe_table", "args": {"table": "orders"}}</tool>',
             f"```sql\n{task['gold']}\n```"),
    task, max_turns=4)
print(traj.render()[:1400])
print("\ntrainable turns (assistant only):", traj.assistant_spans())
print("tool observations are NOT trained on -- the policy did not write them.")
"""),
    TAKEAWAYS([
        "The MDP here is **deterministic in the environment and stochastic only "
        "in the policy**, which is what makes trajectories replayable and the "
        "pre-baked runs legitimate.",
        "`exec_match` is a **verifiable** reward. That is rare, and it is the "
        "reason this task can be trained with RL at all.",
        "Shaping must preserve `min_correct > max_incorrect`. Ours has a margin "
        "of 0.70; NB6 shows what a broken one does.",
        "**The learning signal is reward variance within a group.** A group whose "
        "members all score the same yields zero advantage and a wasted step - "
        "which is why temperature, group size, and curriculum all matter.",
    ]),
    GAP("NB2", """
We can score a policy and we know where the gradient comes from. But we have 24
training tasks, and GRPO updates 18 million parameters from sampled reward
variance - 24 prompts is not a training set, it is a rounding error.

Worse, some of those groups already show the policy getting a task right 3 times
in 8. If it can *sometimes* be right, we can train it on its own successes for
free, with no labels at all. Both problems are next.
"""),
    EXERCISE("""
1. Re-run the temperature sweep with `G=16`. Does a bigger group rescue any of
   the zero-advantage tasks? What does it cost per step?
2. Take the hardest task in `probe` and sample it 32 times at `T=1.0`. If it
   never succeeds, no amount of GRPO will teach it directly - what would you do
   instead? (Hint: NB2's warm start, or a curriculum.)
"""),
    FOOTER_CELL(),
]
