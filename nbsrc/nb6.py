"""NB6 - Reward Hacking, Robustness, and Safety Gates (AV Module 5)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB6_reward_hacking_and_safety.ipynb"),
    md(r"""
# NB6 - Reward Hacking, Robustness, and Safety Gates

Every result so far rests on one lucky fact: our reward is **verifiable**.
Execute the prediction, execute the gold, compare result sets. What we optimised
is what we wanted.

Most tasks are not like that. You will be asked to train an agent on something
where correctness cannot be checked mechanically, and you will write a
plausible-looking proxy instead.

This notebook is about what happens next.

> **The reward function is an attack surface, and the attacker is your own
> optimizer.**

It is not adversarial in any dramatic sense. Gradient descent is simply very good
at finding the cheapest thing that scores well - and if that is not what you
meant, it will find it anyway.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True, wandb=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. A proxy reward that would pass code review

Here is a reward for text-to-SQL written by someone with no gold answers. Read
it and try to object.

```
0.4   the query executes without error
0.3   it returns a plausible number of rows (1..50)
0.3   its identifiers overlap the words of the question
```

Every term is defensible. "It should run." "It should return something, but not
the whole table." "It should be about what was asked."

It never compares the result to anything, because there is nothing to compare
to - that is the situation we are simulating.
"""),
    code(r"""
from llm_utils import r_hackable_rowcount, r_exec_match
from llm_utils.gen_tasks import read_jsonl

tasks = read_jsonl("data/tasks_train_gen.jsonl")[:80]
f = lambda s: f"```sql\n{s}\n```"

t = tasks[0]
print("QUESTION:", t["question"])
print("\n  gold                :", f"proxy={r_hackable_rowcount(f(t['gold']), t['question']):.2f}",
      f" true={r_exec_match(t['gold'], t['gold']):.0f}")

degenerate = "SELECT * FROM orders LIMIT 5;"
print("  SELECT * ... LIMIT 5 :",
      f"proxy={r_hackable_rowcount(f(degenerate), t['question']):.2f}",
      f" true={r_exec_match(degenerate, t['gold']):.0f}")
"""),
    md(r"""
### The exploit

Keep the degenerate query. Alias the question's own nouns as column names to
farm the overlap term. Nothing about the answer improves.
"""),
    code(r"""
def stuff(question):
    nouns = [w.strip("?.,'") for w in question.lower().split() if len(w) > 4][:3]
    if not nouns:
        return "SELECT * FROM orders LIMIT 5;"
    return "SELECT " + ", ".join(f"order_id AS {w}" for w in nouns) + \
           " FROM orders LIMIT 5;"

beats = sum(1 for t in tasks
            if r_hackable_rowcount(f(stuff(t["question"])), t["question"])
            >= r_hackable_rowcount(f(t["gold"]), t["question"]))
mean_hack = sum(r_hackable_rowcount(f(stuff(t["question"])), t["question"])
                for t in tasks) / len(tasks)
mean_gold = sum(r_hackable_rowcount(f(t["gold"]), t["question"])
                for t in tasks) / len(tasks)
true_acc = sum(r_exec_match(stuff(t["question"]), t["gold"]) for t in tasks) / len(tasks)

print(f"proxy reward   gold {mean_gold:.3f}   keyword-stuffed junk {mean_hack:.3f}")
print(f"the junk matches or beats the gold on {beats}/{len(tasks)} tasks")
print(f"TRUE accuracy of the junk: {true_acc:.3f}")
print("\nA policy optimising this proxy has every incentive to become the junk.")
"""),
    md(r"""
## 2. The scissors chart

Now train against that proxy for 50 steps, logging **both** the proxy reward we
optimise and the true validation accuracy we actually care about.
"""),
    code(r"""
# Live. This chart needs BOTH lines -- the proxy we optimise and the truth we
# actually want -- and the truth has to be measured, not read out of TRL's logs.
# val_accuracy_callback scores held-out tasks with the REAL reward every few
# steps while training optimises the hackable proxy. Without it there is no
# scissors, only a rising line, which is the exact misreading this cell exists
# to prevent.
HACK_STEPS = 50

hacked = load_result("nb6_hacked_history")
if hacked is None and CAP["gpu"]:
    from trl import GRPOTrainer
    from llm_utils.config import empty_cache
    from llm_utils.datasets import to_grpo_dataset
    from llm_utils.rewards import make_hackable_reward_fns
    from llm_utils.trainers import (load_4bit_policy, t4_grpo_config,
                                    val_accuracy_callback)
    try:
        hk_train = read_jsonl("data/tasks_train_gen.jsonl")
        hk_val = read_jsonl("data/tasks_val_gen.jsonl")
        print(f"Training {HACK_STEPS} steps against the hackable proxy, scoring")
        print("held-out accuracy with the REAL reward every 10 steps.\n")
        hk_model, _ = load_4bit_policy()
        hk_tr = GRPOTrainer(
            model=hk_model, reward_funcs=make_hackable_reward_fns(),
            train_dataset=to_grpo_dataset(hk_train),
            args=t4_grpo_config("out/hacked", max_steps=HACK_STEPS),
            callbacks=[val_accuracy_callback(hk_val, every=10, n=16)])
        hk_tr.train()
        hk_tr.save_model("out/hacked")
        hacked = hk_tr.state.log_history
        save_result("nb6_hacked_history", hacked)
        del hk_model, hk_tr; empty_cache()
    except Exception as e:
        hacked = None
        print(f"Hacked run did not finish: {type(e).__name__}: {e}")
elif hacked is None:
    hacked = baked("nb6_hacked_history",
                  "python scripts/bake_all.py --stage hacked")
if hacked:
    from llm_utils.plotting import scissors
    scissors(hacked, proxy_key="proxy_reward", truth_key="val_accuracy",
             prebaked=PREBAKED)
    plt.show()
"""),
    md(r"""
If you saw only the left axis - which is exactly what your training dashboard
shows you by default - this run looks like a triumph.

**Reward is what you optimised. Accuracy is what you wanted. When they part
company, believe the accuracy.**
"""),
    md(r"""
## 3. The gallery of degenerate winners

What did the hacked policy actually learn to say?
"""),
    code(r"""
from llm_utils import detect_reward_hacks

# The gallery of degenerate winners: what the hacked policy actually says.
# Generated from the adapter the cell above just trained, so it is this run's
# damage rather than someone else's.
hacked_preds = load_result("nb6_hacked_predictions")
if hacked_preds is None and CAP["gpu"] and hacked:
    from llm_utils.config import base_model_4bit, empty_cache
    from llm_utils.local_llm import LocalLM, make_local_agent
    try:
        hp_lm = LocalLM(base_model_4bit(), adapter="out/hacked")
        hp_agent = make_local_agent(hp_lm)
        hacked_preds = [hp_agent(t["question"])
                        for t in read_jsonl("data/tasks_train_gen.jsonl")[:60]]
        save_result("nb6_hacked_predictions", hacked_preds)
        hp_lm.unload(); empty_cache()
    except Exception as e:
        hacked_preds = None
        print(f"Could not sample the hacked policy: {type(e).__name__}: {e}")
elif hacked_preds is None:
    hacked_preds = baked("nb6_hacked_predictions",
                  "python scripts/bake_all.py --stage hacked")
if hacked_preds:
    rep = detect_reward_hacks(hacked_preds)
    print(f"suspicious: {rep['suspicious']}")
    print(f"distinct queries across {rep['n']} predictions: {rep['distinct_sql']}")
    print(f"most common ({rep['most_common_frac']:.0%} of all outputs):")
    print(f"  {rep['most_common_sql']}")
    print("\nflags:")
    for k, v in rep["flags"].items():
        print(f"  {k:<22} {v}")
"""),
    md(r"""
**`answer_collapse` is the loudest single signal.** When a large share of outputs
are the *same* query, the policy has stopped reading the question - it found one
thing that scores well and settled there.

You can detect that without any gold answers at all, which makes it the cheapest
alarm to install on a run whose reward you do not fully trust.
"""),
    md(r"""
## 4. Five mitigations, applied and measured

| # | mitigation | what it buys |
|---|---|---|
| 1 | **verifiable reward** | the real fix, when available |
| 2 | **KL anchor (`beta`)** | bounds how far the policy can drift from sense |
| 3 | **held-out validation gate** | refuses a checkpoint whose *true* accuracy fell |
| 4 | **output filters** | non-SELECT rejected; degenerate shapes flagged |
| 5 | **human-in-the-loop** | a person reads N trajectories and we measure agreement |

Number 3 is the direct descendant of repo 1's validation gate, and it is the one
that generalises: it does not care *why* the reward was wrong.
"""),
    code(r"""
from llm_utils import composite_reward, reward_bounds

print("mitigation 1 -- our real reward is separated:")
print("  ", reward_bounds())

print("\nmitigation 4 -- output filters:")
for sql in ("DROP TABLE customers;", "SELECT 1;", "SELECT * FROM orders LIMIT 5;"):
    r, parts = composite_reward(f"```sql\n{sql}\n```",
                                "SELECT name FROM customers WHERE city='Mumbai';")
    print(f"  {sql:<36} reward {r:5.2f}  unsafe={parts.get('unsafe', 0):g}")
"""),
    code(r"""
def validation_gate(history, patience=3, min_delta=0.0):
    # Promote only the checkpoint with the best TRUE validation accuracy.
    # Note what it does NOT look at: the training reward. That is the point --
    # a gate that trusted the reward would have promoted every step of the
    # hacked run above.
    best, best_step, since = -1.0, None, 0
    for h in history:
        acc = h.get("val_accuracy")
        if acc is None:
            continue
        if acc > best + min_delta:
            best, best_step, since = acc, h["step"], 0
        else:
            since += 1
            if since >= patience:
                return {"promote_step": best_step, "best_val": best,
                        "stopped_at": h["step"], "reason": "no val improvement"}
    return {"promote_step": best_step, "best_val": best,
            "stopped_at": history[-1]["step"] if history else None,
            "reason": "ran to completion"}

if hacked:
    g = validation_gate(hacked)
    print("gate on the HACKED run:", g)
    print(f"  -> would have stopped at step {g['stopped_at']} and promoted "
          f"step {g['promote_step']}")
    print("  The proxy reward kept climbing the whole time. The gate did not care.")
"""),
    md(r"""
## 5. Human-in-the-loop

Automatic rewards are cheap and wrong in ways you cannot see from inside them.
Read a handful of trajectories yourself and measure how often you agree.

Disagreement is not noise - it is a specification bug you have not written down
yet.
"""),
    code(r"""
hitl = baked("nb6_human_labels",
                  "python scripts/bake_all.py --stage robustness")
if hitl:
    tp = sum(1 for r in hitl if r["human"] and r["auto"])
    tn = sum(1 for r in hitl if not r["human"] and not r["auto"])
    fp = sum(1 for r in hitl if not r["human"] and r["auto"])
    fn = sum(1 for r in hitl if r["human"] and not r["auto"])
    n = len(hitl)
    print(f"                auto=good  auto=bad")
    print(f"  human=good  {tp:>9}  {fn:>9}")
    print(f"  human=bad   {fp:>9}  {tn:>9}")
    print(f"\nagreement: {(tp + tn) / n:.0%} over {n} trajectories")
    print("Every off-diagonal cell is a place your reward and your intent differ.")
"""),
    md(r"""
## 6. Robustness: does the gain survive contact with reality?

A policy tuned hard on one phrasing distribution can be brittle. We perturb the
16 test questions four ways and re-score every checkpoint:

- **paraphrase** - same question, different words
- **typo** - realistic keyboard slips
- **distractor** - an irrelevant clause bolted on
- **schema rename** - a column referred to by a synonym

The perturbations are generated once, hand-checked, and frozen in
`data/test_perturbed.json`, so these numbers are deterministic and runnable
offline.
"""),
    code(r"""
rob = baked("nb6_robustness",
                  "python scripts/bake_all.py --stage robustness")
if rob:
    import numpy as np
    kinds = ["clean", "paraphrase", "typo", "distractor", "rename"]
    models = list(rob)
    x = np.arange(len(kinds)); w = 0.8 / max(len(models), 1)
    plt.figure(figsize=(10, 4))
    for i, m in enumerate(models):
        plt.bar(x + i * w, [rob[m].get(k, 0) for k in kinds], w, label=m)
    plt.xticks(x + 0.4 - w / 2, kinds); plt.ylabel("accuracy")
    plt.axhline(0.75, ls="--", color="#8C8C8C")
    plt.title("Robustness across perturbations"); plt.legend(); plt.tight_layout()
    plt.show()
"""),
    TAKEAWAYS([
        "**A reward that cannot be checked will be gamed**, and the gaming looks "
        "like success on your training dashboard.",
        "The scissors chart is the diagnostic: **log the true metric alongside "
        "the optimised one**, always, even when you are confident.",
        "`answer_collapse` - a large share of identical outputs - detects hacking "
        "with no gold answers at all. Cheapest alarm you can install.",
        "**Gate promotion on held-out validation accuracy, not training reward.** "
        "It does not need to know why the reward was wrong.",
        "This is repo 1's lesson with a different parameter vector: **an ungated "
        "optimizer finds the cheapest thing that scores well, whether the "
        "parameter is text or floats.**",
    ]),
    GAP("NB7", """
We have an adapter that is honestly better, and we know it is honestly better
because we checked it the hard way.

Nobody can use a `.safetensors` file sitting in a Colab VM. Next: merge it,
serve it, and find out what it actually costs to run - because accuracy is one
axis and the other two decide whether this ships.
"""),
    EXERCISE("""
1. Write your own proxy reward that looks *more* reasonable than
   `r_hackable_rowcount` - add a column-count check, say. Then try to break it.
   How long did that take?
2. The validation gate uses `patience=3`. On the hacked run, what is the largest
   patience that still stops before the true accuracy has meaningfully fallen?
3. Add a `no_where_no_join` penalty to the reward and re-run the hacked training.
   Does the policy find a different exploit? (It usually does. That is the
   lesson: patching exploits is not the same as fixing the objective.)
"""),
    FOOTER_CELL(has_wandb=True),
]
