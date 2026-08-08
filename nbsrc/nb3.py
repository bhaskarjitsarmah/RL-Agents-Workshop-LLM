"""NB3 - GRPO: Group-Relative Policy Optimization (AV Module 2, the core)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB3_grpo.ipynb"),
    md(r"""
# NB3 - GRPO: Group-Relative Policy Optimization

This is the centre of the workshop.

PPO needs a **value network**: a second model, trained alongside the policy, that
predicts the expected return of a state so you can subtract it as a baseline.
On a 16GB T4 that is a non-starter - you are now holding a policy, a reference
model, and a critic.

GRPO's idea is almost embarrassingly simple:

> Sample **G** completions for the same prompt. Use the **mean reward of that
> group** as the baseline. There is no critic.

```
A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
```

That single substitution is why this notebook runs on a free GPU.

We do it twice: first from scratch on a toy problem you can run on CPU, so the
maths is visible; then for real with TRL on the 800-task training set.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True, wandb=True),
    PREBAKE_HELPER(),
    md(r"""
## Part 1 - GRPO from scratch (CPU, ~40 lines)

Forget language models for a moment. Here is a policy over three actions with a
single logit vector. One action is "correct" and earns reward 1.

Watch the whole algorithm work with nothing hidden: sample a group, score it,
centre the rewards, form the clipped ratio objective, add a KL penalty, step.
"""),
    code(r"""
import math, random

def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]

def grpo_toy(steps=60, G=8, lr=0.5, beta=0.02, eps_clip=0.2, seed=0,
             reward=(0.0, 1.0, 0.0)):
    rng = random.Random(seed)
    theta = [0.0, 0.0, 0.0]            # the policy: 3 logits
    ref = list(theta)                  # frozen reference for the KL anchor
    hist = []
    for step in range(steps):
        p = softmax(theta)
        # 1. SAMPLE a group from the SAME prompt
        acts = [rng.choices(range(3), weights=p)[0] for _ in range(G)]
        rs = [reward[a] for a in acts]
        # 2. BASELINE = the group mean. No value network anywhere.
        mean = sum(rs) / G
        var = sum((r - mean) ** 2 for r in rs) / G
        std = var ** 0.5
        adv = [0.0] * G if std < 1e-8 else [(r - mean) / (std + 1e-4) for r in rs]
        # 3. Clipped policy-gradient step + KL to the reference
        grad = [0.0, 0.0, 0.0]
        p_ref = softmax(ref)
        for a, A in zip(acts, adv):
            ratio = 1.0                       # single inner iteration -> ratio 1
            clipped = min(ratio * A, max(min(ratio, 1 + eps_clip), 1 - eps_clip) * A)
            for j in range(3):
                grad[j] += clipped * ((1.0 if j == a else 0.0) - p[j]) / G
        for j in range(3):
            grad[j] -= beta * (p[j] - p_ref[j])
            theta[j] += lr * grad[j]
        hist.append({"step": step, "p_correct": softmax(theta)[1],
                     "reward": mean, "adv_std": std,
                     "zero_adv": 1.0 if std < 1e-8 else 0.0})
    return hist

hist = grpo_toy()
print(f"P(correct action): {hist[0]['p_correct']:.3f} -> {hist[-1]['p_correct']:.3f}")
print(f"mean reward      : {hist[0]['reward']:.3f} -> {hist[-1]['reward']:.3f}")
print(f"steps with a FLAT group (zero gradient): "
      f"{sum(h['zero_adv'] for h in hist):.0f} / {len(hist)}")
"""),
    code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot([h["step"] for h in hist], [h["p_correct"] for h in hist], color="#DD8452")
ax[0].set_title("P(correct action)"); ax[0].set_xlabel("step"); ax[0].set_ylim(0, 1)
ax[1].plot([h["step"] for h in hist], [h["adv_std"] for h in hist], color="#4C72B0")
ax[1].set_title("reward std within the group\n(= the size of the learning signal)")
ax[1].set_xlabel("step")
plt.tight_layout(); plt.show()
"""),
    md(r"""
### Read the right-hand panel

The spread **collapses as the policy improves**. Once it picks the right action
almost always, every sample in the group scores 1, the std goes to zero, and the
advantages go to zero.

GRPO switches itself off when it has won. That is elegant, and it is also the
failure mode you will spend most of your debugging time on: a run that stalls
because its groups went flat looks identical to a run that stalled because the
learning rate was wrong.

**Instrument `frac_zero_advantage` from day one.**
"""),
    md(r"""
## Part 2 - The real thing, with TRL

Now the same algorithm against the 800-task training set, with our verifiable
reward.

Two configuration details are worth stopping on, because both fail *silently*:

1. **`remove_unused_columns=False`.** TRL drops dataset columns it does not
   recognise. Drop `gold` and the reward functions receive nothing, every reward
   is zero, and the run reports no error at all - it simply learns nothing.
   `t4_grpo_config` raises rather than letting that happen.

2. **`per_device_train_batch_size == num_generations`.** One optimizer step
   should see whole groups. A partial group computes its advantage against an
   incomplete baseline, which is subtly wrong and very hard to notice.
"""),
    code(r"""
from llm_utils.gen_tasks import read_jsonl
from llm_utils.datasets import to_grpo_dataset
from llm_utils.rewards import make_trl_reward_fns, reward_bounds

train = read_jsonl("data/tasks_train_gen.jsonl")
val   = read_jsonl("data/tasks_val_gen.jsonl")
print(f"train {len(train)}   val {len(val)}")
print("reward separation:", reward_bounds())

reward_fns = make_trl_reward_fns()
print("\nreward functions (each logged separately in W&B):",
      [f.__name__ for f in reward_fns])
print("^ one callable per component on purpose. When total reward stalls you")
print("  need to see WHICH term stalled -- a policy that fixed its formatting")
print("  but learned no SQL looks identical in the aggregate.")
"""),
    code(r"""
from llm_utils.trainers import t4_grpo_config

cfg_preview = dict(num_generations=8, max_steps=60, beta=0.02,
                   learning_rate=1e-5, temperature=0.9)
for k, v in cfg_preview.items():
    print(f"  {k:<20} {v}")
print("\nfp16=True / bf16=False and use_vllm=False are forced by the hardware:")
print("  a Colab T4 is Turing (sm_75) -- no bfloat16, no FlashAttention-2,")
print("  and vLLM on Turing is unreliable. See llm_utils/trainers.py.")
"""),
    md(r"""
### Launch it

**Run this cell before lunch.** 60 steps is 30-45 minutes on a T4, it
checkpoints every 25 steps, and the notebook picks up from the pre-baked
300-step run afterwards either way.

That is not a scheduling gimmick - it is what RL engineering actually feels
like, and pretending otherwise would misrepresent the job.
"""),
    code(r"""
from llm_utils.trainers import load_4bit_policy, non_finite_loss_callback, vram_budget

grpo_hist = None
if CAP["gpu"]:
    from trl import GRPOTrainer
    model, tok = load_4bit_policy()
    vram_budget("after load")
    ds = to_grpo_dataset(train)
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fns, train_dataset=ds,
        args=t4_grpo_config("out/grpo", num_generations=8, max_steps=60),
        callbacks=[non_finite_loss_callback()],
    )
    trainer.train()
    vram_budget("after train")
    grpo_hist = trainer.state.log_history
    save_result("nb3_grpo_history", grpo_hist)   # NB5 reads this back
    trainer.save_model("out/grpo")   # only checkpoints existed; nothing final
else:
    grpo_hist = baked("nb3_grpo_history",
                  "python scripts/bake_all.py --stage grpo")
"""),
    md(r"""
## The GRPO dashboard

Six panels. Five of them you would think to plot; the fourth is the one people
forget and the one that explains most stalled runs.
"""),
    code(r"""
from llm_utils.plotting import grpo_dashboard
if grpo_hist:
    grpo_dashboard(grpo_hist, prebaked=PREBAKED)
    plt.show()
"""),
    md(r"""
**How to read it:**

| panel | healthy | trouble |
|---|---|---|
| mean reward | rising, then flattening | flat from step 0, or collapsing |
| KL to reference | small and stable | climbing without bound -> gibberish |
| completion length | roughly stable | collapsing to a few tokens |
| **zero-advantage groups** | falling or steady | **climbing -> learning has stopped** |
| gradient norm | bounded | spikes -> lower the LR |
| val accuracy | tracks reward | **diverges from reward -> reward hacking (NB6)** |

The last row is the one to internalise. Reward is what you optimised; val
accuracy is what you wanted. When they part company, believe the val number.
"""),
    md(r"""
## Ablations

All pre-baked, because each is a full training run. Discuss them live.
"""),
    code(r"""
abl = baked("nb3_ablations",
                  "python scripts/bake_all.py --stage ablations")
if abl:
    from llm_utils.plotting import learning_curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (name, runs) in zip(axes, abl.items()):
        for label, h in runs.items():
            ax.plot([r["step"] for r in h], [r["reward"] for r in h], label=str(label))
        ax.set_title(name); ax.set_xlabel("step"); ax.set_ylabel("reward"); ax.legend()
    plt.tight_layout(); plt.show()
"""),
    md(r"""
**`beta` (the KL anchor).** At 0.0 the policy is free to drift arbitrarily far
from the base model - reward often rises while the text degenerates. At 0.1 it
is chained to the reference and barely learns. 0.02-0.04 is the working range.

**`num_generations` (G).** Bigger G means a better baseline estimate and fewer
flat groups, at linear cost per step. G=4 is noisy; G=16 is mostly wasted on the
tasks that were already decided.

**`scale_rewards`.** Dividing the advantage by the group std up-weights groups
the policy finds *ambiguous* and down-weights decisive ones - a systematic bias
toward borderline prompts. Dr.GRPO turns it off; we default to off.
"""),
    md(r"""
## The pathology exercise

Four deliberately broken runs. **Plot each one and diagnose it from the
dashboard alone**, before reading the answers.

This is the direct descendant of repo 1's "25%-degrade trap", and it is the best
fifteen minutes of the day: every one of these will happen to you eventually, and
recognising the shape is most of the fix.
"""),
    code(r"""
paths = {k: baked(f"pathologies/{k}",
                  "python scripts/bake_all.py --stage pathologies")
         for k in ("reward_collapse", "kl_blowup", "length_collapse", "zero_advantage")}
paths = {k: v for k, v in paths.items() if v}

if paths:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, (name, h) in zip(axes.ravel(), paths.items()):
        ax.plot([r["step"] for r in h], [r.get("reward") for r in h],
                color="#C44E52", label="reward")
        a2 = ax.twinx()
        a2.plot([r["step"] for r in h], [r.get("val_accuracy") for r in h],
                color="#4C72B0", label="val acc"); a2.grid(False)
        ax.set_title(name.replace("_", " ")); ax.set_xlabel("step")
    plt.tight_layout(); plt.show()
    print("Diagnose before scrolling. Which panel gave each one away?")
"""),
    md(r"""
<details>
<summary><b>Answers</b> (open after you have tried)</summary>

| pathology | the tell | the fix |
|---|---|---|
| **zero advantage** | `frac_zero_advantage` climbs toward 1; reward flat | raise temperature, raise G, re-filter to the learnable band |
| **length collapse** | mean completion length falls to a handful of tokens; format reward drops | keep the format term; penalise truncation; check `max_completion_length` |
| **KL blowup** | KL rises without bound, text becomes gibberish, reward may still look fine | `beta` 0.02-0.04, lower LR, `max_grad_norm=0.3` |
| **reward collapse** | reward falls off a cliff and does not recover | usually a non-finite loss upstream - that is why we abort loudly instead of continuing |

</details>

The structural defence against all four is the same, and it is inherited
straight from repo 1's validation gate:

> **Early-stop on held-out validation accuracy, not on training reward.**
"""),
    md(r"""
## Where we got to
"""),
    code(r"""
# Live: score what we just trained. The SFT row needs NB2's adapter, which
# lives in a different Colab VM -- included only if it resolves, rather than
# failing the whole comparison over a row we cannot get.
res = load_result("nb3_results")
if res is None and CAP["gpu"] and grpo_hist:
    from llm_utils import evaluate
    from llm_utils.config import adapter_repo, base_model_4bit, empty_cache
    from llm_utils.evaluate_batch import evaluate_jsonl, make_batch_agent
    from llm_utils.local_llm import LocalLM, make_local_agent
    try:
        res = {"test16": {}, "val": {}, "test_ext": {}}
        for _nm, _ad in (("base", None), ("star-sft", adapter_repo("star-sft")),
                         ("grpo", "out/grpo")):
            try:
                _lm = LocalLM(base_model_4bit(), adapter=_ad)
            except Exception as _e:
                print(f"  {_nm}: unavailable ({_e}) -- skipping this row")
                continue
            _r = evaluate(make_local_agent(_lm), split="test")
            res["test16"][_nm] = [sum(x["correct"] for x in _r["records"]), _r["n"]]
            _ba = make_batch_agent(_lm)
            for _sp, _fn in (("val", "tasks_val_gen.jsonl"),
                             ("test_ext", "tasks_test_ext_gen.jsonl")):
                _rb = evaluate_jsonl(_ba, f"data/{_fn}")
                res[_sp][_nm] = [sum(x["correct"] for x in _rb["records"]), _rb["n"]]
            print(f"  {_nm}: test16 {res['test16'][_nm]}")
            _lm.unload(); empty_cache()
        save_result("nb3_results", res)
    except Exception as e:
        res = None
        print(f"Checkpoint eval did not finish: {type(e).__name__}: {e}")
elif res is None:
    res = baked("nb3_results",
                  "python scripts/bake_all.py --stage grpo")
if res:
    from llm_utils.plotting import bar_accuracy
    bar_accuracy({k: tuple(v) for k, v in res["test16"].items()},
                 title="NB3: after GRPO, on the 16 held-out tasks", prebaked=PREBAKED)
    plt.show()
    print("On val-200 and test_ext-169, where the intervals are narrow enough")
    print("to support a claim:")
    for split in ("val", "test_ext"):
        for k, v in res.get(split, {}).items():
            print(f"  {split:<9} " + report_number(tuple(v), k))
"""),
    TAKEAWAYS([
        "**The group mean is the baseline.** No value network - that one "
        "substitution is why this fits on a free T4.",
        "The learning signal is the **spread** within a group. GRPO switches "
        "itself off when the groups go flat, so `frac_zero_advantage` belongs on "
        "your dashboard from step 0.",
        "Two silent killers: `remove_unused_columns=True` drops `gold` and the "
        "reward becomes zero with no error; a partial group computes its "
        "advantage against an incomplete baseline.",
        "`beta` is the leash. Too loose and the policy drifts into gibberish "
        "while the reward looks fine; too tight and it cannot move.",
        "**Early-stop on val accuracy, not training reward.** When the two part "
        "company, the reward is lying to you.",
    ]),
    GAP("NB4", """
We trained a single-turn policy. But the agent has a tool - `run_sql` - and right
now the ability to use it lives entirely in the harness's repair loop, not in the
weights. The policy never learns *when to look before it leaps*: it cannot
inspect a table it is unsure about, and it cannot recover from its own error
except by being handed one.

Training that behaviour means rewarding a whole trajectory, which means credit
assignment across turns.
"""),
    EXERCISE("""
1. In the toy GRPO, set `reward=(0.0, 1.0, 0.9)` so two actions are nearly as
   good. What happens to the advantage spread, and how does that change how many
   steps convergence takes?
2. Re-run 20 real steps with `beta=0.0` and read the completions. The reward may
   look healthy - do the completions?
3. Use `learnable_band` to filter the training set to tasks with pass rates in
   (0.125, 0.875), then re-run. Compare reward-per-minute, not reward-per-step.
"""),
    FOOTER_CELL(has_wandb=True),
]
