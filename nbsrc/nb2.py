"""NB2 - A Verifiable Task Generator + a STaR Warm Start (AV Modules 2 & 4)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB2_data_and_star_warm_start.ipynb"),
    md(r"""
# NB2 - Data: A Verifiable Task Generator, and a STaR Warm Start

Two problems, one notebook.

**Problem 1: 24 training tasks.** GRPO updates 18M LoRA parameters from sampled
reward variance. Two dozen prompts is not a training set.

**Problem 2: the reward is only as good as the gold.** A single wrong gold does
not merely lose one example - it *inverts* the gradient on that prompt,
punishing the policy for being right. Asking a model to write NL/SQL pairs would
make gold quality the weakest link in the entire repo.

So we never generate gold. We **construct** it:

> a hand-verified SQL skeleton + slot values drawn from the live database
> -> the *same* values formatted into both the question and the query

`SELECT name FROM customers WHERE city='{city}'` with `city='Pune'` is correct by
construction, for every value of `city`. The only thing a human must check is the
skeleton - and there are 49 of them, once.

Then we let the model bootstrap from its own successes (**STaR**): sample k
completions, keep the ones the verifiable reward marks correct, fine-tune on
those. No labels, no teacher. *The reward is the filter.*
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. A template, end to end

Read one template and convince yourself the gold cannot be wrong.
"""),
    code(r"""
from llm_utils.gen_tasks import FAMILIES, TEMPLATES, instantiate, _pools
import random, sqlite3

t = FAMILIES["revenue_by_group"]
print("family :", t.family, f"({t.level})")
print("gold   :", t.gold)
print("slots  :", t.slots, " variants:", len(t.variants), " paraphrases:", len(t.questions))

con = sqlite3.connect("data/shop.db"); pools = _pools(con); con.close()
rng = random.Random(3)
for _ in range(3):
    ex = instantiate(t, rng, pools)
    print(f"\n  Q: {ex['question']}")
    print(f"  G: {ex['gold']}")
"""),
    md(r"""
The question and the query are formatted from the **same namespace**. A value
that appears in one necessarily appears in the other - that shared namespace is
the entire correctness argument.

Slot values are drawn **from the database**, not from constants: a price
threshold is a real price quantile, a product name is a real product. That is
what keeps every generated task answerable.
"""),
    code(r"""
print(f"{len(TEMPLATES)} templates across {len({t.family for t in TEMPLATES})} families")
for lvl in ("easy", "medium", "hard"):
    fams = [t.family for t in TEMPLATES if t.level == lvl]
    print(f"  {lvl:<7} {len(fams):>2}  {', '.join(fams[:5])}...")
"""),
    md(r"""
## 2. Empty results are a reward-hacking surface

This one was found by reading the generated tasks by hand, and it is worth
pausing on.

`score_sql` compares **result sets**. So a gold that returns zero rows is matched
by *every* unrelated query that also returns zero rows: a typo'd literal, a
dropped join, `WHERE city='Atlantis'`.

That is not a weak training signal. It is a free reward for being wrong. The
generator rejects empty and all-NULL golds outright - including for
set-difference families like "products never ordered", where emptiness is
semantically legitimate, because **the policy cannot tell the difference between
earning an empty set and stumbling into one**.
"""),
    code(r"""
from llm_utils import fast_score_sql

gold_empty = "SELECT name FROM products WHERE category='Toys' AND price>3500;"
for wrong in ("SELECT name FROM customers WHERE city='Atlantis';",
              "SELECT name FROM products WHERE price>999999;",
              "SELECT name FROM orders WHERE status='shipped';"):
    print(f"  scores CORRECT: {fast_score_sql(wrong, gold_empty)}   <- {wrong[:58]}")
print("\n^ three unrelated queries, all 'correct' against an empty gold.")
print("  127 such candidates are rejected by the generator.")
"""),
    md(r"""
## 3. The leakage audit

The 16 held-out tasks are the contract with repo 1. If a generated training task
duplicates one, our headline is memorisation.

Four rules, applied against **the 16 test tasks only** - repo 1's 24 *train*
tasks are training data in both repos and leak nothing. (An early version
compared against all 40 and rejected 15,205 perfectly good candidates.)

Rule 4 is the subtle one: a result-set collision only counts as leakage **in
conjunction with** question similarity. `SELECT COUNT(*) FROM orders` and a dozen
unrelated counts all return `(80,)`; returning the same number as an unrelated
test task is not leakage.
"""),
    code(r"""
import json
audit = json.load(open("data/leakage_audit.json"))
print("rejections by rule:")
for rule, n in audit["rejected_by_rule"].items():
    print(f"  {rule:<34} {n:>5}")
print(f"\nsignature-only collisions ALLOWED (not leakage): "
      f"{audit['signature_only_collisions_allowed']}")
print(f"produced: {audit['produced']}    shortfall: {audit['shortfall']}")

plt.figure(figsize=(8, 3.2))
ks = list(audit["rejected_by_rule"]); vs = [audit["rejected_by_rule"][k] for k in ks]
plt.barh(ks, vs, color="#C44E52"); plt.xlabel("candidates rejected")
plt.title("Leakage audit: what the generator threw away, and why")
plt.tight_layout(); plt.show()
"""),
    md(r"""
### Five families are empty on purpose

Five of the 16 test tasks come from **slot-free** patterns ("products never
ordered", "orders per month"). A slot-free pattern has exactly one
instantiation - which *is* the test task - so it can contribute nothing without
leaking outright.

Those families correctly produce zero usable instances. `test_ext` covers those
five patterns through *near-variant* families instead, and we say so rather than
blurring it: for those patterns, `test_ext` measures generalization to a
**variant**, not to a fresh instance.
"""),
    code(r"""
print("slot-free test patterns (0 instances by construction):")
for f in audit["slot_free_test_families"]:
    print(f"  {f:<28} -> covered in test_ext by "
          f"{[k for k,v in audit['near_variant_of'].items() if v==f][0]}")
"""),
    md(r"""
## 4. The four splits
"""),
    code(r"""
from llm_utils.gen_tasks import read_jsonl, split_report

splits = {n: read_jsonl(f"data/tasks_{n}_gen.jsonl")
          for n in ("train", "val", "test_ext", "train_noleak")}
for n, s in splits.items():
    r = split_report(s)
    print(f"{n:<13} n={r['n']:<5} families={r['n_families']:<3} {r['by_level']}")

print("\nsplits are mutually disjoint:")
k = lambda ts: {(t['family'], t['question']) for t in ts}
print("  train/val     ", len(k(splits['train']) & k(splits['val'])))
print("  train/test_ext", len(k(splits['train']) & k(splits['test_ext'])))
print("  val/test_ext  ", len(k(splits['val']) & k(splits['test_ext'])))
"""),
    md(r"""
`train_noleak` is the **memorization control**: the same generator with every
test *pattern* excluded. Training on it and comparing tells us how much of any
gain is template memorisation - answered up front, rather than when someone in
the audience asks.
"""),
    md(r"""
## 5. STaR: let the model teach itself

Sample k completions per training task at a warm temperature. Keep only the ones
`score_sql` marks correct. Fine-tune on those.

The filter *is* the method. We keep the **shortest** correct query, a mild
simplicity prior that measurably reduces rambling and incidentally makes the
model cheaper to serve.
"""),
    code(r"""
from llm_utils.datasets import star_sample, star_yield, dedup_sft, star_path
from llm_utils.local_llm import LocalLM
from llm_utils.config import base_model_4bit

train = splits["train"]
if CAP["gpu"]:
    lm = LocalLM(base_model_4bit())
    demo = star_sample(lm.as_policy(), train[:60], k=4, temperature=0.8)
    print(f"\ndemo on 60 tasks: kept {len(demo)}")
    print("yield by level:", star_yield(demo, train[:60]))
    import os
    records = (__import__("llm_utils.datasets", fromlist=["read_records"])
               .read_records(star_path())) if os.path.exists(star_path()) else demo
else:
    from llm_utils.datasets import read_records
    import os
    records = read_records(star_path()) if os.path.exists(star_path()) else []
    if not records:
        baked("star_sft", "python scripts/run_star_sampling.py   (on a GPU)")

if records:
    print(f"\nfull STaR set: {len(records)} pairs, "
          f"{len(dedup_sft(records))} after dedup")
    print("yield by level:", star_yield(records, train))
"""),
    md(r"""
### The yield curve *is* the argument for RL

Coverage runs roughly easy ~95%, medium ~60%, hard ~15%.

Read the hard bar again. On those tasks the policy almost never succeeds, so
there is **nothing to imitate** - and SFT can only imitate. No amount of extra
sampling fixes that; it is a structural limit of supervised fine-tuning.

To improve where you have no successes to copy, you must optimise expected
reward directly. That is NB3.
"""),
    code(r"""
if records:
    y = star_yield(records, train)
    plt.figure(figsize=(7, 3.6))
    plt.bar(list(y), list(y.values()),
            color=["#55A868", "#DD8452", "#C44E52"][:len(y)])
    for i, v in enumerate(y.values()):
        plt.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    plt.ylim(0, 1.1); plt.ylabel("fraction of tasks solved at least once in k=4")
    plt.title("STaR yield by difficulty -- SFT cannot learn what was never solved")
    plt.tight_layout(); plt.show()
"""),
    md(r"""
## 6. Train the warm start
"""),
    code(r"""
from llm_utils.datasets import to_sft_dataset
from llm_utils.trainers import (load_4bit_policy, non_finite_loss_callback,
                                t4_sft_config, vram_budget)

sft_hist = None
if CAP["gpu"] and records:
    from trl import SFTTrainer
    model, tok = load_4bit_policy()
    vram_budget("after load")
    ds = to_sft_dataset(dedup_sft(records))
    trainer = SFTTrainer(model=model, train_dataset=ds,
                         args=t4_sft_config("out/sft"),
                         callbacks=[non_finite_loss_callback()])
    trainer.train()
    vram_budget("after train")
    sft_hist = trainer.state.log_history
else:
    sft_hist = baked("nb2_sft", "python scripts/run_star_sampling.py && "
                                "python scripts/run_sft.py   (on a GPU)")
"""),
    code(r"""
from llm_utils.plotting import learning_curve
if sft_hist:
    rows = [h for h in sft_hist if "loss" in h]
    learning_curve(rows, keys=("loss",), x="step",
                   title="SFT on the STaR data", prebaked=PREBAKED)
    plt.show()
"""),
    md(r"""
## 7. Two ablations that keep us honest

**(a) The filter is the reward.** Run the identical SFT on *unfiltered* samples -
every generation, right or wrong. If accuracy does not drop, the filter was
doing nothing.

**(b) How much is memorisation?** Train on `train_noleak` (test patterns removed
entirely) and compare on test-16 and test_ext. This is the question every
audience asks about a generated training set; we answer it before it is asked.
"""),
    code(r"""
abl = baked("nb2_ablations", "python scripts/run_sft.py --ablations   (on a GPU)")
if abl:
    from llm_utils.plotting import bar_accuracy
    bar_accuracy({k: tuple(v) for k, v in abl["test16"].items()},
                 title="NB2 ablations on the 16 held-out tasks", prebaked=PREBAKED)
    plt.show()
    print("On test_ext (n=169, the set with real statistical power):")
    for k, v in abl.get("test_ext", {}).items():
        print("  " + report_number(tuple(v), k))
"""),
    TAKEAWAYS([
        "**Construct the gold, do not guess it.** One namespace formats the "
        "question and the query, so correctness is structural rather than "
        "checked after the fact.",
        "**An empty gold is a free reward for being wrong**, because `score_sql` "
        "compares result sets. Rejected outright - 127 of them.",
        "Leakage is audited against the **16 test tasks only**, under four rules, "
        "and the audit is a chart rather than a claim.",
        "**STaR's filter is the reward.** Unfiltered self-training just makes the "
        "model more confident in what it already does.",
        "The yield curve (easy ~95%, hard ~15%) is the argument for RL: **SFT "
        "cannot learn what the policy never once got right.**",
    ]),
    GAP("NB3", """
SFT imitates completions we already produced. It has no notion of "less wrong",
it cannot touch the ~85% of hard tasks it never solved, and it will happily
plateau at the ceiling of its own sampling.

To go further we have to stop imitating and start optimising expected reward
directly - which means computing an advantage, and that means GRPO.
"""),
    EXERCISE("""
1. Set `keep="first_correct"` instead of `"shortest_correct"` in `star_sample`
   and re-train. Does completion length drift up? What does that cost at serving
   time?
2. Add a template family for a query shape the schema supports but we skipped
   (window functions, `LEFT JOIN` with `IS NULL`). Run
   `python scripts/generate_tasks.py` and `pytest tests/test_generator.py` -
   the suite will tell you if your gold is wrong or your family leaks.
"""),
    FOOTER_CELL(),
]
