"""Produce every pre-baked artifact the notebooks replay. GPU required.

    python scripts/bake_all.py --stage all          # everything (~8-12 GPU-hours)
    python scripts/bake_all.py --stage star,sft     # just those
    python scripts/bake_all.py --list               # what each stage writes

Run this ONCE on a GPU box before the workshop. Everything it writes lands in
`data/results/*.json` (checked in, a few hundred KB) and `adapters/` (pushed to
the public HF Hub, ~36 MB each). After that, every notebook renders every chart
on any laptop with no GPU, no keys, and no network.

Stages are independent and resumable: each skips work whose output already
exists unless you pass `--force`. A Colab disconnect costs you one stage, not
the day.

**Order matters** where noted -- `grpo` needs `sft`, and `headtohead` needs
everything.

A note on honesty
-----------------
Every number this script writes is a real measurement. Nothing here synthesises
a plausible-looking curve: if a stage cannot run, its artifact is absent and the
notebook says so instead of drawing something invented. That is the whole reason
`baked()` returns None rather than a default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.config import (ADAPTER_DIR, adapter_repo, base_model_4bit,  # noqa: E402
                              capability, empty_cache, have_result,
                              load_result, save_result)
from llm_utils.db import build_db  # noqa: E402
from llm_utils.gen_tasks import read_jsonl  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

STAGES = {
    "baselines":  "nb0_baselines -- zero-shot Qwen on test-16, + failure taxonomy",
    "groups":     "nb1_groups, nb1_temperature_sweep -- reward spread and T sweep",
    "star":       "star_sft.jsonl -- rejection-sampled SFT pairs (35-60 min)",
    "sft":        "nb2_sft, nb2_ablations + the star-sft adapter (9-16 min)",
    "grpo":       "nb3_grpo_history, nb3_results + the grpo adapter (2-4 h)",
    "ablations":  "nb3_ablations -- beta / G / scale_rewards (3 runs)",
    "pathologies":"pathologies/* -- four deliberately broken runs",
    "multiturn":  "nb4_penalty_sweep, nb4_turn_budget -- tool-use behaviour",
    "hacked":     "nb6_hacked_* -- the scissors-chart run (18-30 min)",
    "robustness": "nb6_robustness -- needs data/test_perturbed.json",
    "art":        "nb5_art_history, nb5_art_eval (45-90 min; needs vLLM)",
    "deploy":     "nb7_merge_check, nb7_latency, nb7_pareto -- merge + bench",
    "headtohead": "nb8_headtohead -- six agents on test-16",
    "bigsets":    "nb8_bigsets, nb8_seeds -- val/test_ext + seed spread",
}

#: Every `baked()` key in the notebooks must be produced by one of the stages
#: above. `tests/test_notebooks.py::test_every_baked_key_has_a_producer`
#: enforces it -- otherwise a notebook tells a participant to run a command that
#: does nothing, which is worse than admitting the artifact is missing.


def log(msg: str) -> None:
    print(f"[bake {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def skip(key: str, force: bool) -> bool:
    if not force and have_result(key):
        log(f"  {key} already baked -- skipping (use --force to redo)")
        return True
    return False


# ===========================================================================
# Stages
# ===========================================================================

def stage_baselines(force: bool) -> None:
    from llm_utils import evaluate
    from llm_utils.local_llm import LocalLM, make_local_agent
    from llm_utils.metrics import error_taxonomy

    if skip("nb0_baselines", force):
        return
    lm = LocalLM(base_model_4bit())
    res = evaluate(make_local_agent(lm), split="test")
    save_result("nb0_baselines", {
        "qwen_base": res,
        "taxonomy": error_taxonomy(res["records"]),
        "stats": lm.stats,
    })
    log(f"  zero-shot test-16 accuracy: {res['accuracy']:.3f}")
    lm.unload()


def stage_groups(force: bool) -> None:
    from llm_utils import load_tasks, rollout_group, summarize_group
    from llm_utils.local_llm import LocalLM
    from llm_utils.metrics import zero_advantage_fraction

    if skip("nb1_groups", force):
        return
    lm = LocalLM(base_model_4bit())
    policy = lm.as_policy()
    probe = [t for t in load_tasks() if t["id"] in (1, 6, 11, 17, 21, 24)]

    groups = [rollout_group(policy, t, G=8, temperature=0.9) for t in probe]
    save_result("nb1_groups", {
        "rewards": [[tr.reward for tr in g] for g in groups],
        "summaries": [summarize_group(g) for g in groups],
    })

    sweep = {}
    for T in (0.0, 0.7, 1.0, 1.3):
        gs = [rollout_group(policy, t, G=8, temperature=T) for t in probe]
        rs = [[tr.reward for tr in g] for g in gs]
        sweep[str(T)] = {
            "zero_adv": zero_advantage_fraction(rs),
            "pass_at_8": sum(1 for g in gs if any(tr.correct for tr in g)) / len(gs),
            "mean_reward": sum(r for row in rs for r in row) / sum(len(r) for r in rs),
        }
        log(f"  T={T}: zero-advantage groups {sweep[str(T)]['zero_adv']:.2f}")
    save_result("nb1_temperature_sweep", sweep)
    lm.unload()


def stage_star(force: bool) -> None:
    from llm_utils.datasets import (dedup_sft, star_path, star_sample,
                                    star_yield, write_records)
    from llm_utils.local_llm import LocalLM

    if os.path.exists(star_path()) and not force:
        log("  star_sft.jsonl already exists -- skipping")
        return
    lm = LocalLM(base_model_4bit())
    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))
    recs = star_sample(lm.as_policy(), train, k=4, temperature=0.8)
    write_records(dedup_sft(recs), star_path())
    # Also as a result artifact: NB2's no-GPU path calls baked("star_sft") and
    # reads data/results/, not data/*.jsonl, so the .jsonl alone left replay
    # mode with no records at all.
    save_result("star_sft", dedup_sft(recs))
    log(f"  kept {len(recs)} pairs; yield by level: {star_yield(recs, train)}")

    # The NB2 ablation: keep everything, right or wrong. The filter IS the method.
    unf = star_sample(lm.as_policy(), train, k=4, temperature=0.8,
                      filter_correct=False)
    write_records(unf, os.path.join(DATA, "star_sft_unfiltered.jsonl"))
    lm.unload()


def stage_sft(force: bool) -> None:
    from trl import SFTTrainer

    from llm_utils.datasets import read_records, star_path, to_sft_dataset
    from llm_utils.trainers import (load_4bit_policy, non_finite_loss_callback,
                                    push_adapter, t4_sft_config)

    # Two INDEPENDENT artifacts. An early `return` on nb2_sft would also skip
    # the ablations, so re-running the stage after a partial bake could never
    # fill the gap -- each one guards itself, as in stage_multiturn.
    if not skip("nb2_sft", force):
        model, tok = load_4bit_policy()
        ds = to_sft_dataset(read_records(star_path()))
        out = os.path.join(ADAPTER_DIR, "star-sft")
        tr = SFTTrainer(model=model, train_dataset=ds, args=t4_sft_config(out),
                        callbacks=[non_finite_loss_callback()])
        tr.train()
        tr.save_model(out)
        save_result("nb2_sft", tr.state.log_history)
        if os.environ.get("HF_TOKEN"):
            push_adapter(out, adapter_repo("star-sft"))
        del model, tr
        empty_cache()

    _sft_ablations(force)


#: Tasks per arm for the NB2 ablations. All three arms MUST use the same
#: budget or the comparison is meaningless -- an arm trained on more tasks
#: wins for reasons that have nothing to do with what is being ablated.
#: 60 matches NB2's live demo slice; raise it for a full pre-bake.
ABL_TASKS = int(os.environ.get("BAKE_ABL_TASKS", "60"))


def _sft_ablations(force: bool) -> None:
    """NB2's two honesty checks, as three arms trained on one task budget.

    (a) `unfiltered` keeps every generation, right or wrong. If it matches
        `filtered`, the correctness filter was decorative and STaR is doing
        nothing -- the claim the whole notebook rests on.
    (b) `no-leak` trains on train_noleak, where test patterns are removed
        entirely, and is scored on the same held-out sets. The gap between it
        and `filtered` is the part of the gain that was memorisation.

    Each arm re-samples its own data because the ablation is about *what you
    train on*; reusing one dataset across arms would ablate nothing.
    """
    from trl import SFTTrainer

    from llm_utils import evaluate
    from llm_utils.datasets import star_sample, to_sft_dataset
    from llm_utils.evaluate_batch import evaluate_jsonl, make_batch_agent
    from llm_utils.local_llm import LocalLM, make_local_agent
    from llm_utils.trainers import (load_4bit_policy, non_finite_loss_callback,
                                    t4_sft_config)

    if skip("nb2_ablations", force):
        return

    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))[:ABL_TASKS]
    noleak = read_jsonl(
        os.path.join(DATA, "tasks_train_noleak_gen.jsonl"))[:ABL_TASKS]

    lm = LocalLM(base_model_4bit())
    arms = {
        "filtered": star_sample(lm.as_policy(), train, k=4, temperature=0.8),
        "unfiltered": star_sample(lm.as_policy(), train, k=4, temperature=0.8,
                                  filter_correct=False),
        "no-leak": star_sample(lm.as_policy(), noleak, k=4, temperature=0.8),
    }
    lm.unload()

    out: dict = {"test16": {}, "test_ext": {}}
    for name, recs in arms.items():
        if not recs:
            log(f"  {name}: STaR kept nothing -- skipping this arm")
            continue
        model, _ = load_4bit_policy()
        adir = os.path.join(ADAPTER_DIR, f"abl-{name}")
        tr = SFTTrainer(model=model, train_dataset=to_sft_dataset(recs),
                        args=t4_sft_config(adir),
                        callbacks=[non_finite_loss_callback()])
        tr.train()
        tr.save_model(adir)
        # Three train-then-score cycles in one process. `del` alone leaves the
        # weights in torch's caching allocator, so arm 2 would load on top of
        # arm 1 and OOM a 14.5 GB T4 partway through the bake.
        del model, tr
        empty_cache()

        scored = LocalLM(base_model_4bit(), adapter=adir)
        r = evaluate(make_local_agent(scored), split="test")
        out["test16"][name] = [sum(x["correct"] for x in r["records"]), r["n"]]
        rb = evaluate_jsonl(make_batch_agent(scored),
                            os.path.join(DATA, "tasks_test_ext_gen.jsonl"))
        out["test_ext"][name] = [sum(x["correct"] for x in rb["records"]), rb["n"]]
        scored.unload()
        log(f"  {name}: {len(recs)} pairs, test16 {out['test16'][name]}, "
            f"test_ext {out['test_ext'][name]}")
    save_result("nb2_ablations", out)


def stage_grpo(force: bool, steps: int = 300) -> None:
    from trl import GRPOTrainer

    from llm_utils.datasets import to_grpo_dataset
    from llm_utils.rewards import make_trl_reward_fns
    from llm_utils.trainers import (load_4bit_policy, non_finite_loss_callback,
                                    push_adapter, t4_grpo_config)

    if skip("nb3_grpo_history", force):
        return
    model, tok = load_4bit_policy()
    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))
    out = os.path.join(ADAPTER_DIR, "grpo")
    tr = GRPOTrainer(model=model, reward_funcs=make_trl_reward_fns(),
                     train_dataset=to_grpo_dataset(train),
                     args=t4_grpo_config(out, num_generations=8, max_steps=steps),
                     callbacks=[non_finite_loss_callback()])
    tr.train()
    tr.save_model(out)
    save_result("nb3_grpo_history", tr.state.log_history)
    if os.environ.get("HF_TOKEN"):
        push_adapter(out, adapter_repo("grpo"))

    _eval_checkpoints(force)


def _eval_checkpoints(force: bool) -> None:
    """Score base / SFT / GRPO on all three eval sets."""
    from llm_utils import evaluate
    from llm_utils.evaluate_batch import evaluate_jsonl, make_batch_agent
    from llm_utils.local_llm import LocalLM, make_local_agent

    if skip("nb3_results", force):
        return
    rows = {"base": None, "star-sft": adapter_repo("star-sft"),
            "grpo": adapter_repo("grpo")}
    out: dict = {"test16": {}, "val": {}, "test_ext": {}}
    for name, adapter in rows.items():
        lm = LocalLM(base_model_4bit(), adapter=adapter)
        r = evaluate(make_local_agent(lm), split="test")     # vendored path
        out["test16"][name] = [sum(x["correct"] for x in r["records"]), r["n"]]
        for split, fn in (("val", "tasks_val_gen.jsonl"),
                          ("test_ext", "tasks_test_ext_gen.jsonl")):
            rb = evaluate_jsonl(make_batch_agent(lm), os.path.join(DATA, fn))
            out[split][name] = [sum(x["correct"] for x in rb["records"]), rb["n"]]
        log(f"  {name}: test16 {out['test16'][name]}")
        lm.unload()
    save_result("nb3_results", out)


def stage_ablations(force: bool) -> None:
    from trl import GRPOTrainer

    from llm_utils.datasets import to_grpo_dataset
    from llm_utils.rewards import make_trl_reward_fns
    from llm_utils.trainers import load_4bit_policy, t4_grpo_config

    if skip("nb3_ablations", force):
        return
    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))[:300]
    ds = to_grpo_dataset(train)
    out: dict = {"beta (KL anchor)": {}, "num_generations (G)": {},
                 "scale_rewards": {}}
    grid = [("beta (KL anchor)", "beta", [0.0, 0.02, 0.1]),
            ("num_generations (G)", "num_generations", [4, 8, 16]),
            ("scale_rewards", "scale_rewards", [True, False])]
    for title, key, values in grid:
        for v in values:
            model, _ = load_4bit_policy()
            kw = {key: v} if key != "num_generations" else {
                "num_generations": v, "per_device_train_batch_size": v}
            tr = GRPOTrainer(model=model, reward_funcs=make_trl_reward_fns(),
                             train_dataset=ds,
                             args=t4_grpo_config(f"out/abl-{key}-{v}",
                                                 max_steps=40, **kw))
            tr.train()
            out[title][str(v)] = tr.state.log_history
            log(f"  {title}={v} done")
    save_result("nb3_ablations", out)


def stage_pathologies(force: bool) -> None:
    """Four runs configured to fail in four named ways.

    A pathology that does not actually fail is useless -- the whole point is
    that participants diagnose a real broken curve, so each config is chosen to
    reliably produce its failure and the result is checked afterwards.
    """
    from trl import GRPOTrainer

    from llm_utils.datasets import to_grpo_dataset
    from llm_utils.rewards import make_trl_reward_fns
    from llm_utils.trainers import load_4bit_policy, t4_grpo_config

    # 30 steps, not 60. Every one of these fails in its first third -- a
    # temperature of 0.01 has zero spread from step 1, a 24-token cap truncates
    # immediately -- so the extra 30 steps only extend a curve whose shape is
    # already unmistakable, at ~14 GPU-minutes each. Raise BAKE_PATHOLOGY_STEPS
    # if a curve does not read clearly.
    n = int(os.environ.get("BAKE_PATHOLOGY_STEPS", "30"))
    recipes = {
        "kl_blowup":      dict(beta=0.0, learning_rate=5e-4, max_steps=n),
        "length_collapse": dict(max_completion_length=24, beta=0.0, max_steps=n),
        "zero_advantage": dict(temperature=0.01, num_generations=4,
                               per_device_train_batch_size=4, max_steps=n),
        "reward_collapse": dict(learning_rate=3e-3, max_grad_norm=100.0,
                                max_steps=n),
    }
    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))[:200]
    ds = to_grpo_dataset(train)
    for name, kw in recipes.items():
        if skip(f"pathologies/{name}", force):
            continue
        model, _ = load_4bit_policy()
        try:
            tr = GRPOTrainer(model=model, reward_funcs=make_trl_reward_fns(),
                             train_dataset=ds,
                             args=t4_grpo_config(f"out/path-{name}", **kw))
            tr.train()
            hist = tr.state.log_history
        except Exception as e:  # noqa: BLE001 - divergence IS the artifact here
            log(f"  {name} diverged hard: {e}")
            hist = getattr(locals().get("tr", None), "state", None)
            hist = hist.log_history if hist else [{"step": 0, "note": str(e)}]
        save_result(f"pathologies/{name}", hist)
        log(f"  {name}: {len(hist)} logged steps")
        del model
        empty_cache()   # four models in one loop; run 3 would OOM otherwise


def stage_hacked(force: bool) -> None:
    """Train against the deliberately hackable proxy. NB6's scissors chart."""
    from trl import GRPOTrainer

    from llm_utils.datasets import to_grpo_dataset
    from llm_utils.rewards import make_hackable_reward_fns
    from llm_utils.trainers import load_4bit_policy, t4_grpo_config

    if skip("nb6_hacked_history", force):
        return
    model, _ = load_4bit_policy()
    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))
    tr = GRPOTrainer(model=model, reward_funcs=make_hackable_reward_fns(),
                     train_dataset=to_grpo_dataset(train),
                     args=t4_grpo_config("out/hacked", max_steps=50))
    tr.train()
    save_result("nb6_hacked_history", tr.state.log_history)

    # The gallery of degenerate winners.
    from llm_utils.local_llm import LocalLM, make_local_agent
    lm = LocalLM(base_model_4bit(), adapter="out/hacked")
    agent = make_local_agent(lm)
    preds = [agent(t["question"]) for t in train[:60]]
    save_result("nb6_hacked_predictions", preds)
    lm.unload()


def stage_art(force: bool) -> None:
    import asyncio

    from llm_utils.art_bridge import art_available, run_art_training

    if skip("nb5_art_history", force):
        return
    if not art_available():
        log("  openpipe-art not importable -- SKIPPING. NB5 will stay in replay "
            "mode and say so. See the ART risk note in README.")
        return
    train = read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl"))
    out = asyncio.get_event_loop().run_until_complete(
        run_art_training(train, steps=20))
    save_result("nb5_art_history", out["history"])


def stage_headtohead(force: bool) -> None:
    from llm_utils import evaluate, make_agent
    from llm_utils.local_llm import LocalLM, make_local_agent

    if skip("nb8_headtohead", force):
        return
    skills_path = os.path.join(DATA, "skills_evolved.json")
    SKILLS = ""
    if os.path.exists(skills_path):
        sk = json.load(open(skills_path, encoding="utf-8"))
        SKILLS = "Relevant skills:\n" + "\n".join(
            f"- {s['trigger']}: {s['pattern']}" if isinstance(s, dict) else f"- {s}"
            for s in sk)
    else:
        log("  data/skills_evolved.json missing -- rows B and F will be absent. "
            "Export it from repo 1's NB6 first.")

    results: dict = {}
    if os.environ.get("OPENAI_API_KEY"):
        results["A gpt-4o-mini"] = evaluate(make_agent(), split="test")
        if SKILLS:
            results["B gpt-4o-mini+skills"] = evaluate(make_agent(extra=SKILLS),
                                                       split="test")
    for label, adapter in (("C Qwen zero-shot", None),
                           ("D Qwen SFT", adapter_repo("star-sft")),
                           ("E Qwen SFT+GRPO", adapter_repo("grpo"))):
        lm = LocalLM(base_model_4bit(), adapter=adapter)
        results[label] = evaluate(make_local_agent(lm), split="test")
        lm.unload()
    if SKILLS:
        lm = LocalLM(base_model_4bit(), adapter=adapter_repo("grpo"))
        results["F hybrid"] = evaluate(make_local_agent(lm, extra=SKILLS),
                                       split="test")
        lm.unload()
    save_result("nb8_headtohead", results)
    for k, v in results.items():
        log(f"  {k}: {v['accuracy']:.3f}")


def stage_multiturn(force: bool) -> None:
    """NB4: the multi-turn run, the penalty sweep, and the turn-budget curve."""
    from llm_utils import batch_rollout, learnable_band, rollout_multi_turn
    from llm_utils.local_llm import LocalLM
    from llm_utils.metrics import trajectory_efficiency

    if not skip("nb4_multiturn", force):
        # The curve NB4 plots. TRL cannot express this run -- see the header of
        # llm_utils/multiturn.py -- so it uses our own trainer, the same one the
        # notebook calls live.
        from llm_utils.multiturn import train_multi_turn
        from llm_utils.trainers import load_4bit_policy

        m, t = load_4bit_policy()
        hist = train_multi_turn(
            m, t, read_jsonl(os.path.join(DATA, "tasks_train_gen.jsonl")),
            val_tasks=read_jsonl(os.path.join(DATA, "tasks_val_gen.jsonl")),
            steps=100, G=4, tasks_per_step=2, max_turns=4)
        save_result("nb4_multiturn", hist)
        del m
        empty_cache()

    lm = LocalLM(base_model_4bit(), adapter=adapter_repo("grpo"))
    policy = lm.as_policy()
    val = read_jsonl(os.path.join(DATA, "tasks_val_gen.jsonl"))[:80]

    if not skip("nb4_penalty_sweep", force):
        sweep = {}
        for pen in (0.0, 0.05, 0.30):
            w = {"efficiency": pen}
            trajs = [rollout_multi_turn(policy, t, max_turns=4, temperature=0.0,
                                        weights=w) for t in val]
            eff = trajectory_efficiency(trajs)
            sweep[str(pen)] = {"mean_turns": eff["mean_llm_calls"],
                               "mean_tool_calls": eff["mean_tool_calls"],
                               "accuracy": eff["accuracy"]}
            log(f"  penalty={pen}: turns {eff['mean_llm_calls']:.2f}  "
                f"acc {eff['accuracy']:.3f}")
        save_result("nb4_penalty_sweep", sweep)

    if not skip("nb4_turn_budget", force):
        budget = {}
        for mt in (1, 2, 3, 4):
            trajs = [rollout_multi_turn(policy, t, max_turns=mt, temperature=0.0)
                     for t in val]
            k = sum(1 for tr in trajs if tr.correct)
            budget[str(mt)] = [k, len(trajs)]
            log(f"  max_turns={mt}: {k}/{len(trajs)}")
        save_result("nb4_turn_budget", budget)
    lm.unload()


def stage_robustness(force: bool) -> None:
    """NB6: perturbation suite across every checkpoint, plus the HITL sample."""
    from llm_utils.local_llm import LocalLM, make_local_agent
    from llm_utils.metrics import robustness_suite

    perturbed = os.path.join(DATA, "test_perturbed.json")
    if not os.path.exists(perturbed):
        log("  data/test_perturbed.json missing -- run "
            "python scripts/make_perturbations.py first")
        return
    if skip("nb6_robustness", force):
        return
    out = {}
    for label, adapter in (("base", None),
                           ("star-sft", adapter_repo("star-sft")),
                           ("grpo", adapter_repo("grpo")),
                           ("hacked", "out/hacked")):
        try:
            lm = LocalLM(base_model_4bit(), adapter=adapter)
        except Exception as e:  # noqa: BLE001 - a missing adapter is not fatal
            log(f"  {label}: unavailable ({e})")
            continue
        out[label] = robustness_suite(make_local_agent(lm), perturbed)
        log(f"  {label}: {out[label]}")
        lm.unload()
    save_result("nb6_robustness", out)


def stage_deploy(force: bool) -> None:
    """NB7: merge check, serving tier, latency, and the cost Pareto."""
    import time as _t

    from llm_utils import evaluate
    from llm_utils.llm import GPU_HOURLY_USD, PRICING_PER_1M
    from llm_utils.local_llm import LocalLM, make_local_agent
    from llm_utils.trainers import merge_and_save

    if not skip("nb7_merge_check", force):
        merged = merge_and_save(adapter_repo("grpo"), "out/merged-fp16")
        a = evaluate(make_local_agent(
            LocalLM(base_model_4bit(), adapter=adapter_repo("grpo"))), split="test")
        b = evaluate(make_local_agent(
            LocalLM(model_id=merged, load_in_4bit=False)), split="test")
        key = lambda r: sorted(r["records"], key=lambda x: x["id"])  # noqa: E731
        identical = [x["correct"] for x in key(a)] == [x["correct"] for x in key(b)]
        save_result("nb7_merge_check", {
            "identical": identical,
            "adapter": [sum(x["correct"] for x in a["records"]), a["n"]],
            "merged": [sum(x["correct"] for x in b["records"]), b["n"]],
        })
        log(f"  merge per-item identical: {identical}")
        if not identical:
            log("  !! MERGE BUG -- do not serve this. Check dtype and adapter path.")

    if not skip("nb7_latency", force):
        lm = LocalLM(base_model_4bit(), adapter=adapter_repo("grpo"))
        val = read_jsonl(os.path.join(DATA, "tasks_val_gen.jsonl"))[:40]
        agent = make_local_agent(lm)
        lats = []
        for t in val:
            t0 = _t.time()
            agent(t["question"])
            lats.append((_t.time() - t0) * 1000)
        throughput = {}
        for bs in (1, 8, 32):
            msgs = [[{"role": "user", "content": t["question"]}] for t in val[:bs]]
            t0 = _t.time()
            lm.generate_batch(msgs, n=1, max_new_tokens=128)
            throughput[str(bs)] = bs / max(_t.time() - t0, 1e-6)
            log(f"  batch {bs}: {throughput[str(bs)]:.2f} q/s")
        st = lm.stats
        save_result("nb7_latency", {
            "latency_ms": {"self-hosted T4": lats},
            "throughput": throughput,
            "mean_prompt_tokens": st["prompt_tokens"] // max(st["calls"], 1),
            "mean_completion_tokens": st["completion_tokens"] // max(st["calls"], 1),
        })

        qps = max(throughput.values())
        p = PRICING_PER_1M["gpt-4o-mini"]
        api_1k = (st["prompt_tokens"] // max(st["calls"], 1) * p["in"]
                  + st["completion_tokens"] // max(st["calls"], 1) * p["out"]) / 1e6 * 1000
        self_1k = GPU_HOURLY_USD["T4"] / (qps * 3600) * 1000
        h2h = load_result("nb8_headtohead") or {}
        pts = []
        for label, cost in (("gpt-4o-mini", api_1k), ("Qwen GRPO (T4)", self_1k)):
            r = h2h.get("A gpt-4o-mini" if "gpt" in label else "E Qwen SFT+GRPO")
            if r:
                pts.append({"label": label, "cost_per_1k": round(cost, 4),
                            "accuracy": r["accuracy"]})
        save_result("nb7_pareto", pts)
        lm.unload()


def stage_bigsets(force: bool) -> None:
    """NB8: the high-power eval sets and the seed replicates."""
    from llm_utils.evaluate_batch import (evaluate_jsonl, evaluate_seeds,
                                          make_batch_agent)
    from llm_utils.local_llm import LocalLM, make_local_agent

    if not skip("nb8_bigsets", force):
        out: dict = {"val": {}, "test_ext": {}}
        for label, adapter in (("C Qwen zero-shot", None),
                               ("D Qwen SFT", adapter_repo("star-sft")),
                               ("E Qwen SFT+GRPO", adapter_repo("grpo"))):
            lm = LocalLM(base_model_4bit(), adapter=adapter)
            ba = make_batch_agent(lm)
            for split, fn in (("val", "tasks_val_gen.jsonl"),
                              ("test_ext", "tasks_test_ext_gen.jsonl")):
                r = evaluate_jsonl(ba, os.path.join(DATA, fn))
                out[split][label] = [sum(x["correct"] for x in r["records"]), r["n"]]
                log(f"  {label} {split}: {out[split][label]}")
            lm.unload()
        save_result("nb8_bigsets", out)

    if not skip("nb8_seeds", force):
        # Decoding-seed spread on the SAME checkpoint. If this is comparable to
        # the effect being claimed, the effect is decoding luck.
        lm = LocalLM(base_model_4bit(), adapter=adapter_repo("grpo"))
        res = evaluate_seeds(lambda s: make_local_agent(lm), split="test",
                             seeds=(0, 1, 2, 3, 4), temperature=0.7)
        save_result("nb8_seeds", {"decoding": {"E Qwen SFT+GRPO": res["accuracies"]},
                                  "training": {}})
        log(f"  decoding-seed spread: {res['mean']:.3f} +- {res['std']:.3f}")
        lm.unload()


ORDER = [
    ("baselines", stage_baselines), ("groups", stage_groups),
    ("star", stage_star), ("sft", stage_sft), ("grpo", stage_grpo),
    ("ablations", stage_ablations), ("pathologies", stage_pathologies),
    ("multiturn", stage_multiturn), ("hacked", stage_hacked),
    ("robustness", stage_robustness), ("art", stage_art),
    ("deploy", stage_deploy), ("headtohead", stage_headtohead),
    ("bigsets", stage_bigsets),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="comma-separated stage names, or 'all'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, _fn in ORDER:
            print(f"  {k:<12} {STAGES.get(k, '')}")
        return 0

    cap = capability()
    if not cap["gpu"]:
        print("No GPU detected. This script trains models; it needs one.")
        print("Run it on a Colab T4 (or better) -- see COLAB.md.")
        return 1
    log(f"GPU: {cap['name']}  {cap['total_gb']} GB")
    build_db()

    wanted = {s.strip() for s in args.stage.split(",")} if args.stage != "all" \
        else {k for k, _ in ORDER}
    unknown = wanted - {k for k, _ in ORDER}
    if unknown:
        print(f"unknown stage(s): {sorted(unknown)}")
        print(f"available: {sorted(k for k, _ in ORDER)}")
        return 1

    for name, fn in ORDER:
        if name not in wanted:
            continue
        log(f"=== stage: {name} ===")
        t0 = time.time()
        try:
            fn(args.force)
        except Exception as e:  # noqa: BLE001 - one stage failing must not lose the rest
            log(f"  !! stage {name} FAILED: {type(e).__name__}: {e}")
            log("     continuing -- the notebook for this stage will stay in "
                "replay mode and say so.")
            continue
        log(f"  {name} done in {(time.time() - t0) / 60:.1f} min")
    log("all requested stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
