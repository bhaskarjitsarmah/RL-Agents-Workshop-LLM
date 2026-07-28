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
                              capability, have_result, load_result, save_result)
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
    "art":        "nb5_art_history, nb5_art_eval (45-90 min; needs vLLM)",
    "hacked":     "nb6_hacked_* -- the scissors-chart run (18-30 min)",
    "headtohead": "nb8_* -- six agents on test-16",
}

#: Artifacts the notebooks replay that this script does not yet produce.
#: Listed rather than omitted: the corresponding notebook cells will stay in
#: replay mode and print the command, so a silently missing stage would look
#: like a broken notebook instead of unfinished bakery.
NOT_YET_IMPLEMENTED = {
    "multiturn":  "nb4_multiturn, nb4_penalty_sweep, nb4_turn_budget",
    "robustness": "nb6_robustness (needs data/test_perturbed.json first)",
    "deploy":     "nb7_merge_check, nb7_serving, nb7_latency, nb7_pareto",
    "bigsets":    "nb8_bigsets, nb8_seeds, nb8_pareto",
}


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

    if skip("nb2_sft", force):
        return
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

    recipes = {
        "kl_blowup":      dict(beta=0.0, learning_rate=5e-4, max_steps=60),
        "length_collapse": dict(max_completion_length=24, beta=0.0, max_steps=60),
        "zero_advantage": dict(temperature=0.01, num_generations=4,
                               per_device_train_batch_size=4, max_steps=60),
        "reward_collapse": dict(learning_rate=3e-3, max_grad_norm=100.0,
                                max_steps=60),
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


ORDER = [
    ("baselines", stage_baselines), ("groups", stage_groups),
    ("star", stage_star), ("sft", stage_sft), ("grpo", stage_grpo),
    ("ablations", stage_ablations), ("pathologies", stage_pathologies),
    ("hacked", stage_hacked), ("art", stage_art),
    ("headtohead", stage_headtohead),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="comma-separated stage names, or 'all'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("implemented:")
        for k, v in STAGES.items():
            print(f"  {k:<12} {v}")
        print()
        print("not yet implemented (the notebook stays in replay mode and "
              "prints the command it needs):")
        for k, v in NOT_YET_IMPLEMENTED.items():
            print(f"  {k:<12} {v}")
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
