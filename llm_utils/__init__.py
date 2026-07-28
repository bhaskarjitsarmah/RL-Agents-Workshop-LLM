"""Shared utilities for the RL-for-LLM-Agents workshop, weight-optimization half.

Repo 1 (`RL-Agents-Workshop`) froze the brain and evolved the harness.
This repo freezes the harness and evolves the brain.

Same task, same 16 held-out tests, same `score_sql`. The environment
(`db.py`), the eval set (`tasks.py`), and the scorer (`evaluate.py`) are
byte-identical vendored copies -- that identity is what makes the head-to-head
in NB8 honest, and `tests/test_vendored_parity.py` enforces it.

Public API (the repo-1 surface, unchanged):
    llm(messages, **kw)        -> str          single chat completion
    METER                      -> CostMeter    call/token/cost tracker
    build_db(path)             -> str          deterministic toy "shop" DB
    load_tasks()               -> list[dict]   the 40 NL -> gold SQL tasks
    score_sql(pred, gold, db)  -> bool         execution-match correctness (THE reward)
    evaluate(agent_fn, ...)    -> dict         run any agent over a split
    make_agent(..., llm_fn=)   -> agent_fn     the looped agent; `llm_fn` is the one addition

Plus this repo's additions -- import them from their modules:
    llm_utils.config      capability(), preflight(), load_result(), base_model()
    llm_utils.gen_tasks   generate_tasks()      the verifiable task generator
    llm_utils.rollout     SQLEnv, Trajectory, rollout_group(), advantages()
    llm_utils.rewards     composite_reward(), make_trl_reward_fns()
    llm_utils.local_llm   LocalLM, make_local_agent()
    llm_utils.datasets    star_sample(), to_sft_dataset(), to_grpo_dataset()
    llm_utils.trainers    t4_sft_config(), t4_grpo_config(), merge_and_save()
    llm_utils.metrics     report_number(), paired_bootstrap(), mcnemar()

The heavy modules (local_llm, trainers, datasets) are NOT imported here on
purpose: they pull in torch/transformers, and a laptop doing data generation or
offline replay must be able to `import llm_utils` without them installed.
"""

from .config import (
    BASE_MODELS, HF_NAMESPACE, MODEL_SIZE, adapter_repo, base_model,
    base_model_4bit, capability, empty_cache, free_vram, gpu_report,
    have_result, load_result, preflight, save_result, setup_colab, torch_dtype,
)
from .llm import (
    CostMeter, METER, LANGFUSE_AVAILABLE, PRICING_PER_1M, embed, flush, llm,
    observe, reset_client,
)
from .db import DB_PATH, SCHEMA_TEXT, build_db, load_tasks, run_sql, score_sql
from .evaluate import evaluate
from .agents import (
    baseline_prompt, extract_sql, make_agent, make_baseline_agent, repair_prompt,
)

__all__ = [
    # repo-1 surface (unchanged)
    "llm", "embed", "METER", "CostMeter", "preflight", "flush", "observe",
    "build_db", "load_tasks", "score_sql", "run_sql", "DB_PATH", "SCHEMA_TEXT",
    "evaluate", "extract_sql", "baseline_prompt", "make_baseline_agent",
    "make_agent", "repair_prompt",
    # this repo's additions
    "capability", "gpu_report", "torch_dtype", "free_vram", "empty_cache",
    "setup_colab", "load_result", "save_result", "have_result",
    "base_model", "base_model_4bit", "adapter_repo", "BASE_MODELS",
    "MODEL_SIZE", "HF_NAMESPACE", "PRICING_PER_1M", "LANGFUSE_AVAILABLE",
    "reset_client",
]
