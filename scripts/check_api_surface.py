"""Print the ACTUAL API surface of the installed training libraries.

Run this first, on the machine that will run the workshop:

    python scripts/check_api_surface.py

Why this exists
---------------
TRL's `SFTConfig` / `GRPOConfig` field names have been added, renamed, and
removed repeatedly across minor versions (`max_seq_length` vs `max_length`,
`loss_type`, `scale_rewards`, `epsilon_high`, `mask_truncated_completions`,
`num_iterations`, `vllm_mode`, `sync_ref_model`...). OpenPipe ART is younger and
moves faster still. Any config this repo builds from memory or from documentation
is a guess.

So: don't guess. `trainers.py` filters unknown kwargs at runtime and reports what
it dropped, and this script tells you the ground truth ahead of time. **If it
reports dropped fields, correct `trainers.py` against this output** -- a silently
dropped `remove_unused_columns=False` means the reward functions never receive
the `gold` column and accuracy sits at chance with no error message.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_versions() -> None:
    rule("VERSIONS")
    from llm_utils.config import gpu_report

    rep = gpu_report()
    for k in ("torch", "transformers", "trl", "peft", "unsloth", "bitsandbytes",
              "datasets", "accelerate"):
        print(f"  {k:<16} {rep.get(k) or '-- not installed --'}")
    print(f"\n  GPU        {rep['name'] or 'none'}")
    if rep["gpu"]:
        cap = rep["capability"]
        print(f"  capability sm_{cap[0]}{cap[1]}"
              f"   bf16={rep['bf16']}   flash_attn={rep['flash_attn']}"
              f"   vllm_ok={rep['vllm_ok']}")
        print(f"  memory     {rep['total_gb']} GB total, {rep['free_gb']} GB free")
        if not rep["bf16"]:
            print("\n  -> fp16 ONLY. Every config must set fp16=True, bf16=False.")
        if not rep["vllm_ok"]:
            print("  -> vLLM/FlashAttention unsupported here: keep use_vllm=False "
                  "and fast_inference=False.")


def show_dataclass_fields(cls, name: str) -> set[str]:
    print(f"\n--- {name} ({cls.__module__}.{cls.__qualname__}) ---")
    if not dataclasses.is_dataclass(cls):
        print("  not a dataclass; falling back to __init__ signature")
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        for p in sorted(params):
            print(f"  {p}")
        return params
    names = {f.name for f in dataclasses.fields(cls)}
    for f in sorted(dataclasses.fields(cls), key=lambda x: x.name):
        default = "" if f.default is dataclasses.MISSING else f"= {f.default!r}"
        print(f"  {f.name:<38} {default}")
    return names


# The kwargs trainers.py wants to set. Anything reported MISSING here must be
# removed from (or renamed in) trainers.py.
WANTED_GRPO = {
    "learning_rate", "lr_scheduler_type", "warmup_ratio", "adam_beta1",
    "adam_beta2", "weight_decay", "max_grad_norm", "optim",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "num_generations", "max_prompt_length", "max_completion_length",
    "temperature", "top_p", "beta", "loss_type", "scale_rewards", "epsilon",
    "num_iterations", "fp16", "bf16", "gradient_checkpointing", "max_steps",
    "logging_steps", "save_steps", "report_to", "use_vllm",
    "remove_unused_columns", "output_dir", "seed",
    "mask_truncated_completions", "sync_ref_model", "log_completions",
}
WANTED_SFT = {
    "learning_rate", "lr_scheduler_type", "warmup_ratio", "weight_decay",
    "max_grad_norm", "optim", "per_device_train_batch_size",
    "gradient_accumulation_steps", "num_train_epochs", "fp16", "bf16",
    "gradient_checkpointing", "logging_steps", "save_steps", "report_to",
    "output_dir", "seed", "max_seq_length", "max_length", "packing",
    "completion_only_loss", "dataset_text_field",
}


def check(cls, wanted: set[str], name: str) -> None:
    have = show_dataclass_fields(cls, name)
    missing = sorted(wanted - have)
    print(f"\n  WANTED but ABSENT in this version ({len(missing)}):")
    if missing:
        for m in missing:
            print(f"    !! {m}")
        print("\n  -> remove/rename these in llm_utils/trainers.py.")
    else:
        print("    (none -- trainers.py's kwargs are all valid here)")


def show_trl() -> None:
    rule("TRL CONFIGS")
    try:
        from trl import GRPOConfig, SFTConfig
    except Exception as e:  # noqa: BLE001
        print(f"  could not import trl configs: {e}")
        return
    check(SFTConfig, WANTED_SFT, "SFTConfig")
    check(GRPOConfig, WANTED_GRPO, "GRPOConfig")

    rule("GRPOTrainer SIGNATURE")
    try:
        from trl import GRPOTrainer

        print(inspect.signature(GRPOTrainer.__init__))
        print("\nreward_funcs docstring excerpt:")
        doc = (GRPOTrainer.__doc__ or "")
        idx = doc.find("reward_funcs")
        print("  " + (doc[idx:idx + 700].replace("\n", "\n  ") if idx >= 0
                      else "(not found -- inspect manually)"))
    except Exception as e:  # noqa: BLE001
        print(f"  {e}")


def show_art() -> None:
    rule("OPENPIPE ART")
    try:
        import art
    except Exception as e:  # noqa: BLE001
        print(f"  not installed / import failed: {e}")
        print("  -> NB5 runs in replay mode. See README's ART risk note.")
        return
    print("  version:", getattr(art, "__version__", "unknown"))
    print("  public names:", sorted(n for n in dir(art) if not n.startswith("_")))
    for cls_name in ("TrainableModel", "Trajectory", "TrajectoryGroup",
                     "TrainConfig", "LocalBackend"):
        obj = getattr(art, cls_name, None)
        if obj is None:
            print(f"\n  {cls_name}: NOT on the top-level `art` namespace "
                  "-- find its real import path before writing NB5.")
            continue
        try:
            print(f"\n  art.{cls_name}{inspect.signature(obj)}")
        except (TypeError, ValueError):
            print(f"\n  art.{cls_name}: (no introspectable signature)")


def show_unsloth() -> None:
    rule("UNSLOTH")
    try:
        from unsloth import FastLanguageModel
    except Exception as e:  # noqa: BLE001
        print(f"  not installed / import failed: {e}")
        print("  -> trainers.load_4bit_policy falls back to the "
              "BitsAndBytes + peft path.")
        return
    for fn in ("from_pretrained", "get_peft_model"):
        f = getattr(FastLanguageModel, fn, None)
        if f is None:
            print(f"  FastLanguageModel.{fn} MISSING")
            continue
        try:
            print(f"\n  FastLanguageModel.{fn}{inspect.signature(f)}")
        except (TypeError, ValueError):
            print(f"\n  FastLanguageModel.{fn}: (no introspectable signature)")


if __name__ == "__main__":
    show_versions()
    show_unsloth()
    show_trl()
    show_art()
    rule("DONE")
    print("Correct llm_utils/trainers.py against the output above before\n"
          "authoring notebooks. Do not trust remembered field names.")
