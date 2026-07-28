"""T4-safe training configuration, LoRA loading, and adapter export.

Everything here is shaped by one hardware fact: **a Colab T4 is Turing (sm_75)**.

    no bfloat16          -> fp16=True, bf16=False, dtype=torch.float16 explicitly
    no FlashAttention-2  -> attn_implementation="sdpa"
    vLLM unreliable      -> use_vllm=False, fast_inference=False

fp16 + LoRA can produce non-finite gradients and a stuck GradScaler, which
manifests as a *flat* loss curve rather than an error -- the worst kind of
failure, because it looks like "the model didn't learn". `NonFiniteLossCallback`
aborts loudly instead.

On API churn
------------
TRL's `SFTConfig`/`GRPOConfig` field names have been added, renamed, and removed
repeatedly (`max_seq_length` vs `max_length`, `loss_type`, `scale_rewards`,
`epsilon_high`, `mask_truncated_completions`, `num_iterations`, `vllm_mode`,
`sync_ref_model`). Any config written from memory is a guess.

So `_filter_kwargs` drops kwargs the installed version does not define **and
prints what it dropped**. Run `python scripts/check_api_surface.py` first and
correct this file against its output. One dropped kwarg matters more than the
rest: `remove_unused_columns=False`. Without it TRL silently drops the `gold`
column, the reward functions receive nothing, and training optimises noise while
reporting no error at all -- so it is asserted rather than merely requested.
"""

from __future__ import annotations

import dataclasses
import os

from .config import ADAPTER_DIR, base_model, base_model_4bit, gpu_report, torch_dtype


def _filter_kwargs(cls, kw: dict, verbose: bool = True) -> dict:
    """Keep only kwargs the installed config dataclass actually defines."""
    try:
        valid = {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        import inspect

        valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kept = {k: v for k, v in kw.items() if k in valid}
    dropped = sorted(set(kw) - set(kept))
    if dropped and verbose:
        print(f"[trainers] {cls.__name__}: dropped unsupported kwargs {dropped}\n"
              f"           (installed version does not define them -- run "
              f"scripts/check_api_surface.py and update trainers.py)")
    return kept


# ===========================================================================
# Policy loading
# ===========================================================================

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_4bit_policy(model_id: str | None = None, r: int = 16,
                     lora_alpha: int = 32, lora_dropout: float = 0.0,
                     use_unsloth: bool = True, max_seq_len: int = 2048,
                     gradient_checkpointing: bool = True):
    """Load the base model in 4-bit with a LoRA adapter attached.

    Returns (model, tokenizer). theta is ~18M adapter parameters over 7
    projections -- about 36MB in fp16, which is why the pre-baked adapters are
    trivial to host and download mid-session.

    Two paths on purpose: Unsloth is faster, and its install occasionally fails
    on the day. The notebook prints which one it took.
    """
    import torch

    dtype = torch_dtype()
    mid = model_id or base_model_4bit()

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tok = FastLanguageModel.from_pretrained(
                model_name=mid, max_seq_length=max_seq_len, dtype=dtype,
                load_in_4bit=True, fast_inference=False,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", target_modules=LORA_TARGETS,
                use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
                random_state=3407,
            )
            print(f"[trainers] unsloth path, dtype={dtype}, model={mid}")
            _prep_tokenizer(tok)
            return model, tok
        except Exception as e:  # noqa: BLE001
            print(f"[trainers] unsloth unavailable ({e}); falling back to "
                  "transformers + bitsandbytes")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base_model())
    model = AutoModelForCausalLM.from_pretrained(
        base_model(), quantization_config=quant, dtype=dtype,
        device_map="auto", attn_implementation="sdpa")
    # Keeps LayerNorms and the LM head in fp32 -- the main defence against
    # fp16 LoRA producing NaN gradients on Turing.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing)
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout or 0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.config.use_cache = False   # incompatible with gradient checkpointing
    print(f"[trainers] hf path, dtype={dtype}, model={base_model()}")
    _prep_tokenizer(tok)
    return model, tok


def _prep_tokenizer(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # TRAINING. Flip to "left" for generation.
    return tok


# ===========================================================================
# Configs
# ===========================================================================

def t4_sft_config(output_dir: str, **overrides):
    """SFT on the STaR data. Conservative because fp16 is unforgiving."""
    from trl import SFTConfig

    kw = dict(
        output_dir=output_dir,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, max_grad_norm=0.3, optim="paged_adamw_8bit",
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=2,
        fp16=True, bf16=False, gradient_checkpointing=True,
        logging_steps=5, save_steps=100, save_total_limit=2,
        report_to="wandb", seed=3407,
        max_length=1024, max_seq_length=1024,   # name churned; filter keeps one
        packing=False,
    )
    kw.update(overrides)
    return SFTConfig(**_filter_kwargs(SFTConfig, kw))


def t4_grpo_config(output_dir: str, num_generations: int = 8, **overrides):
    """GRPO on the verifiable reward.

    `per_device_train_batch_size == num_generations` so one optimizer step sees
    whole groups: a partial group would compute its advantage against an
    incomplete baseline, which is subtly wrong and very hard to notice.
    """
    from trl import GRPOConfig

    kw = dict(
        output_dir=output_dir,
        learning_rate=1e-5, lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
        max_grad_norm=0.3, optim="paged_adamw_8bit",
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=2,
        num_generations=num_generations,
        max_prompt_length=768, max_completion_length=192,
        temperature=0.9, top_p=1.0,
        beta=0.02,                       # KL anchor to the frozen reference
        loss_type="dr_grpo", scale_rewards=False,
        epsilon=0.2, num_iterations=1,
        fp16=True, bf16=False, gradient_checkpointing=True,
        max_steps=150, logging_steps=1, save_steps=25, save_total_limit=3,
        report_to="wandb", use_vllm=False, log_completions=True,
        seed=3407,
        # NON-NEGOTIABLE: without this TRL drops the `gold` column and the
        # reward functions silently receive nothing.
        remove_unused_columns=False,
    )
    kw.update(overrides)
    filtered = _filter_kwargs(GRPOConfig, kw)
    if "remove_unused_columns" not in filtered:
        raise RuntimeError(
            "This TRL version has no `remove_unused_columns` on GRPOConfig. "
            "Verify how it passes dataset columns to reward functions before "
            "training -- if `gold` does not reach them, the run optimises "
            "nothing and reports no error.")
    if filtered.get("remove_unused_columns") is not False:
        raise RuntimeError("remove_unused_columns must be False")
    return GRPOConfig(**filtered)


# ===========================================================================
# Guardrails
# ===========================================================================

def non_finite_loss_callback():
    """Abort loudly on a NaN/Inf loss instead of producing a flat curve.

    fp16 + LoRA on Turing can wedge the GradScaler. Silent divergence is the
    expensive failure mode here: the curve looks disappointing rather than
    broken, and a participant spends the afternoon tuning a dead run.
    """
    from transformers import TrainerCallback

    class NonFiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            import math

            loss = (logs or {}).get("loss")
            if loss is not None and not math.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss ({loss}) at step {state.global_step}. "
                    "fp16 + LoRA has diverged. Lower the learning rate, check "
                    "max_grad_norm, and confirm dtype is float16 on this GPU.")
    return NonFiniteLoss()


def vram_budget(label: str = "") -> dict:
    """Print measured VRAM. Call before and after a run; compare to the estimate.

    Budget for 1.5B 4-bit + LoRA + GRPO(G=8) is ~4.5-6 GB of ~15 GB. The three
    ways to blow it: disabling gradient checkpointing, num_generations=16 with
    long completions, and materialising a separate fp16 reference model.
    """
    import torch

    if not torch.cuda.is_available():
        return {"gpu": False}
    free, total = torch.cuda.mem_get_info()
    rep = {"free_gb": round(free / 1024**3, 2),
           "total_gb": round(total / 1024**3, 2),
           "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    print(f"[vram{' ' + label if label else ''}] "
          f"allocated {rep['allocated_gb']} GB, peak {rep['peak_gb']} GB, "
          f"free {rep['free_gb']}/{rep['total_gb']} GB")
    return rep


# ===========================================================================
# Export
# ===========================================================================

def merge_and_save(adapter_dir: str, out_dir: str, base_id: str | None = None,
                   dtype: str = "float16") -> str:
    """Merge a LoRA adapter into fp16 base weights for serving.

    NB7 re-evaluates the merged model and must reproduce the adapter's test-16
    accuracy EXACTLY. A merge bug shows up there or never -- it is silent
    otherwise, because the merged model still produces fluent SQL.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = base_id or base_model()
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=getattr(torch, dtype), device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[trainers] merged {adapter_dir} -> {out_dir} ({dtype})")
    return out_dir


def push_adapter(adapter_dir: str, repo_id: str, token: str | None = None,
                 private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(folder_path=adapter_dir, repo_id=repo_id)
    print(f"[trainers] pushed {adapter_dir} -> https://huggingface.co/{repo_id}")
    return repo_id


def adapter_path(tag: str) -> str:
    return os.path.join(ADAPTER_DIR, tag)
