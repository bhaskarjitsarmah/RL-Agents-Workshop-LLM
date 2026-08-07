# Running on Colab: what will go wrong, and what to do

Everything here is a real failure mode on free-tier Colab, not a hypothetical.

## Your GPU is a T4, and that decides almost everything

A Colab T4 is **Turing, compute capability sm_75**. Three consequences shape
every training config in this repo:

| | consequence |
|---|---|
| **no bfloat16** | `dtype=torch.float16`, and the trainer's `fp16=`/`bf16=` derived from the same flag — see below |
| **no FlashAttention-2** | `attn_implementation="sdpa"`; FA2 needs sm_80+ |
| **vLLM is unreliable** | `use_vllm=False` and `fast_inference=False` by default; NB0–NB4 run on plain HF `generate` and accept the 2–4× slowdown |

### The dtype rule: decide once, read everywhere

`llm_utils/config.torch_dtype()` is the single place the fp16-vs-bf16 decision is
made, and `_precision_flags()` in `llm_utils/trainers.py` derives the trainer's
`fp16=`/`bf16=` from that same flag. **Never set them by hand on a config.**

That is not style. When the model loads in one dtype and the trainer is
configured for the other, `fp16=True` makes Trainer install a `GradScaler` whose
CUDA kernel exists only for `Half` and `Float`, and the first unscale dies with:

```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
not implemented for 'BFloat16'
```

Two things used to allow that mismatch, both now closed: the configs hardcoded
`fp16=True`, and `from_pretrained(torch_dtype=...)` was **silently ignored** on
transformers v5 (5.5.0 on Colab as of Aug 2026), which renamed the argument to
`dtype=` — so the model loaded at the checkpoint's bfloat16 no matter what was
asked for. `_dtype_kwarg()` picks the right name per version, and
`_assert_dtype()` fails at load time rather than mid-run.

On a T4 you will see `bf16=False dtype=float16`. Note that `bf16` here comes from
`torch.cuda.is_bf16_supported()`, whose answer for Turing has changed across
torch releases — which is exactly why nothing else in the repo is allowed to
form its own opinion about dtype.

**fp16 + LoRA can also diverge into non-finite gradients**, silently — you get a
flat loss curve rather than an error, which reads as "the model didn't learn".
`non_finite_loss_callback()` aborts loudly instead. Do not remove it.

## Installing Unsloth replaces torch → restart required

The setup cell detects this and prints:

```
torch was replaced during install (2.5.1 -> 2.4.0).
RESTART THE RUNTIME NOW (Runtime -> Restart session), then re-run this cell.
```

**Do exactly that.** If you press on, you get a cryptic CUDA error three cells
later that looks like a code bug. Re-running the setup cell after the restart
skips the install and continues.

## Restart the runtime between notebooks

Colab does not free GPU memory when you open a different notebook. A leftover
1.5B model from the previous one is the most common cause of an OOM halfway
through a training run.

Every notebook opens with a reminder. If you skip it:

```python
lm.unload()                      # frees the model
from llm_utils import empty_cache, free_vram
empty_cache(); print(free_vram(), "GB free")
```

## VRAM budget

1.5B in 4-bit + LoRA + GRPO at G=8 is roughly **4.5–6 GB of ~15 GB**.
Comfortable. Three ways to blow it:

1. turning off gradient checkpointing
2. `num_generations=16` together with a long `max_completion_length`
3. materialising a separate fp16 reference model

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if you see fragmentation
errors despite having free memory.

## Disconnects

Free Colab disconnects after roughly 90 minutes of no interaction, GPU quota can
be revoked mid-session, and closing the tab ends everything.

What the repo does about it:

- **No live training cell runs longer than ~25 minutes** — except the lunch GRPO
  run, which checkpoints every 25 steps.
- Every downstream cell can load the **pre-baked adapter** instead of the one you
  just trained. A participant who loses their runtime at 14:00 still completes
  NB5–NB8.
- Stages in `scripts/bake_all.py` are independent and resumable, so a disconnect
  costs one stage rather than the day.

**Before you run anything: File → Save a copy in Drive.** Otherwise your edits
die with the runtime.

### I got disconnected mid-training

```python
# 1. Re-run the setup cell (it is idempotent).
# 2. Resume from the last checkpoint:
trainer.train(resume_from_checkpoint=True)

# or just move on with the pre-baked adapter:
from llm_utils.config import adapter_repo
lm = LocalLM(adapter=adapter_repo("grpo"))
```

## No GPU at all?

Everything still works, in **replay mode**. `capability()["gpu"]` is False, every
training cell short-circuits, and every chart renders from the pre-baked runs in
`data/results/` — watermarked *pre-baked replay* so you always know whose run you
are looking at.

Fully CPU-runnable: **NB0, NB1, NB2's data half, NB3's from-scratch GRPO, NB6's
analysis, and NB8.**

You can also run *live* rollouts with no GPU by pointing at any
OpenAI-compatible endpoint:

```python
lm = LocalLM(backend="openai", base_url="https://...", api_model="...")
```

## Slow runtime? Drop to 0.5B

```python
import os; os.environ["MODEL_SIZE"] = "0.5B"   # before importing llm_utils
```

Roughly halves every training time. Accuracy is lower, the arc is identical, and
the 0.5B adapters are pre-baked too.

## Before you touch any training code

```bash
python scripts/check_api_surface.py
```

It prints the **actual** dataclass fields of `SFTConfig` / `GRPOConfig` and ART's
namespace for the versions Colab resolved today. TRL's field names churn between
minor versions; correct `llm_utils/trainers.py` against that output rather than
against documentation or memory.

If it reports dropped kwargs, fix them before training — one of them
(`remove_unused_columns=False`) silently causes the run to optimise nothing.
