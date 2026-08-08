# Workshop Runbook — Run This on a Free T4 (Colab or Kaggle)

NB0 measures the gap, NB1–NB3 close it by moving weights, NB4–NB5 extend it to tool loops, NB6 shows how it goes wrong, NB7–NB8 ship it.

No-nonsense, copy-paste setup for **Self-Improving Agents by Optimizing the
Weights**. Follow it top to bottom. It works on a free Google Colab T4 and on a
free Kaggle T4 — pick **one** track below.

> **TL;DR** — Open a notebook in Colab → set runtime to **T4 GPU** → **Runtime →
> Run all**. Restart the runtime before each new notebook. That's the whole game.

The nine notebooks are meant to be run **in order, NB0 → NB8**. Each one is
self-contained: the first cell clones the repo, installs the stack, reads your
keys, and prepares the data. Nothing is pasted in by hand.

---

## What you need before you start

| Thing | Required? | Notes |
|---|---|---|
| A Google account | **Required** (for Colab) | Colab is free; a T4 is not *guaranteed* on free tier but usually available. |
| A Weights & Biases key | Recommended | Free at <https://wandb.ai/authorize>. Without it, training still runs — curves just save locally instead of the cloud dashboard. |
| An OpenAI key | Optional | Only for the `gpt-4o-mini` comparison rows in NB0 / NB8. Without it, the published number is used instead. A few cents if you do use it. |
| A Hugging Face token | Optional | Only if you want to push **your own** trained adapter. The pre-baked ones are public. |

**You do not need:** a GPU of your own, a vector database, Langfuse, or any paid
service. The training loop makes **zero API calls** — the model runs on the free
T4 and the reward is a local SQLite query.

---

# TRACK A — Google Colab (recommended)

### 1. Open the notebook

Click a badge below (opens straight in Colab), or go to
<https://colab.research.google.com> → **GitHub** tab → paste
`bhaskarjitsarmah/RL-Agents-Workshop-LLM` → pick a notebook from the list.

| # | Notebook | Open |
|---|---|---|
| NB0 | Two agents, one scoreboard | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB0_two_agents_one_scoreboard.ipynb) |
| NB1 | The MDP | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB1_the_mdp.ipynb) |
| NB2 | Data + STaR warm start | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB2_data_and_star_warm_start.ipynb) |
| NB3 | GRPO | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB3_grpo.ipynb) |
| NB4 | Multi-turn RL | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB4_multi_turn_rl.ipynb) |
| NB5 | OpenPipe ART | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB5_openpipe_art.ipynb) |
| NB6 | Reward hacking & safety | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB6_reward_hacking_and_safety.ipynb) |
| NB7 | Deployment | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB7_deployment.ipynb) |
| NB8 | Capstone: weights vs harness | [Colab](https://colab.research.google.com/github/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB8_capstone_weights_vs_harness.ipynb) |

### 2. Save your own copy

**File → Save a copy in Drive.** Do this first. If you don't, your edits die
when the runtime disconnects.

### 3. Turn on the T4 GPU

**Runtime → Change runtime type → T4 GPU → Save.**

(The notebooks already request a T4, but confirm it — landing on a CPU runtime is
the #1 avoidable mistake.)

### 4. (Recommended) Add your keys

Easiest way — Colab Secrets:

1. Click the **key icon** in the left sidebar.
2. **+ Add new secret**, name `WANDB_API_KEY`, paste your key, enable *Notebook
   access*. (Repeat for `OPENAI_API_KEY` if you want the `gpt-4o-mini` rows.)

That's all. The setup cell reads the secrets itself. Two rules:

- *Notebook access* must be **on** for each secret, or Colab hides it.
- Add secrets **before** running the setup cell. It reads them once; add one
  afterwards and it looks ignored. Fix by re-running the setup cell.

**Added the key after the setup cell ran?** Don't re-run everything —
fix `CAP` in place with this cell, then re-run the `gpt-4o-mini` cell:

```python
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

from llm_utils import capability
CAP = capability()                              # recompute with the key now present
print("openai visible now:", CAP["openai"])     # should print True
```

Sanity check that Colab can even see the secret:

```python
from google.colab import userdata
print(bool(userdata.get("OPENAI_API_KEY")))     # True = name + Notebook-access toggle are correct
```

**No W&B key?** Skip it entirely. Training still runs; the setup cell
automatically switches W&B to offline mode and tells you so.

### 5. Nothing to install by hand

The notebook's first cell clones the repo, installs everything, reads your Colab
Secrets, and prepares the data. Do not add a cell. Do not run `!pip install`.

It takes 1–2 minutes the first time. Two of its lines are best-effort and may
print a failure — that is fine and expected:

- **unsloth** — a 2x speedup. Without it, training runs on transformers +
  bitsandbytes and produces the same adapter, just slower.
- **openpipe-art** — NB5 only. Without it, NB5 replays a pre-baked run.

If the **core** install fails, the cell stops with `CORE INSTALL FAILED`. Restart
the session and run again.

### 6. Then Run all

**Runtime → Run all.** The setup cell reports your GPU; a healthy run shows:

```
GPU: Tesla T4  sm_75  14.56 GB  bf16=False  dtype=float16
versions: torch=2.11.0+cu128  transformers=5.5.0  trl=0.24.0  peft=0.19.1  unsloth=...
keys present: WANDB_API_KEY
```

`bf16=False` on a T4 is correct — it's a Turing card. Never set `fp16=` or
`bf16=` on a training config by hand; the notebooks derive both from that one
flag, and overriding it is what causes the `BFloat16` crash below.

Read the markdown between the cells as they run — that's the workshop. No live
training cell runs longer than ~25 minutes.

### 7. Before opening the NEXT notebook — restart

Colab does **not** free GPU memory when you switch notebooks, and a leftover 1.5B
model is the most common cause of an out-of-memory crash mid-training.

**Runtime → Restart session** before starting each new notebook. (Or, to stay in
the same one, run `lm.unload()` and then `empty_cache()`.)

---

# TRACK B — Kaggle (T4 ×2, also free)

Kaggle works, but the notebooks' **auto-setup only fires on Colab** — on Kaggle
you must clone + install yourself in one extra cell first. Do this:

### 1. Create the notebook

- <https://www.kaggle.com/code> → **+ New Notebook**.
- Right sidebar → **Settings**:
  - **Accelerator → GPU T4 ×2** (or P100).
  - **Internet → On** ← required, or the pip install and model download fail.

### 2. Import the workshop notebook

**File → Import Notebook → GitHub / URL**, and paste the raw URL of the notebook
you want, e.g.:

```
https://github.com/bhaskarjitsarmah/RL-Agents-Workshop-LLM/blob/main/notebooks/NB0_two_agents_one_scoreboard.ipynb
```

### 3. Add ONE cell at the very top, before everything else

```python
import os, sys
# Clone the repo and install the GPU stack (Kaggle doesn't auto-clone like Colab).
if not os.path.exists("/kaggle/working/RL-Agents-Workshop-LLM"):
    os.system("git clone -q https://github.com/bhaskarjitsarmah/RL-Agents-Workshop-LLM.git /kaggle/working/RL-Agents-Workshop-LLM")
os.chdir("/kaggle/working/RL-Agents-Workshop-LLM")
# uv, not pip: same resolution, much faster (identical to the Colab setup cell).
os.system(f"{sys.executable} -m pip install -q uv")
os.system(f"{sys.executable} -m uv pip install --system -q -r requirements-colab.txt")
# Optional speedup, best-effort. If it fails, training still runs, just slower.
os.system(f"{sys.executable} -m uv pip install --system -q unsloth unsloth_zoo")

# Optional: W&B (add these as Kaggle "Secrets" under Add-ons -> Secrets)
# from kaggle_secrets import UserSecretsClient
# os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Run it. If it reports a new torch version, **restart the kernel** (top menu →
**Run → Restart kernel**) and re-run it. Only the unsloth line can cause that;
drop it if you'd rather not risk the restart.

### 4. Run the notebook's own setup cell, then the rest

The repo's setup cell will detect it's *not* on Colab and just adjust the working
directory — that's expected. Because you already cloned into
`/kaggle/working/RL-Agents-Workshop-LLM` and `%cd`'d in, all the `data/...` paths
resolve correctly. Continue top to bottom.

### 5. Restart the kernel between notebooks

Same reason as Colab — free the GPU memory.

---

## The notebook order (and what each one delivers)

Run them in sequence. Roughly a full day; each is 20–40 min of runtime plus
discussion.

| # | Title | You leave with |
|---|---|---|
| **NB0** | Two agents, one scoreboard | The gap: a 1.5B model vs `gpt-4o-mini` on the same 16 tasks. |
| **NB1** | The MDP | State / action / reward / trajectory — where the learning signal comes from. |
| **NB2** | Data + STaR warm start | A verifiable task generator and a bootstrapped warm start. |
| **NB3** | GRPO | The core training run. **This is the "lunch run"** — it checkpoints every 25 steps. |
| **NB4** | Multi-turn RL | Training the tool loop; masking tool output out of the loss. |
| **NB5** | OpenPipe ART | The same experiment in a production wrapper, with W&B. |
| **NB6** | Reward hacking & safety | The reward as an attack surface, plus safety gates. |
| **NB7** | Deployment | Merge, serve, and measure latency and \$/1k queries. |
| **NB8** | Capstone | Weights vs harness, head to head — and the hybrid you'd ship. |

---

## Troubleshooting — the failures that actually happen

| Symptom | Cause | Fix |
|---|---|---|
| `bitsandbytes>=… required` / `No module named 'trl'` / `torchvision::nms does not exist` | The install didn't take — you're on Colab's stock packages. | **Runtime → Restart session → Run all**. Don't paste `!pip install` cells to patch around it. |
| `NotImplementedError: … not implemented for 'BFloat16'` | Old clone: model and trainer disagree on dtype. | `!rm -rf /content/RL-Agents-Workshop-LLM`, then re-run the setup cell to get the current code. |
| `[trainers] unsloth did not load …` | — | **Not an error.** Training continues, ~2x slower, same result. |
| `GPU: none -> replay mode` | You're on a CPU runtime. | Runtime → Change runtime type → **T4 GPU**, re-run the setup cell. |
| Cryptic **CUDA error** a few cells in | You skipped the torch-replaced restart. | Runtime → **Restart session**, re-run the setup cell, then continue. |
| **Out of memory** mid-training | Leftover model from the previous notebook. | Restart the runtime before each notebook. In-place: `lm.unload(); empty_cache()`. |
| **Disconnected** mid-training | Free-tier idle timeout (~90 min) or quota. | Re-run the setup cell (idempotent), then `trainer.train(resume_from_checkpoint=True)`, or move on with the pre-baked adapter. |
| Runtime feels very slow | Throttled free tier. | Before the first import: `import os; os.environ["MODEL_SIZE"] = "0.5B"`. Halves every training time; same lesson. |
| W&B asks you to log in / says offline | No `WANDB_API_KEY`. | Fine — offline mode is automatic. Add the key (Track A step 4) to get cloud curves. |
| `No OPENAI_API_KEY -> using repo 1's published result` **even though you added the Colab secret** | The secret is in the panel but not in `os.environ`, and/or `CAP` was computed before you loaded it. | Load it with `userdata.get(...)`, then **re-run the setup cell** — or recompute `CAP = capability()` in place (Track A step 4). Also confirm the secret is named exactly `OPENAI_API_KEY` with *Notebook access* ON. |
| Fragmentation error despite free VRAM | Memory fragmentation. | `import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` before importing. |
| `[some_key] not baked yet` printed instead of a chart | A pre-baked artifact is missing (see note below). | On a T4 most cells run live. If a replay cell is missing its artifact, run the live path above it, or see the presenter note. |

---

## For the presenter — do this the day before

- **Run the Colab GPU check the day before.** It's the only step that can fail in
  a way you can't fix in the room. Open NB0's badge, set T4, run the first cell,
  confirm `GPU: Tesla T4 …` and a `versions:` line listing `trl` and `peft`. A
  missing `versions:` entry means the install didn't take.
- **Run `python scripts/check_api_surface.py` on that same runtime** — it prints
  the real `SFTConfig`/`GRPOConfig` fields, in case TRL renamed one.
- **Do a dry run of NB0 → NB3** end to end on a fresh Colab runtime, timing each.
- Have your **W&B key** and (optionally) **OpenAI key** ready as Colab Secrets.

> **Known gap to close before the workshop:** the repo's `.gitignore` says the
> pre-baked replay results in `data/results/*.json` are checked in, but **they
> currently are not** (the directory is empty). On a live T4 this rarely bites,
> because training cells run for real. But anyone in **replay mode** (no GPU), or
> any cell that falls back to a pre-baked run on T4 (e.g. parts of NB5/NB6), will
> print `not baked yet` instead of a chart. To fix it, run the bake job once on a
> GPU and commit the output:
>
> ```bash
> python scripts/bake_all.py --stage all     # ~8-12 GPU-hours; or bake key stages
> git add data/results && git commit -m "Ship pre-baked replay results"
> ```
>
> `python scripts/bake_all.py --list` shows what each stage writes.

---

*Questions during the workshop: start with the troubleshooting table above — it
covers essentially every real free-tier failure. See also [SETUP.md](SETUP.md)
and [COLAB.md](COLAB.md) for the deeper explanations.*
