# Workshop Runbook — Run This on a Free T4 (Colab or Kaggle)

No-nonsense, copy-paste setup for **Self-Improving Agents by Optimizing the
Weights**. Follow it top to bottom. It works on a free Google Colab T4 and on a
free Kaggle T4 — pick **one** track below.

> **TL;DR** — Open a notebook in Colab → set runtime to **T4 GPU** →
> **Runtime → Run all**. The first cell installs everything (~3 min). If that
> very first run stops with a torch message, **Restart session** once and **Run
> all** again. Restart the runtime before each new notebook. That's the whole game.

The nine notebooks are meant to be run **in order, NB0 → NB8**. Each one is
self-contained: the first cell clones the repo and installs the pinned stack (with
`uv`) on its own. You do **not** install anything on your laptop, and there are
**no manual pip commands** — it's handled for you.

---

## What you need before you start

| Thing | Required? | Notes |
|---|---|---|
| A Google account | ✅ (for Colab) | Colab is free; a T4 is not *guaranteed* on free tier but usually available. |
| A Weights & Biases key | ⭐ Recommended | Free at <https://wandb.ai/authorize>. Without it, training still runs — curves just save locally instead of the cloud dashboard. |
| An OpenAI key | ❌ Optional | Only for the `gpt-4o-mini` comparison rows in NB0 / NB8. Without it, the published number is used instead. A few cents if you do use it. |
| A Hugging Face token | ❌ Optional | Only if you want to push **your own** trained adapter. The pre-baked ones are public. |

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

1. Click the **🔑 key icon** in the left sidebar.
2. **+ Add new secret**, name `WANDB_API_KEY`, paste your key, enable *Notebook
   access*. (Repeat for `OPENAI_API_KEY` if you want the `gpt-4o-mini` rows.)

> ⚠️ **Adding a secret is not enough.** Colab keeps secrets locked away — a key
> in the panel is **not** in the environment until your code pulls it out with
> `userdata.get(...)`. And the `CAP` variable (which decides whether the
> `gpt-4o-mini` / W&B cells fire) is computed **once, in the setup cell**. So you
> must load the key **and then re-run the setup cell**, or `CAP` stays stale and
> the key looks ignored.

…paste this as a **new cell at the very top**, above the setup cell, run it, then
run the setup cell:

```python
import os
from google.colab import userdata
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
# Optional — only if you want the gpt-4o-mini comparison rows (NB0 / NB8):
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

**Already ran the setup cell and the key was ignored?** Don't re-run everything —
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

### 5. Run all

**Runtime → Run all.** The first cell installs the pinned stack with `uv`
(~3 minutes the first time) and then reports your GPU. A healthy first cell ends
with:

```
GPU: Tesla T4  sm_75  14.7 GB  bf16=False  dtype=float16
```

`bf16=False` is **correct** — a T4 is a Turing card and only does fp16. Every
config in the repo depends on that.

### 6. If the first run stops with a torch error — restart once

On a fresh runtime this usually just works. But if the *first* run ever stops with
a torch-related error (it changed torch underneath a already-imported copy), do
this **once**:

**Runtime → Restart session → Run all** again. The second time, everything is
already installed, so it's fast and clean. You will not need to do this again for
that notebook.

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
# uv, not pip: it resolves the pinned stack cleanly (same as the Colab setup cell).
os.system(f"{sys.executable} -m pip install -q uv")
os.system(f"{sys.executable} -m uv pip install --system -q -r requirements-colab.txt")

# Optional: W&B (add these as Kaggle "Secrets" under Add-ons -> Secrets)
# from kaggle_secrets import UserSecretsClient
# os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Run it. **If torch gets replaced, restart the kernel** (top menu → **Run →
Restart kernel**) and re-run this cell — same rule as Colab.

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
| A later cell says `bitsandbytes>=… required` / `torchvision::nms does not exist` / `unexpected keyword argument 'dtype'` | The install didn't fully take (rare, on a messy runtime). | **Runtime → Restart session → Run all**. On a fresh runtime the `uv` install is clean. |
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
  confirm you see `GPU: Tesla T4 … bf16=False`.
- **Do a dry run of NB0 → NB3** end to end on a fresh Colab runtime, timing each.
- Have your **W&B key** and (optionally) **OpenAI key** ready as Colab Secrets.

> ⚠️ **Known gap to close before the workshop:** the repo's `.gitignore` says the
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
