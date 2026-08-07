# Pre-work: 10 minutes, before the workshop

Shorter than repo 1's setup, because this repo needs almost nothing. The policy
runs on your GPU and the reward is a local SQLite query, so **the training loop
makes no API calls at all**.

> **Do the Colab check (step 3) the day before.** It is the only step that can
> fail in a way you cannot fix in the room.

## 1. Weights & Biases — the only required key

This is where the GRPO curves live, and reading them is half of NB3.

1. Sign up at <https://wandb.ai>
2. Copy your key from <https://wandb.ai/authorize>
3. Put it in `.env` as `WANDB_API_KEY`

No account? Set `WANDB_MODE=offline`. Training still runs and still logs; the
curves land in `./wandb` instead of the cloud. The notebooks do this for you
automatically if the key is missing.

## 2. Clone and install (local)

```bash
git clone https://github.com/bhaskarjitsarmah/RL-Agents-Workshop-LLM.git
cd RL-Agents-Workshop-LLM
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # Windows: copy .env.example .env
pytest tests/ -q                                      # 133 tests, ~110s
```

If those tests pass, your environment is correct. They check the vendored files
byte-for-byte, the generated task set, the reward's separation property, the
MDP, and all nine notebooks.

**This local install cannot train.** It is for evaluation, data generation, the
reward and MDP experiments, analysis, and replaying every chart. Training is
Colab-only — `bitsandbytes` on Windows is unreliable, and a half-working local
install is worse than a clearly absent one.

## 3. Check your Colab GPU — do this the day before

Open any notebook's Colab badge, then:

**Runtime → Change runtime type → T4 GPU**, and run the first cell.

You should see something like:

```
GPU: Tesla T4  sm_75  14.7 GB  bf16=True  dtype=bfloat16
```

If it says `GPU: none`, you are on a CPU runtime — switch it. If Colab refuses
you a GPU (free-tier quota is not guaranteed), you can still follow the entire
day in **replay mode**: every notebook renders every chart from the pre-baked
runs that ship with the repo. You will read real curves; you just will not have
produced them yourself.

`bf16=True` on a T4 is expected, and is not a claim that Turing has bf16 tensor
cores — it does not. Current PyTorch emulates bf16 there, and we take that deal
deliberately: bf16 needs no `GradScaler`, and the scaler is what crashes with
`_amp_foreach_non_finite_check_and_unscale_ not implemented for 'BFloat16'` the
moment the model and the trainer disagree about dtype. The choice is made once,
in `llm_utils/config.py:torch_dtype()`, and every training config reads that same
flag — so never set `fp16=`/`bf16=` on a config by hand. See [COLAB.md](COLAB.md).

## 4. Optional keys

| key | needed for | without it |
|---|---|---|
| `OPENAI_API_KEY` | the `gpt-4o-mini` comparison rows in NB0/NB8 | repo 1's published `baseline_test.json` is used instead |
| `HF_TOKEN` | pushing **your own** adapters | the pre-baked adapters are public; you only need this to publish your own |
| `LANGFUSE_*` | tracing | silently skipped, no warning |

## 5. If you also did the harness workshop

NB8's hybrid row needs repo 1's evolved skill library. Run
[RL-Agents-Workshop](https://github.com/bhaskarjitsarmah/RL-Agents-Workshop)'s
NB6 once and export it:

```python
json.dump(theta, open("data/skills_evolved.json", "w"), indent=2)
```

Copy that file into this repo's `data/`. Without it, NB8 runs with four of six
rows and says so.

## Verify

```bash
python -c "from llm_utils import preflight; preflight()"
pytest tests/ -q
python scripts/run_notebook_cells.py notebooks/NB0_two_agents_one_scoreboard.ipynb
```

The last command runs NB0 end-to-end with no GPU and no keys. If it prints
`all 10 code cells ran clean`, you are ready.

## What you will *not* be asked to do

- No Qdrant, no vector database (repo 1 needed one; this one does not)
- No Langfuse account
- No OpenAI spend beyond a few cents, and only if you want the comparison rows
- No GPU purchase

See you at the workshop.
