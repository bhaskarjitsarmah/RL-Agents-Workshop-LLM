# Vendored code: the fairness contract

This repo claims that its head-to-head against
[`RL-Agents-Workshop`](https://github.com/bhaskarjitsarmah/RL-Agents-Workshop) is
honest — same environment, same eval set, same scorer, same prompts, **only the
weights moved**.

A claim like that is worth nothing unless it is checked mechanically. This file
records what was copied and from where; `tests/test_vendored_parity.py` enforces
it on every run.

**Source:** `RL-Agents-Workshop` @ commit `4185c4a2a3389a402fb1b9cc98a03678470c41b3`

## Verbatim — byte-identical, hash-asserted

| File | sha256 |
|---|---|
| `llm_utils/db.py` | `b42332538e5feb7abd323cfc7ad026d9f349d66d88c63ecf0d9182538ac5198f` |
| `llm_utils/tasks.py` | `9c2a681e7b144a2b5b2eb323dce5b5e78dac4437e649d836fd9951d1688d3723` |
| `llm_utils/evaluate.py` | `f1a57bb19e911a3a03c58679398f265605eb6949b5fa79dec7b298a956416aeb` |
| `data/baseline_test.json` | `4e3dbe9318871d9a070e587093a99236dacc3c993e117559a6b17ae07be319a7` |

Why each one must not move:

- **`db.py`** — `score_sql` *is* the reward function, and `build_db(seed=42)` *is*
  the environment. A one-line "improvement" here silently rebases every number in
  both repos. The test hashes the **row content** of all four tables rather than
  the `.db` file bytes, because SQLite page layout varies by version.
- **`tasks.py`** — the 16 held-out tasks (ids 3, 5, 8, 9, 12, 14, 16, 18, 22, 24,
  27, 30, 33, 36, 38, 40) are the contract. Same questions *and* same gold SQL.
- **`evaluate.py`** — the `agent_fn(question) -> sql` contract, the
  crash-scores-zero semantics, and the `by_level` aggregation. This one function
  measures gpt-4o-mini, the base Qwen, the fine-tuned Qwen, and the ART-served
  policy identically. No fast paths, no goalpost-moving.
- **`baseline_test.json`** — repo 1's published result: `accuracy 0.75`,
  `easy 1.0 / medium 0.5 / hard 0.75`. The number to beat.

Do **not** add a `# vendored from ...` header to these files — it changes the
hash. Provenance lives here instead.

## Modified — minimal and documented

### `llm_utils/agents.py` — exactly one additive change

`make_agent` and `make_baseline_agent` gained an `llm_fn` parameter:

```python
def make_agent(model=None, extra="", max_repairs=2, llm_fn=None):
    _llm = llm_fn or llm            # the only new line of logic
```

This is the hinge the entire repo turns on. It lets the *same* agent — same
prompt, same repair loop, same parser — be driven by a local Qwen policy, an
ART-served endpoint, or the OpenAI client, and be scored by the *same*
`evaluate()`. It is also why NB8's hybrid row is one line of code.

`extract_sql`, `BASELINE_SYSTEM`, `baseline_prompt`, `REPAIR_SYSTEM`, and
`repair_prompt` are **byte-identical**. `test_prompt_surface_unchanged` pins a
hash over the rendered prompts, verified equal to the same digest computed
against repo 1's original module.

A direct consequence for reward design: **the format reward must reward a
` ```sql ` fenced block**, because the unmodified `extract_sql` is what parses
the policy's output at eval time. Train to the parser you will be scored by.

### `llm_utils/llm.py` — two changes

1. **Langfuse is optional.** Repo 1 hard-required the `langfuse.openai` drop-in.
   Here it degrades to plain `openai` with a no-op `observe`. A Colab install is
   already 4–6 minutes, only two notebooks trace anything, and a participant with
   no Langfuse account must still be able to `import llm_utils`.
2. **`preflight()` moved to `config.py`** (it now checks the GPU and a different
   key set) and is re-exported here, so repo 1's import lines still work.

Also added: `reset_client()`, because NB5 and NB7 repoint `OPENAI_BASE_URL` at a
served endpoint mid-notebook and then re-run the vendored `evaluate()`.

## Known quirk inherited from repo 1

`db.run_sql` closes its connection on success but **not** on the error path:

```python
except Exception as e:
    return None, f"{type(e).__name__}: {e}"     # con is never closed
```

Invisible at repo 1's scale. Here the policy emits invalid SQL constantly during
GRPO, and anything that keeps a traceback alive keeps the handle open — on
Windows that makes the next `build_db()` fail with `PermissionError: [WinError 32]`.

We cannot fix it without breaking the hash, so instead:

- `tests/conftest.py` builds the database **once per session**;
- `llm_utils/sqlio.py` provides `safe_run_sql` (closes in a `finally`, adds a
  query timeout) and `fast_score_sql` (caches the gold side — the DB is immutable
  during a run, so re-executing gold on every rollout is pure waste).

`fast_score_sql` is verified against the vendored `score_sql` on the full 40×40
cross-product plus a corpus of degenerate queries. **The vendored `score_sql`
remains the authority for every headline number**; `sqlio` is strictly a
training-loop accelerator that is proven to agree with it.

## Re-vendoring

If repo 1 changes and you want to pull the update:

1. Re-copy the four verbatim artifacts.
2. Re-apply the `llm_fn` hook to `agents.py` and the two `llm.py` changes.
3. Recompute all hashes and update this file **and**
   `tests/test_vendored_parity.py`.
4. **Re-baseline repo 1** and update `data/baseline_test.json`.
5. Re-run every pre-baked training run whose reward depends on `score_sql`.

Step 5 is the expensive one. That is the intended incentive: the environment is
supposed to be frozen.
