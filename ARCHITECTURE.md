# Architecture: Self-Improving Agents by Optimizing the Weights

> **The one thesis:** an agent = a **harness** + a **policy**. Repo 1 froze the
> policy and evolved the harness. This repo freezes the harness and evolves the
> policy.
>
> **The reward is the loss · the LoRA adapter is the parameter vector · the
> trajectory group is the gradient.**

This document is the map. It connects the nine notebooks into one picture and
shows what each one moves.

> 🖼️ **Slide-ready exports** (SVG + PNG, 3x) of every diagram below live in
> [`docs/diagrams/`](docs/diagrams/). Re-export after editing with
> `python scripts/export_diagrams.py`.

---

## 1. The anatomy: what is frozen and what learns

```mermaid
flowchart TB
    User["NL question<br/>(Which product generated the most completed revenue?)"]

    subgraph FROZEN["HARNESS · frozen"]
        C["baseline_prompt<br/>schema + question"]
        E["execution loop<br/>generate → run → repair ×2"]
        T["run_sql"]
        P["extract_sql<br/>the parser"]
        V["score_sql<br/>execution match"]
    end

    subgraph LEARN["POLICY · learns"]
        W["Qwen2.5-Coder-1.5B + LoRA r=16<br/><b>θ = 18M params</b><br/><i>the only thing that moves</i>"]
    end

    User --> C --> W
    W -->|"candidate SQL"| P --> E
    E --> T --> E
    E --> V
    V -->|"reward"| G["GRPO<br/>A = (r − group mean)/std"]
    G -->|"gradient"| W

    style FROZEN fill:#eee,stroke:#888
    style LEARN fill:#fde8d8,stroke:#DD8452
```

Everything in the grey box is a **vendored copy of repo 1's code**, and
`tests/test_vendored_parity.py` fails the build if a single byte changes. That
is not tidiness — `score_sql` *is* the reward we optimise 18M parameters
against, and if it drifted, every number in both repos would silently stop being
comparable.

The one modification is a single `llm_fn` hook in `make_agent`, which is what
lets a local policy, an ART-served endpoint, and the OpenAI client all drive the
*same* agent.

---

## 2. The notebook journey

```mermaid
flowchart LR
    NB0["NB0 · Two agents<br/>one scoreboard<br/><i>the 35-point gap</i>"]
    NB1["NB1 · The MDP<br/>state/action/reward<br/><i>variance is the signal</i>"]
    NB2["NB2 · Data + STaR<br/>construct the gold<br/><i>the filter is the reward</i>"]
    NB3["NB3 · GRPO<br/>group mean = baseline<br/><i>no value network</i>"]
    NB4["NB4 · Multi-turn<br/>mask the tool output"]
    NB5["NB5 · OpenPipe ART<br/>the production wrapper"]
    NB6["NB6 · Reward hacking<br/><i>the scissors chart</i>"]
    NB7["NB7 · Deployment<br/>merge, serve, $/1k"]
    NB8["NB8 · Capstone<br/>weights vs harness<br/><i>+ the hybrid</i>"]

    NB0 --> NB1 --> NB2 --> NB3 --> NB4 --> NB5 --> NB6 --> NB7 --> NB8
```

| NB | Module | Moves | The new idea |
|---|---|---|---|
| **NB0** | M1 | nothing | the policy is differentiable now — and it starts 35 points behind |
| **NB1** | M1 | nothing | **reward variance within a sampled group** is the learning signal |
| **NB2** | M2, M4 | the data | construct the gold; then bootstrap from the model's own successes |
| **NB3** | M2 | **θ** | the group mean is the baseline — that is why it fits on a T4 |
| **NB4** | M4 | **θ** | credit assignment across turns; mask the environment's text |
| **NB5** | M3 | **θ** | ART owns the infrastructure you should not be writing |
| **NB6** | M5 | — | the reward is an attack surface and your optimizer is the attacker |
| **NB7** | M5 | — | accuracy is one axis; latency and $/1k are the other two |
| **NB8** | — | — | harness and weights are orthogonal; ship both |

---

## 3. The training loop

```mermaid
flowchart TB
    A["1 · SAMPLE a GROUP<br/>G completions for ONE prompt<br/>at T=0.9"]
    B["2 · REWARD (verifiable)<br/>execute pred vs gold<br/>+ format / executes / nonempty"]
    C{"3 · Is there SPREAD?<br/>std(r) > 0 ?"}
    D["4 · ADVANTAGE<br/>A_i = (r_i − mean)/(std+ε)<br/><b>no value network</b>"]
    E["5 · CLIPPED UPDATE + KL<br/>to the frozen reference"]
    F["6 · VALIDATION GATE<br/>held-out accuracy, NOT reward"]
    Z["ZERO GRADIENT<br/>the step is wasted<br/>→ raise T, raise G,<br/>re-filter the curriculum"]

    A --> B --> C
    C -- "yes" --> D --> E --> F
    C -- "no (flat group)" --> Z
    Z -.-> A
    F -- "val improved" --> A
    F -- "val fell" --> STOP["stop; promote the best checkpoint"]

    style C fill:#fff3cd,stroke:#d4a017
    style Z fill:#f8d7da,stroke:#C44E52
    style F fill:#f3f9f3,stroke:#5a5
```

**Step 3 is the one people skip.** GRPO's baseline is the group mean, so a group
whose members all score the same produces zero advantage for every member and
the entire step does nothing. Instrument `frac_zero_advantage` from step 0.

**Step 6 is inherited from repo 1.** Its validation gate stopped an unvetted
skill pool from degrading accuracy ~25%; here the same gate stops a reward-hacked
checkpoint from being promoted. Same idea, different parameter vector.

### The correspondence with repo 1

| Repo 1 (harness) | This repo (weights) |
|---|---|
| the skill document | the **LoRA adapter** (18M params, 36MB) |
| a reflection proposing an edit | the **advantage** of a sampled completion |
| accept/reject an edit | a **gradient step** |
| error on a held-out split | the **verifiable reward** |
| the validation gate | **early stopping on val accuracy** |
| EvoSkill mutate + select | **GRPO group sampling** |
| the 25%-degrade trap | **reward hacking** (NB6) |
| W&B curves | W&B curves |

Both are reinforcement learning. One does it in text space with no gradients;
the other does it in parameter space with them.

---

## 4. Data flow

```mermaid
flowchart LR
    subgraph GEN["Generator (49 template families)"]
        TPL["skeleton + slots<br/>drawn from the live DB"]
        VAL["validate: executes?<br/>SELECT? non-empty?"]
        LEAK["4-rule leakage audit<br/>vs the 16 test tasks"]
    end
    TPL --> VAL --> LEAK
    LEAK --> TRAIN["train 800"]
    LEAK --> VALSET["val 200"]
    LEAK --> EXT["test_ext 169"]
    LEAK --> NOLEAK["train_noleak 800<br/><i>memorization control</i>"]

    TRAIN --> STAR["STaR: sample k,<br/>keep the correct"] --> SFT["SFT adapter"]
    TRAIN --> GRPO["GRPO"] --> ADAPT["grpo adapter"]
    SFT --> GRPO
    VALSET -.->|"gate"| GRPO
    T16["the 16 held-out tasks<br/><b>never trained on</b>"] --> SCORE["vendored evaluate()"]
    ADAPT --> SCORE
    EXT --> SCORE
```

**Three eval sets, three jobs.** test-16 is the *comparability* number and the
only one commensurate with repo 1's 0.75 — its interval is ±20pp, so treat it
accordingly. val-200 gates. test_ext-169 is the *generalization* number with real
power (±7pp) and is where claims should be made.

---

## 5. The no-GPU contract

> **Every notebook renders every chart on a machine with no GPU, no API key, and
> no network.**

```mermaid
flowchart LR
    CELL["training cell"] --> Q{"CAP['gpu'] ?"}
    Q -- "yes" --> LIVE["train live"] --> HIST["history"]
    Q -- "no" --> REPLAY["load_result(key)"] --> HIST
    HIST --> CHART["the same plotting code"]
    REPLAY -. "absent?" .-> SAY["print the command<br/>that produces it<br/><b>never invent a curve</b>"]

    style SAY fill:#f8d7da,stroke:#C44E52
```

Replayed charts carry a *pre-baked* watermark. When an artifact has not been
baked yet, the cell prints how to produce it rather than drawing something
fabricated — a chart built from nothing is worse than no chart.

---

## 6. Module map

```
llm_utils/
  db.py tasks.py evaluate.py   VENDORED byte-identical -- the fairness contract
  agents.py                    VENDORED + one llm_fn hook (the hinge)
  llm.py                       VENDORED, Langfuse optional
  sqlio.py       safe_run_sql (closes on the error path) + cached gold scoring
  config.py      capability(), load_result() -- the no-GPU contract
  gen_tasks.py   49 families, slot domains, the 4-rule leakage audit
  rollout.py     SQLEnv, Trajectory, rollout_group, advantages
  rewards.py     the verifiable reward + the deliberately hackable proxy
  local_llm.py   LocalLM.as_llm_fn() -- local policy drives the vendored agent
  datasets.py    STaR sampling, SFT/GRPO dataset builders, curriculum filter
  trainers.py    T4-safe configs, kwarg filtering, merge, push
  art_bridge.py  our Trajectory <-> ART; art_probe() before trusting it
  metrics.py     Wilson CI, paired bootstrap, McNemar -- no bare accuracies
  plotting.py    house style; the 0.75 line on every accuracy chart
```

---

## TL;DR

1. **Agent = harness + policy.** We hold the harness byte-identical and move the
   policy. The comparison with repo 1 is enforced by a hash, not a promise.
2. **The reward must be verifiable**, and it must keep `min_correct >
   max_incorrect`. Ours has a margin of 0.70.
3. **GRPO's baseline is the group mean** — no critic, which is why it fits on a
   free T4. A flat group is a wasted step; measure that.
4. **Gate on validation accuracy, not reward.** When they diverge, the reward is
   lying to you.
5. **At n=16, report intervals and paired tests.** One task is 6.25 points.
6. **Harness and weights are orthogonal.** The hybrid is one line of code —
   which is the dividend of never having touched the harness.
