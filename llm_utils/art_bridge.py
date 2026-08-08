"""OpenPipe ART integration -- the production wrapper for the same experiment.

NB3/NB4 hand-roll rollout collection, reward assignment, advantage computation,
loss masking, checkpointing, and serving. That is infrastructure, not research.
ART owns all of it: GRPO + LoRA + vLLM + W&B behind a client/server split, and
it is multi-turn-native rather than bolted on.

The punchline of NB5 is one line, and it is why `agents.py` got its `llm_fn`
hook: ART serves an **OpenAI-compatible** endpoint, so

    art_openai_agent(model)   ->   make_agent(llm_fn=...)   ->   evaluate()

scores the ART-trained policy with the *same vendored evaluate()* that produced
repo 1's 0.75 and NB3's TRL numbers. Three different backends, one scorer.

A warning about this module's API assumptions
---------------------------------------------
ART moves faster than TRL. Names used below -- `art.TrainableModel`,
`art.LocalBackend`, `model.register`, `model.openai_client`, `art.Trajectory`,
`art.TrajectoryGroup`, `art.gather_trajectory_groups`, `model.train`,
`art.TrainConfig` -- are what this module *expects*, not what it has verified.
`art_probe()` prints the real namespace, and NB5's first cell runs it before
anything else. **Treat that output as ground truth and correct this file.**

ART is async, so the training entry points are coroutines; notebook cells use a
bare `await`, which works under ipykernel.

If `LocalBackend` will not initialise on a free T4 -- a real risk, because it
wants vLLM and vLLM on Turing is fragile -- NB5 replays a pre-baked run and
ships the exact code to run elsewhere. `art_available()` is that gate.
"""

from __future__ import annotations

import os
from typing import Callable

from .agents import make_agent
from .config import base_model
from .rewards import composite_reward
from .rollout import Trajectory


def _local_backend_cls():
    """`LocalBackend`, wherever this ART version keeps it.

    It was a top-level re-export; in current ART it lives in `art.local` (and
    `art.local.backend`). The probe used to check `hasattr(art, "LocalBackend")`
    alone, so a perfectly working install reported the name missing, NB5 set
    ART_OK=False, and every live cell silently fell back to a replay that does
    not exist. Returns None when genuinely absent.
    """
    import importlib

    for mod, attr in (("art", "LocalBackend"),
                      ("art.local", "LocalBackend"),
                      ("art.local.backend", "LocalBackend")):
        try:
            return getattr(importlib.import_module(mod), attr)
        except Exception:  # noqa: BLE001 - try the next location
            continue
    return None


def art_available() -> bool:
    try:
        import art  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def art_probe() -> dict:
    """Print ART's ACTUAL public surface. Run this before trusting anything here."""
    try:
        import art
    except Exception as e:  # noqa: BLE001
        print(f"openpipe-art not importable: {e}")
        return {"available": False, "error": str(e)}

    names = sorted(n for n in dir(art) if not n.startswith("_"))
    info = {"available": True,
            "version": getattr(art, "__version__", "unknown"),
            "names": names}
    print(f"art.__version__ = {info['version']}")
    print("public names:", names)
    expected = ["TrainableModel", "Trajectory", "TrajectoryGroup", "TrainConfig",
                "gather_trajectory_groups"]
    missing = [n for n in expected if not hasattr(art, n)]
    if _local_backend_cls() is None:      # checked by import, not by hasattr
        missing.append("LocalBackend")
    info["missing_expected"] = missing
    if missing:
        print("\n!! not on the top-level namespace:", missing)
        print("   Find their real import paths before running NB5's live cells;")
        print("   llm_utils/art_bridge.py assumes the names above.")
    return info


def traj_to_art(traj: Trajectory, extra_metrics: dict | None = None):
    """Our Trajectory -> art.Trajectory.

    ART wants the message list plus a scalar reward, and accepts arbitrary
    metrics that surface in W&B. We pass the reward COMPONENTS through, so an
    ART run is diagnosable with the same per-component breakdown as the TRL run
    -- otherwise the two curves in NB5 would not be comparable in any detail.
    """
    import art

    metrics = {k: float(v) for k, v in (traj.reward_parts or {}).items()
               if isinstance(v, (int, float))}
    metrics.update({
        "correct": float(traj.correct),
        "n_llm_calls": float(traj.n_llm_calls),
        "n_tool_calls": float(traj.n_tool_calls),
    })
    metrics.update(extra_metrics or {})
    return art.Trajectory(
        messages_and_choices=traj.to_messages(),
        reward=float(traj.reward),
        metrics=metrics,
    )


def group_to_art(trajs: list[Trajectory]):
    import art

    return art.TrajectoryGroup([traj_to_art(t) for t in trajs])


def art_openai_agent(model, extra: str = "", max_repairs: int = 2) -> Callable:
    """The vendored agent, driven by an ART-served policy.

    `model.openai_client()` is OpenAI-compatible, so this is the same one-line
    trick as `make_local_agent`. It is what lets NB5 close the loop by scoring
    an ART policy with the unmodified `evaluate()`.
    """
    client = model.openai_client()
    model_name = getattr(model, "name", None) or base_model()

    def _llm(messages, model=None, temperature: float = 0.0,
             max_tokens: int = 800, **kw) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = client.chat.completions.create(
            model=model_name, messages=messages, temperature=temperature,
            max_tokens=min(max_tokens, 512))
        return resp.choices[0].message.content or ""

    return make_agent(extra=extra, max_repairs=max_repairs, llm_fn=_llm)


async def art_rollout(model, task: dict, temperature: float = 0.9,
                      max_new_tokens: int = 192, weights: dict | None = None):
    """One ART rollout, scored by OUR reward.

    Deliberately reuses `composite_reward` and the vendored `baseline_prompt`
    rather than anything ART-specific: NB5 compares TRL-GRPO against ART-GRPO,
    and that comparison is only meaningful if the reward and the prompt are
    identical on both sides. The framework is the variable; nothing else is.
    """
    import art

    from .agents import baseline_prompt, extract_sql

    client = model.openai_client()
    messages = baseline_prompt(task["question"])
    resp = await client.chat.completions.create(
        model=model.name, messages=messages,
        temperature=temperature, max_tokens=max_new_tokens)
    text = resp.choices[0].message.content or ""
    reward, parts = composite_reward(text, task["gold"], n_llm_calls=1,
                                     weights=weights)
    return art.Trajectory(
        messages_and_choices=messages + [{"role": "assistant", "content": text}],
        reward=float(reward),
        metrics={"correct": float(parts.get("correct", 0.0)),
                 "exec_match": float(parts.get("exec_match", 0.0)),
                 "format": float(parts.get("format", 0.0)),
                 "sql_len": float(len(extract_sql(text)))},
    )


async def run_art_training(tasks: list[dict], project: str = "rl-weights-workshop",
                           model_name: str = "qwen-sql",
                           base_model_id: str | None = None,
                           steps: int = 20, groups_per_step: int = 8,
                           rollouts_per_group: int = 8,
                           learning_rate: float = 1e-5,
                           temperature: float = 0.9,
                           log_every: int = 1) -> dict:
    """The ART training loop. Returns a history list shaped like the TRL runs.

    Same shape on purpose: `plotting.learning_curve` and `grpo_dashboard` then
    plot an ART run and a TRL run with the same code, which is the comparison
    NB5 exists to make.
    """
    import random

    import art

    cls = _local_backend_cls()
    if cls is None:
        raise RuntimeError("ART is installed but LocalBackend is not importable "
                           "from art, art.local or art.local.backend.")
    try:
        backend = cls()
    except ModuleNotFoundError as e:
        # ART's local backend loads per-architecture handlers that import
        # megatron.core. Megatron targets multi-GPU A100-class training and is
        # not practically installable on a free T4, so this is a real capability
        # limit rather than a misconfiguration -- say so plainly instead of
        # surfacing an import error from a file nobody has heard of.
        raise RuntimeError(
            f"ART's LocalBackend needs {e.name!r}, which this runtime does not "
            f"have. Local ART training expects a megatron/vLLM stack that a free "
            f"T4 cannot provide. NB5 replays a pre-baked run instead; the code "
            f"above is what you would run on suitable hardware.") from e
    model = art.TrainableModel(
        name=model_name, project=project,
        base_model=base_model_id or base_model(),
    )
    await model.register(backend)

    rng = random.Random(0)
    history: list[dict] = []
    for step in range(steps):
        batch = rng.sample(tasks, min(groups_per_step, len(tasks)))
        groups = await art.gather_trajectory_groups(
            (art.TrajectoryGroup(
                art_rollout(model, t, temperature=temperature)
                for _ in range(rollouts_per_group))
             for t in batch)
        )
        await model.train(groups,
                          config=art.TrainConfig(learning_rate=learning_rate))

        rewards = [tr.reward for g in groups for tr in g]
        correct = [tr.metrics.get("correct", 0.0) for g in groups for tr in g]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        # A group whose members all score the same yields zero advantage; the
        # TRL dashboard logs this too, so the panels line up.
        flat = sum(1 for g in groups
                   if len({round(tr.reward, 6) for tr in g}) <= 1)
        row = {"step": step, "reward": mean,
               "accuracy": sum(correct) / len(correct) if correct else 0.0,
               "frac_zero_advantage": flat / len(groups) if groups else 0.0,
               "n_rollouts": len(rewards)}
        history.append(row)
        if step % log_every == 0:
            print(f"  step {step:>3}  reward {row['reward']:.3f}  "
                  f"acc {row['accuracy']:.3f}  "
                  f"zero-adv groups {row['frac_zero_advantage']:.2f}")

    return {"history": history, "model": model, "backend": backend,
            "steps": steps}


def art_notes() -> str:
    """What ART owns that NB3/NB4 hand-rolled. Rendered as NB5's closing table."""
    return """
| Concern                          | NB3/NB4 (by hand) | ART |
|---|---|---|
| rollout collection & batching    | ours              | owned |
| reward -> trajectory plumbing    | ours              | owned |
| advantage computation            | ours              | owned |
| multi-turn loss masking          | ours (fiddly)     | native |
| inference server + LoRA hot-swap | none              | vLLM, owned |
| checkpointing / resume           | ours              | owned |
| W&B logging                      | ours              | native |
| reward when NOT verifiable       | n/a               | RULER |
""".strip()
