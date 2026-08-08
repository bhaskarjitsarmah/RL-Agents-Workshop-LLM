"""Multi-turn GRPO -- the trainer NB4 describes.

Everything else in this repo trains on a *dataset*: TRL's `SFTTrainer` and
`GRPOTrainer` take prompt/completion rows, generate one completion per prompt,
and score it. That cannot express a tool loop, where the policy speaks several
times, the environment speaks between those turns, and the reward arrives once
at the end for the whole episode.

So this module closes the loop NB4 explains:

    rollout   -> `batch_rollout(..., multi_turn=True)` gives G trajectories per
                 task, each a real episode against `SQLEnv`
    credit    -> `advantages()` scores each trajectory against its own group
                 mean (the GRPO baseline -- no critic, which is why it fits on
                 a T4)
    mask      -> `Trajectory.assistant_spans()` decides which tokens carry the
                 gradient. Tool observations are environment text the policy did
                 not write; training on them is a gradient through your own
                 database. THIS is the part that makes multi-turn different, and
                 the part that is silently wrong in most implementations.
    update    -> one REINFORCE-with-baseline step over the masked assistant
                 tokens only.

The loss, in full:

    L = - (1/|A|) * sum_{t in A} A_i * log pi(x_t | x_<t)

where `A` is the set of assistant token positions in trajectory i and `A_i` is
its group advantage. No ratio, no clipping, no reference model: with one
optimizer step per batch of fresh rollouts the policy that generated them IS
the current policy, so the PPO ratio is identically 1 and the KL anchor has
nothing to anchor against. Take more than one step per rollout batch and that
stops being true -- which is exactly what `epsilon`/`beta` exist for in NB3's
single-turn GRPO, and why this trainer does not reuse a batch.

Deliberately small: ~15 steps is a demonstration you can watch, not a run that
converges. The point is that turns-per-episode falls while accuracy holds.
"""

from __future__ import annotations

from .rollout import advantages, batch_rollout


def policy_from_model(model, tok, default_max_new_tokens: int = 192):
    """A `Policy` callable that generates with the model being trained.

    `LocalLM` deliberately owns a *separate* model for evaluation. Here we need
    the opposite: rollouts must come from the weights we are about to update,
    or the advantages describe a policy that no longer exists.
    """
    import torch

    @torch.no_grad()
    def policy(messages, n: int = 1, temperature: float = 0.7,
               max_new_tokens: int | None = None, **_kw) -> list[str]:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        was_cache, was_training = model.config.use_cache, model.training
        model.config.use_cache = True     # gradient checkpointing turns this off
        model.eval()
        try:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens or default_max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        finally:
            model.config.use_cache = was_cache
            model.train(was_training)
        cut = enc["input_ids"].shape[1]
        return [tok.decode(o[cut:], skip_special_tokens=True) for o in out]

    return policy


def assistant_mask(tok, messages: list[dict], max_len: int = 1024):
    """Token ids for the whole episode, plus 1.0 on assistant tokens only.

    Built by re-templating cumulative prefixes rather than by string search:
    chat templates insert their own role headers and special tokens, so any
    attempt to locate assistant text by matching the decoded string drifts by a
    few tokens -- and a mask that is off by a few tokens trains on the role
    header instead of the reply, quietly.
    """
    ids: list[int] = []
    mask: list[float] = []
    prev = 0
    for i, m in enumerate(messages):
        cur = tok.apply_chat_template(messages[:i + 1], tokenize=True,
                                      add_generation_prompt=False)
        if len(cur) <= prev:              # template produced nothing new
            continue
        chunk = cur[prev:]
        ids.extend(chunk)
        mask.extend([1.0 if m["role"] == "assistant" else 0.0] * len(chunk))
        prev = len(cur)
    return ids[:max_len], mask[:max_len]


def _trajectory_loss(model, tok, traj, adv: float, max_len: int):
    """-A * mean log pi over the assistant tokens. None when nothing is maskable."""
    import torch

    ids, mask = assistant_mask(tok, traj.to_messages(), max_len=max_len)
    if sum(mask) == 0 or len(ids) < 2:
        return None
    x = torch.tensor([ids], device=model.device)
    m = torch.tensor([mask], device=model.device)[:, 1:]   # predict t from t-1
    logits = model(x).logits[:, :-1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -(adv * (tok_logp * m).sum() / m.sum().clamp(min=1.0))


def evaluate_turns(policy, tasks: list[dict], max_turns: int = 4,
                   hide_schema: bool = False) -> dict:
    """Greedy pass for the curve: accuracy and turns on held-out tasks."""
    from .rollout import rollout_multi_turn

    if not tasks:
        return {"accuracy": 0.0, "mean_turns": 0.0}
    trajs = [rollout_multi_turn(policy, t, max_turns=max_turns, temperature=0.0,
                                hide_schema=hide_schema) for t in tasks]
    n = len(trajs)
    return {
        "accuracy": sum(1 for t in trajs if t.correct) / n,
        "mean_turns": sum(t.n_llm_calls for t in trajs) / n,
    }


def train_multi_turn(model, tok, tasks: list[dict], val_tasks: list[dict] | None = None,
                     steps: int = 15, G: int = 4, tasks_per_step: int = 2,
                     max_turns: int = 4, temperature: float = 0.9,
                     lr: float = 1e-5, max_len: int = 1024,
                     eval_every: int = 5, eval_n: int = 16,
                     weights: dict | None = None, hide_schema: bool = False,
                     verbose: bool = True) -> list[dict]:
    """Run multi-turn GRPO and return a log_history-shaped list.

    Each entry: {"step", "reward", "mean_turns", "val_accuracy"} -- the keys
    NB4's three-panel chart plots. `val_accuracy` is only measured every
    `eval_every` steps (it costs a greedy pass over `eval_n` held-out tasks);
    intermediate steps carry the last measured value so the line is continuous
    rather than full of holes.
    """
    import torch

    val_tasks = val_tasks or []
    try:                                  # unsloth patches training/inference modes
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model)
    except Exception:                     # noqa: BLE001 - plain peft model
        pass
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "no trainable parameters -- load the policy with load_4bit_policy() "
            "so a LoRA adapter is attached before training it.")
    opt = torch.optim.AdamW(trainable, lr=lr)
    policy = policy_from_model(model, tok)

    history: list[dict] = []
    last_acc = None
    for step in range(1, steps + 1):
        batch = [tasks[(step * tasks_per_step + i) % len(tasks)]
                 for i in range(tasks_per_step)]
        groups = batch_rollout(policy, batch, G=G, temperature=temperature,
                               multi_turn=True, max_turns=max_turns,
                               weights=weights, hide_schema=hide_schema)

        opt.zero_grad(set_to_none=True)
        n_terms, total = 0, 0.0
        for group in groups:
            advs = advantages(group, scale=False)      # Dr.GRPO, as in NB3
            for traj, adv in zip(group, advs):
                if adv == 0.0:                         # a flat group teaches nothing
                    continue
                loss = _trajectory_loss(model, tok, traj, adv, max_len)
                if loss is None:
                    continue
                loss.backward()
                total += float(loss.detach())
                n_terms += 1
        if n_terms:
            torch.nn.utils.clip_grad_norm_(trainable, 0.3)
            opt.step()

        flat = [t for g in groups for t in g]
        rec = {
            "step": step,
            "reward": sum(t.reward for t in flat) / max(len(flat), 1),
            "mean_turns": sum(t.n_llm_calls for t in flat) / max(len(flat), 1),
            "loss": total / n_terms if n_terms else 0.0,
            "zero_advantage": 1.0 - (n_terms / max(len(flat), 1)),
        }
        if val_tasks and (step % eval_every == 0 or step == steps):
            last_acc = evaluate_turns(policy, val_tasks[:eval_n],
                                      max_turns=max_turns,
                                      hide_schema=hide_schema)["accuracy"]
        rec["val_accuracy"] = last_acc
        history.append(rec)
        if verbose:
            acc = "-" if last_acc is None else f"{last_acc:.3f}"
            print(f"  step {step:>3}  reward {rec['reward']:+.3f}  "
                  f"turns {rec['mean_turns']:.2f}  val_acc {acc}  "
                  f"zero-adv {rec['zero_advantage']:.0%}")

    # Backfill the leading Nones so the chart starts at the first measurement
    # instead of breaking the line -- an absent measurement, not a zero.
    first = next((h["val_accuracy"] for h in history
                  if h["val_accuracy"] is not None), None)
    for h in history:
        if h["val_accuracy"] is None:
            h["val_accuracy"] = first
    return history
