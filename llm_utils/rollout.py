"""The MDP: state, action, transition, reward -- made runnable.

AV Module 1 asks participants to see an LLM agent as a policy in an MDP rather
than a text predictor. This module is that slide, executable:

    state s_t        the message list: system prompt + schema + question +
                     every prior tool observation
    action a_t       a sampled token sequence from pi_theta -- either a tool
                     call or a final ```sql block
    transition       DETERMINISTIC: parse the action, run it against SQLite,
                     append the observation. No stochastic environment; all the
                     randomness lives in the policy.
    reward r         terminal `score_sql` plus shaping (see rewards.py)
    discount         gamma = 1, horizon <= 4. Short enough that trajectory-level
                     credit assignment is honest.
    policy pi_theta  Qwen2.5-Coder-1.5B + LoRA -- **theta is 18M adapter
                     parameters, not 1.5B**

The single most important object here is the *group*. `rollout_group` samples G
completions for one prompt; GRPO's baseline is that group's mean reward, so the
learning signal is the **spread** of rewards within the group. A group whose
members all score identically produces zero advantage for every member and
wastes the entire step -- which is why `advantages()` and
`metrics.zero_advantage_fraction` exist, and why NB1's temperature sweep is not
a curiosity but the setup for NB3.

Single-turn is `max_turns=1` with the action forced to be final, so NB1's
formalism and NB3's training run through literally the same code path.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .agents import baseline_prompt, extract_sql
from .db import DB_PATH, SCHEMA_TEXT
from .rewards import composite_reward
from .sqlio import exec_info, is_write_statement, safe_run_sql

#: A policy maps (messages, n, temperature, max_new_tokens) -> n completions.
#: Deliberately narrow: LocalLM, an OpenAI client, ART's served endpoint, and a
#: scripted stub in the tests all satisfy it.
Policy = Callable[..., list]


@dataclass
class Step:
    role: str                      # system | user | assistant | tool
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None


@dataclass
class Trajectory:
    """One episode, with everything needed to train on it or audit it."""

    task: dict
    steps: list[Step] = field(default_factory=list)
    final_sql: str = ""
    raw_completion: str = ""
    reward: float = 0.0
    reward_parts: dict = field(default_factory=dict)
    correct: bool = False
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False
    terminated_reason: str = ""

    def to_messages(self) -> list[dict]:
        return [{"role": s.role, "content": s.content} for s in self.steps]

    def assistant_spans(self) -> list[int]:
        """Indices of the assistant turns -- the ONLY tokens we train on.

        Tool observations are environment text, not policy output. Including
        them in the loss teaches the model to predict its own database, which is
        both useless and actively harmful: it dilutes the gradient with tokens
        the policy has no control over. Masking them is the whole trick of
        multi-turn RL (NB4).
        """
        return [i for i, s in enumerate(self.steps) if s.role == "assistant"]

    def render(self) -> str:
        out = []
        for s in self.steps:
            tag = s.role.upper()
            body = s.content if len(s.content) < 600 else s.content[:600] + " ..."
            out.append(f"--- {tag} ---\n{body}")
        out.append(f"--- REWARD {self.reward:.3f} "
                   f"(correct={self.correct}, turns={self.n_llm_calls}, "
                   f"{self.terminated_reason}) ---")
        return "\n".join(out)


# ===========================================================================
# The environment
# ===========================================================================

TOOL_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)


def _looks_like_sql(text: str) -> bool:
    """Does this parse as an attempted query, rather than prose?

    Used only by `SQLEnv` to decide whether the policy has submitted an answer.
    Scoring still goes through the vendored parser unchanged.
    """
    t = (text or "").strip().lstrip("(").lower()
    return t.startswith("select") or t.startswith("with")

MULTITURN_SYSTEM = (
    "You are a precise text-to-SQL assistant for a SQLite database. You may "
    "inspect the database before answering.\n\n"
    "To use a tool, reply with exactly:\n"
    '<tool>{"name": "<tool_name>", "args": {...}}</tool>\n\n'
    "Available tools:\n"
    "  list_tables    {}                        list table names\n"
    "  describe_table {\"table\": \"orders\"}       show a table's columns\n"
    "  sample_rows    {\"table\": \"orders\"}       show a few example rows\n"
    "  run_query      {\"sql\": \"SELECT ...\"}     run a read-only query\n\n"
    "When you know the answer, reply with ONLY the final query inside a ```sql "
    "code block -- no tool call, no explanation."
)


class SQLEnv:
    """A deterministic, resettable text-to-SQL environment.

    Gym-ish on purpose: `reset` returns an observation, `step` returns
    (obs, reward, done, info). Participants who have written an RL loop before
    recognise the shape immediately, and the ones who haven't get the shape for
    free.

    Determinism matters more here than it might seem. The transition function is
    a SQLite query against an immutable database, so replaying a trajectory
    reproduces it exactly -- which is what makes the pre-baked runs in
    `data/results/` legitimate replays rather than approximations.
    """

    TOOLS = ("list_tables", "describe_table", "sample_rows", "run_query")

    def __init__(self, db_path: str = DB_PATH, max_turns: int = 4,
                 schema_text: str = SCHEMA_TEXT, row_limit: int = 10):
        self.db_path = db_path
        self.max_turns = max_turns
        self.schema_text = schema_text
        self.row_limit = row_limit
        self.task: dict | None = None
        self.steps: list[Step] = []
        self.turn = 0
        self.done = False

    # -- lifecycle ---------------------------------------------------------
    def reset(self, task: dict) -> list[dict]:
        self.task = task
        self.turn = 0
        self.done = False
        self.steps = [
            Step("system", MULTITURN_SYSTEM),
            Step("user", f"Database schema:\n{self.schema_text}\n\n"
                         f"Question: {task['question']}"),
        ]
        return self.observation()

    def observation(self) -> list[dict]:
        return [{"role": s.role, "content": s.content} for s in self.steps]

    # -- transition --------------------------------------------------------
    def step(self, action_text: str) -> tuple[list[dict], float, bool, dict]:
        """Apply one action. Deterministic given the action."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; reset() first")
        self.turn += 1
        self.steps.append(Step("assistant", action_text))

        call = self._parse_tool(action_text)
        if call is not None:
            obs_text = self._run_tool(call)
            self.steps.append(Step("tool", obs_text, tool_name=call.get("name"),
                                   tool_args=call.get("args"), tool_result=obs_text))
            if self.turn >= self.max_turns:
                self.done = True
                return (self.observation(), 0.0, True,
                        {"reason": "max_turns", "sql": "", "tool_calls": 1})
            return self.observation(), 0.0, False, {"reason": "tool", "tool_calls": 1}

        sql = extract_sql(action_text)
        # `extract_sql` is vendored and deliberately permissive: given prose with
        # no fence it returns the prose. That is right for the single-turn
        # harness (the prose simply scores 0), but here it would turn "I don't
        # know" into a submitted answer and rob the policy of its one correction.
        # So the ENV -- not the parser -- decides what counts as a submission.
        if not _looks_like_sql(sql):
            # A malformed action gets ONE corrective observation rather than an
            # immediate zero. Terminating here would make a parser mismatch look
            # exactly like reward collapse in the training curves -- the most
            # expensive misdiagnosis available in this workshop.
            if self.turn < self.max_turns:
                self.steps.append(Step(
                    "tool", "Your reply contained neither a <tool>...</tool> call "
                            "nor a ```sql block. Reply with one of them."))
                return self.observation(), 0.0, False, {"reason": "malformed",
                                                        "tool_calls": 0}
            self.done = True
            return (self.observation(), 0.0, True,
                    {"reason": "malformed_final", "sql": "", "tool_calls": 0})

        self.done = True
        return (self.observation(), 0.0, True,
                {"reason": "submitted", "sql": sql, "tool_calls": 0})

    # -- tools -------------------------------------------------------------
    def _parse_tool(self, text: str) -> dict | None:
        m = TOOL_RE.search(text or "")
        if not m:
            return None
        try:
            call = json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            return {"name": "__bad_json__", "args": {}}
        return call if isinstance(call, dict) else {"name": "__bad_json__", "args": {}}

    def _run_tool(self, call: dict) -> str:
        name = call.get("name")
        args = call.get("args") or {}
        if name == "__bad_json__":
            return "Tool call was not valid JSON. Expected: " \
                   '<tool>{"name": "...", "args": {...}}</tool>'
        if name == "list_tables":
            return "Tables: customers, products, orders, order_items"
        if name == "describe_table":
            table = str(args.get("table", ""))
            if table not in ("customers", "products", "orders", "order_items"):
                return f"No such table: {table!r}. Tables: customers, products, orders, order_items"
            rows, err = safe_run_sql(f"PRAGMA table_info({table})", self.db_path)
            if err or not rows:
                # PRAGMA is blocked by our read-only guard; fall back to the
                # schema text the model already has, rather than leaking an error.
                return f"Columns of {table}: see the schema in the first message."
            return f"{table}(" + ", ".join(r[1] for r in rows) + ")"
        if name == "sample_rows":
            table = str(args.get("table", ""))
            if table not in ("customers", "products", "orders", "order_items"):
                return f"No such table: {table!r}"
            rows, err = safe_run_sql(f"SELECT * FROM {table} LIMIT 3", self.db_path)
            return f"{table} sample: {rows}" if not err else f"Error: {err}"
        if name == "run_query":
            sql = str(args.get("sql", ""))
            if is_write_statement(sql):
                return "Refused: only read-only SELECT queries are allowed."
            rows, err = safe_run_sql(sql, self.db_path)
            if err:
                return f"Error: {err}"
            shown = rows[: self.row_limit]
            more = f" (+{len(rows) - len(shown)} more rows)" if len(rows) > len(shown) else ""
            return f"{len(rows)} row(s): {shown}{more}"
        return f"Unknown tool {name!r}. Available: {', '.join(self.TOOLS)}"


# ===========================================================================
# Rollouts
# ===========================================================================

def rollout_single_turn(policy: Policy, task: dict, temperature: float = 0.7,
                        max_new_tokens: int = 256, weights: dict | None = None,
                        extra: str = "") -> Trajectory:
    """One prompt in, one completion out, reward computed. The GRPO unit.

    Uses the VENDORED `baseline_prompt`, so the policy sees exactly the string
    the eval harness will send it. Training and evaluation must not diverge on
    prompt formatting -- that mismatch is the single most common silent failure
    in this kind of work, and it shows up as "the fine-tuned model got worse".
    """
    t0 = time.time()
    messages = baseline_prompt(task["question"], extra=extra)
    outs = policy(messages, n=1, temperature=temperature,
                  max_new_tokens=max_new_tokens)
    text = outs[0] if outs else ""
    sql = extract_sql(text)
    reward, parts = composite_reward(text, task["gold"], n_llm_calls=1,
                                     weights=weights)
    return Trajectory(
        task=task,
        steps=[Step(m["role"], m["content"]) for m in messages]
              + [Step("assistant", text)],
        final_sql=sql, raw_completion=text,
        reward=reward, reward_parts=parts,
        correct=bool(parts.get("correct")),
        n_llm_calls=1, n_tool_calls=0,
        latency_s=time.time() - t0,
        terminated_reason="single_turn",
    )


def rollout_multi_turn(policy: Policy, task: dict, env: SQLEnv | None = None,
                       max_turns: int = 4, temperature: float = 0.7,
                       max_new_tokens: int = 256,
                       weights: dict | None = None) -> Trajectory:
    """Let the policy inspect the database before committing to an answer.

    The efficiency penalty in the reward is what stops this degenerating into
    "always call four tools": looking is useful, but it costs. NB4's ablation
    turns that penalty up to 0.30 and shows the policy abandoning tools entirely
    -- a benign, live demonstration of reward mis-specification.
    """
    env = env or SQLEnv(max_turns=max_turns)
    t0 = time.time()
    env.reset(task)
    sql, reason, tool_calls = "", "max_turns", 0

    for _ in range(max_turns):
        outs = policy(env.observation(), n=1, temperature=temperature,
                      max_new_tokens=max_new_tokens)
        action = outs[0] if outs else ""
        _, _, done, info = env.step(action)
        tool_calls += info.get("tool_calls", 0)
        if done:
            sql, reason = info.get("sql", ""), info.get("reason", "")
            break

    last = next((s.content for s in reversed(env.steps) if s.role == "assistant"), "")
    reward, parts = composite_reward(last, task["gold"],
                                     n_llm_calls=env.turn, weights=weights)
    if not sql:
        # Terminated without submitting: no credit, but keep the breakdown so
        # the failure is visible in the dashboard rather than just "0".
        parts["exec_match"] = 0.0
    return Trajectory(
        task=task, steps=list(env.steps), final_sql=sql, raw_completion=last,
        reward=reward, reward_parts=parts, correct=bool(parts.get("correct")),
        n_llm_calls=env.turn, n_tool_calls=tool_calls,
        latency_s=time.time() - t0, terminated_reason=reason,
    )


def rollout_group(policy: Policy, task: dict, G: int = 8,
                  temperature: float = 0.9, multi_turn: bool = False,
                  **kw) -> list[Trajectory]:
    """Sample G completions for ONE prompt. This is the GRPO group.

    Temperature defaults higher than eval (0.9 vs 0.0) on purpose: GRPO learns
    from the *spread* of rewards inside the group, and a greedy group has no
    spread at all. NB1 demonstrates that with a temperature sweep before NB3
    depends on it.
    """
    fn = rollout_multi_turn if multi_turn else rollout_single_turn
    return [fn(policy, task, temperature=temperature, **kw) for _ in range(G)]


def batch_rollout(policy: Policy, tasks: list[dict], G: int = 1,
                  temperature: float = 0.9, multi_turn: bool = False,
                  **kw) -> list[list[Trajectory]]:
    """One group per task. Returns a list of groups, not a flat list."""
    return [rollout_group(policy, t, G=G, temperature=temperature,
                          multi_turn=multi_turn, **kw) for t in tasks]


# ===========================================================================
# Advantages -- the heart of GRPO
# ===========================================================================

def advantages(trajs: list[Trajectory], scale: bool = True,
               eps: float = 1e-4) -> list[float]:
    """A_i = (r_i - mean(r)) / (std(r) + eps), computed WITHIN the group.

    This is the entirety of GRPO's departure from PPO: there is no value
    network, no critic to train, no second model in memory. The group mean *is*
    the baseline. That is why GRPO fits on a 16GB T4 and PPO does not.

    `scale=False` gives Dr.GRPO's un-normalised advantage. Dividing by the
    group std systematically up-weights groups the policy finds ambiguous
    (small std) and down-weights decisive ones, which biases learning toward
    borderline prompts. NB3 ablates it.

    If every member scores the same, every advantage is 0.0 and the step does
    nothing. Do not paper over that with noise -- measure it
    (`metrics.zero_advantage_fraction`) and fix it upstream with temperature,
    a bigger G, or a curriculum filter.
    """
    rs = [t.reward for t in trajs]
    n = len(rs)
    if n == 0:
        return []
    mean = sum(rs) / n
    if not scale:
        return [r - mean for r in rs]
    var = sum((r - mean) ** 2 for r in rs) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n          # a flat group: honestly zero, not fudged
    return [(r - mean) / (std + eps) for r in rs]


def summarize_group(trajs: list[Trajectory]) -> dict:
    """The per-group diagnostics NB1 and NB3 plot."""
    if not trajs:
        return {}
    rs = [t.reward for t in trajs]
    n = len(rs)
    mean = sum(rs) / n
    var = sum((r - mean) ** 2 for r in rs) / n
    n_correct = sum(1 for t in trajs if t.correct)
    return {
        "task_id": trajs[0].task.get("id"),
        "level": trajs[0].task.get("level"),
        "G": n,
        "reward_mean": mean,
        "reward_std": var ** 0.5,
        "reward_min": min(rs),
        "reward_max": max(rs),
        "pass_rate": n_correct / n,
        "pass_at_1": float(trajs[0].correct),
        "pass_at_g": float(n_correct > 0),
        # A group is only useful if its members DISAGREE.
        "zero_advantage": var ** 0.5 < 1e-4,
        "mean_turns": sum(t.n_llm_calls for t in trajs) / n,
        "mean_tool_calls": sum(t.n_tool_calls for t in trajs) / n,
    }


def learnable_band(groups: list[list[Trajectory]], lo: float = 0.125,
                   hi: float = 0.875) -> list[dict]:
    """Keep only tasks whose pass rate sits strictly between lo and hi.

    A task the policy solves 8/8 gives zero advantage. So does one it fails 8/8.
    Both consume a full generation pass and contribute nothing to the gradient.
    Re-filtering the training set to the learnable band after the first ~50 steps
    is the cheapest real speed-up available in this workshop -- and it is a
    curriculum, which is exactly the "task design" idea from AV Module 4.
    """
    keep = []
    for g in groups:
        if not g:
            continue
        rate = sum(1 for t in g if t.correct) / len(g)
        if lo <= rate <= hi:
            keep.append(g[0].task)
    return keep
