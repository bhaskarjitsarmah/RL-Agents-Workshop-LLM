"""A backend-agnostic LLM wrapper, a cost meter, and optional Langfuse tracing.

VENDORED from RL-Agents-Workshop/workshop_utils/llm.py with two changes:

1. **Langfuse is optional, not required.** The original hard-required the
   `langfuse.openai` drop-in. Here it degrades to plain `openai` with a no-op
   `observe` decorator. Rationale: a Colab install is already 4-6 minutes, only
   two notebooks need tracing, and a participant with no Langfuse account must
   still be able to `import llm_utils` and run the MDP / reward / GRPO
   notebooks. When Langfuse *is* installed and keyed, tracing is identical to
   repo 1.
2. **`preflight()` lives in `config.py`** (it now also checks the GPU and a
   different key set) and is re-exported from here for source compatibility, so
   every `from llm_utils.llm import preflight` line from repo 1 still works.

The `llm()` signature is unchanged, which is what lets `LocalLM.as_llm_fn()`
stand in for it and drive the vendored agent unmodified.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# --- Observability is optional here (repo 1 hard-required it). -------------
LANGFUSE_AVAILABLE = False
try:
    # Drop-in replacement for `openai`: identical API, every call traced.
    from langfuse.openai import OpenAI
    from langfuse import get_client, observe

    LANGFUSE_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import/version problem falls back cleanly
    from openai import OpenAI  # type: ignore[assignment]

    def observe(*d_args, **d_kwargs):  # type: ignore[misc]
        """No-op stand-in for langfuse.observe when Langfuse isn't installed."""
        def _decorator(fn):
            return fn

        # Support both @observe and @observe(name="...")
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return _decorator

    def get_client():  # type: ignore[misc]
        raise RuntimeError(
            "Langfuse is not installed. `pip install langfuse` and set "
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to enable tracing."
        )


DEFAULT_MODEL = os.environ.get("WORKSHOP_MODEL", "gpt-4o-mini")

# Approximate USD per 1M tokens. Update if you use a different model/endpoint.
# The "self-hosted-*" rows are derived in NB7 from measured throughput and the
# GPU's hourly rate -- they are what make the cost axis of the Pareto chart real.
PRICING_PER_1M = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    # Filled in by NB7 from measured tokens/sec; these are placeholders until then.
    "self-hosted-t4": {"in": 0.0, "out": 0.0},
}

# Hourly USD rates used by NB7 to turn throughput into $/1k queries.
GPU_HOURLY_USD = {"T4": 0.35, "L4": 0.70, "A10G": 1.00, "A100-40GB": 2.00}


@dataclass
class CostMeter:
    """Tracks calls, tokens, and approximate cost across the whole notebook."""

    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def record(self, model: str, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.in_tokens += in_tok
        self.out_tokens += out_tok
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        m["calls"] += 1
        m["in"] += in_tok
        m["out"] += out_tok

    def cost(self) -> float:
        total = 0.0
        for model, m in self.by_model.items():
            p = PRICING_PER_1M.get(model, {"in": 0.0, "out": 0.0})
            total += m["in"] / 1_000_000 * p["in"]
            total += m["out"] / 1_000_000 * p["out"]
        return total

    def reset(self) -> None:
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.by_model = {}

    def report(self) -> str:
        return (
            f"[cost meter] calls={self.calls}  "
            f"in_tok={self.in_tokens:,}  out_tok={self.out_tokens:,}  "
            f"~${self.cost():.4f}"
        )

    def __str__(self) -> str:  # so `print(METER)` works
        return self.report()


# One global meter shared by every helper. Call METER.reset() between experiments.
METER = CostMeter()

_client = None


def _get_client():
    global _client
    if _client is None:
        base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default OpenAI
        _client = OpenAI(base_url=base_url) if base_url else OpenAI()
    return _client


def reset_client() -> None:
    """Drop the cached client so a changed OPENAI_BASE_URL takes effect.

    NB5 and NB7 repoint OPENAI_BASE_URL at an ART-served / self-hosted endpoint
    mid-notebook and then re-run the vendored `evaluate()`; without this they
    would keep talking to the old endpoint.
    """
    global _client
    _client = None


def llm(
    messages,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    meter: CostMeter | None = None,
    **kwargs,
) -> str:
    """Run one chat completion and return the assistant's text.

    `messages` may be a plain string (treated as a single user turn) or a list
    of {"role", "content"} dicts. Token usage is recorded on the global METER;
    the call is traced to Langfuse when it is installed and configured.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    model = model or DEFAULT_MODEL
    client = _get_client()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    usage = resp.usage
    (meter or METER).record(
        model,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )
    return resp.choices[0].message.content or ""


def embed(texts, model: str = "text-embedding-3-small"):
    """Return embedding vector(s) for a string or list of strings.

    Pass a single string -> get one vector back; pass a list -> get a list of
    vectors. `text-embedding-3-small` has 1536 dimensions.
    """
    single = isinstance(texts, str)
    inputs = [texts] if single else list(texts)
    resp = _get_client().embeddings.create(model=model, input=inputs)
    vectors = [d.embedding for d in resp.data]
    return vectors[0] if single else vectors


def flush() -> None:
    """Flush buffered Langfuse events. No-op when Langfuse isn't installed."""
    if not LANGFUSE_AVAILABLE:
        return
    try:
        get_client().flush()
    except Exception:  # noqa: BLE001 - never let telemetry break a notebook
        pass


# Re-exported for source compatibility with repo 1 (`from .llm import preflight`).
from .config import preflight  # noqa: E402  (deliberate: avoids a circular import)
