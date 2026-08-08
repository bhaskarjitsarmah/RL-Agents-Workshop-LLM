"""Environment, capability detection, and the pre-baked-results contract.

This module is the reason every notebook in this repo runs *everywhere*.

**The hard rule (stated in ARCHITECTURE.md and enforced here):**
    Every notebook must render every chart on a machine with no GPU,
    no OpenAI key, and no network.

The mechanism is `capability()` plus `load_result()`. Training cells branch:

    history = train(...) if CAP["gpu"] else load_result("nb3_grpo_history")

and every plotting cell consumes `history` regardless of where it came from.
Charts built from replayed data get a "pre-baked" watermark so nobody mistakes
someone else's run for their own.

Nothing here imports torch at module scope -- a Windows/CPU participant doing
data generation and offline replay never needs it installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Where save_result()/load_result() live. Overridable because on Colab EVERY
# notebook is its own VM: NB3 trains the GRPO curve into its own /content and
# NB5, running in a different runtime, cannot see it -- so NB5's "TRL vs ART"
# comparison can never render, no matter what NB3 did. Point this at Drive
# (RESULTS_DIR=/content/drive/MyDrive/rl-workshop-results) and the results
# outlive the VM and are shared across notebooks.
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR", os.path.join(DATA_DIR, "results"))
ADAPTER_DIR = os.path.join(REPO_ROOT, "adapters")

# --- Identity -------------------------------------------------------------
REPO_NAME = "RL-Agents-Workshop-LLM"
REPO_URL = "https://github.com/bhaskarjitsarmah/RL-Agents-Workshop-LLM.git"

# Public HF Hub namespace holding the pre-baked adapters. Public on purpose:
# it keeps HF_TOKEN out of the required key set for every participant.
HF_NAMESPACE = os.environ.get("HF_NAMESPACE", "bhaskarjitsarmah")

# The policy under training. MODEL_SIZE lets a throttled runtime follow the
# whole arc at reduced accuracy (see SETUP.md).
MODEL_SIZE = os.environ.get("MODEL_SIZE", "1.5B")
BASE_MODELS = {
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
}
UNSLOTH_4BIT = {
    "1.5B": "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
    "0.5B": "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit",
}


def base_model(size: str | None = None) -> str:
    return BASE_MODELS[size or MODEL_SIZE]


def base_model_4bit(size: str | None = None) -> str:
    return UNSLOTH_4BIT[size or MODEL_SIZE]


def adapter_repo(tag: str, size: str | None = None) -> str:
    """Hub id for a pre-baked adapter, e.g. adapter_repo("grpo") ->
    'bhaskarjitsarmah/qwen25c-1.5b-sql-grpo'."""
    s = (size or MODEL_SIZE).lower()
    return f"{HF_NAMESPACE}/qwen25c-{s}-sql-{tag}"


# --- Keys -----------------------------------------------------------------
# Far shorter than repo 1's list. Training needs W&B; everything else is
# optional, because the policy is local and the reward is a local SQLite query.
REQUIRED_ENV: list[str] = ["WANDB_API_KEY"]
OPTIONAL_ENV: list[str] = [
    "OPENAI_API_KEY",       # only for the gpt-4o-mini comparison rows (NB0, NB8)
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "HF_TOKEN",             # only if you push your own adapters
]


def in_colab() -> bool:
    return "google.colab" in sys.modules


@lru_cache(maxsize=1)
def gpu_report() -> dict:
    """Everything about this machine that changes how the notebooks behave.

    Safe to call with no torch installed -- returns {"gpu": False, ...}.
    """
    rep: dict = {
        "gpu": False, "name": None, "capability": None, "total_gb": 0.0,
        "free_gb": 0.0, "bf16": False, "flash_attn": False, "vllm_ok": False,
        "colab": in_colab(), "platform": sys.platform,
    }
    for lib in ("torch", "transformers", "trl", "peft", "unsloth",
                "bitsandbytes", "datasets", "accelerate"):
        try:
            rep[lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001 - absence is information, not an error
            rep[lib] = None

    try:
        import torch
    except Exception:  # noqa: BLE001
        return rep

    if not torch.cuda.is_available():
        return rep

    props = torch.cuda.get_device_properties(0)
    cap = (props.major, props.minor)
    free_b, total_b = torch.cuda.mem_get_info()
    rep.update(
        gpu=True,
        name=props.name,
        capability=cap,
        total_gb=round(total_b / 1024**3, 2),
        free_gb=round(free_b / 1024**3, 2),
        # Whether to run the whole pipeline (model load + trainer) in bf16.
        # On current PyTorch this is True even on a T4 (bf16 is emulated on
        # Turing). We WANT that: the 4-bit checkpoints compute in bf16, so bf16
        # training keeps everything consistent AND avoids the fp16 GradScaler,
        # whose kernel has no bf16 version ("_amp_foreach_non_finite_check_and_
        # unscale_ not implemented for BFloat16"). The training configs key off
        # this flag (fp16 = not bf16), so model and trainer never disagree.
        bf16=bool(torch.cuda.is_bf16_supported()),
        flash_attn=cap >= (8, 0),
        vllm_ok=cap >= (8, 0),
    )
    return rep


def torch_dtype():
    """The ONE place the fp16-vs-bf16 decision is made.

    A T4 is Turing (sm_75) and has no bf16. Passing bf16 there produces either a
    hard error or -- worse -- silently degraded training. Autodetection has
    picked wrong on this card before, so every call site takes the answer from
    here instead of guessing.
    """
    import torch

    return torch.bfloat16 if gpu_report()["bf16"] else torch.float16


def dtype_kwarg(dtype=None) -> dict:
    """`{"dtype": ...}` or `{"torch_dtype": ...}` -- whichever this version takes.

    transformers renamed `torch_dtype` to `dtype` in v5 and changed the default
    to "auto". The old name then falls through to **kwargs and is IGNORED, so a
    T4 asking for float16 quietly gets the checkpoint's bfloat16 instead. In
    training that surfaces as a GradScaler crash; at inference it surfaces as
    nothing at all -- just emulated bf16, slower, and a different dtype from the
    one the adapter was trained in.

    Decided from the version, not `inspect`: the Auto* classes take
    `(*model_args, **kwargs)`, so the parameter never appears in the signature
    under either name.
    """
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype" if major >= 5 else "torch_dtype": dtype or torch_dtype()}


def quiet_generation_config(model) -> None:
    """Drop `max_length` so it stops fighting our `max_new_tokens`.

    Qwen ships `max_length=32768` in generation_config. Every single generate()
    call then prints a five-line "Both max_new_tokens and max_length seem to
    have been set" warning -- hundreds of them across one notebook, burying the
    output participants are supposed to read. `max_new_tokens` was already
    winning; this only stops the narration.
    """
    cfg = getattr(model, "generation_config", None)
    if cfg is not None and getattr(cfg, "max_length", None):
        cfg.max_length = None


def free_vram() -> float:
    """Free GiB on device 0 (0.0 with no GPU). Print it before every big run."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
    except Exception:  # noqa: BLE001
        return 0.0


def empty_cache() -> None:
    """Free what can be freed. Pair with `LocalLM.unload()` between notebooks."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def capability() -> dict:
    """The dict every notebook branches on. `CAP = capability()`."""
    rep = gpu_report()
    return {
        "gpu": rep["gpu"],
        "bf16": rep["bf16"],
        "vllm": rep["vllm_ok"],
        "colab": rep["colab"],
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "wandb": bool(os.environ.get("WANDB_API_KEY")),
        "langfuse": bool(os.environ.get("LANGFUSE_PUBLIC_KEY")
                         and os.environ.get("LANGFUSE_SECRET_KEY")),
        "train": rep["gpu"] and rep.get("torch") is not None,
        "name": rep["name"],
        "total_gb": rep["total_gb"],
    }


def preflight(*extra_keys: str, require_gpu: bool = False,
              require_openai: bool = False, quiet: bool = False) -> dict:
    """Fail fast on a missing key; warn (don't fail) on a missing capability.

    Deliberately gentler than repo 1's preflight. There, a missing key meant the
    notebook could not run at all. Here the only *hard* requirement is whatever
    the caller declares -- because a GPU-less participant is expected to run
    most of this repo in replay mode, and that must not raise.
    """
    required = list(extra_keys)
    if require_openai:
        required.append("OPENAI_API_KEY")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Missing required keys in your .env: " + ", ".join(missing)
            + "\nCopy .env.example to .env and fill them in -- see SETUP.md."
        )

    cap = capability()
    if require_gpu and not cap["gpu"]:
        raise RuntimeError(
            "This cell needs a GPU. In Colab: Runtime -> Change runtime type -> T4 GPU.\n"
            "Or set USE_PREBAKED = True to replay the pre-baked run instead."
        )

    if not quiet:
        rep = gpu_report()
        if cap["gpu"]:
            print(f"GPU: {rep['name']}  sm_{rep['capability'][0]}{rep['capability'][1]}  "
                  f"{rep['total_gb']} GB  bf16={rep['bf16']}  dtype="
                  f"{'bfloat16' if rep['bf16'] else 'float16'}")
        else:
            print("GPU: none -> replay mode. Training cells will load pre-baked "
                  "results; every chart still renders.")
        vers = {k: rep[k] for k in ("torch", "transformers", "trl", "peft", "unsloth")
                if rep.get(k)}
        if vers:
            print("versions:", "  ".join(f"{k}={v}" for k, v in vers.items()))
        have = [k for k in REQUIRED_ENV + OPTIONAL_ENV if os.environ.get(k)]
        print("keys present:", ", ".join(have) if have else "(none)")
    return cap


def setup_colab(repo_url: str = REPO_URL, branch: str = "main",
                reqs: str = "requirements-colab.txt") -> None:
    """Clone + install on Colab; no-op everywhere else. Idempotent.

    NOTE: installing Unsloth replaces torch, so Colab usually demands a runtime
    restart afterwards. We detect that and print a loud instruction rather than
    letting the notebook die three cells later with a cryptic CUDA error.
    """
    if not in_colab():
        return
    name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    if not os.path.exists(f"/content/{name}"):
        subprocess.run(["git", "clone", "-q", "-b", branch, repo_url,
                        f"/content/{name}"], check=True)
    os.chdir(f"/content/{name}")
    before = _torch_version()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", reqs], check=False)
    after = _torch_version()
    if before and after and before != after:
        print("=" * 68)
        print("  torch was replaced during install "
              f"({before} -> {after}).")
        print("  RESTART THE RUNTIME NOW (Runtime -> Restart session), then")
        print("  re-run this cell. It will skip the install and continue.")
        print("=" * 68)


def _torch_version() -> str | None:
    try:
        return __import__("torch").__version__
    except Exception:  # noqa: BLE001
        return None


# --- Pre-baked results ----------------------------------------------------

def result_path(key: str) -> str:
    return os.path.join(RESULTS_DIR, f"{key}.json")


def save_result(key: str, obj) -> str:
    """Persist a training history / eval result so it can be replayed offline."""
    path = result_path(key)
    # dirname, not RESULTS_DIR: nested keys like "pathologies/kl_blowup" need
    # their subdirectory too, and only finding that out AFTER the run that
    # produced the artifact is the most expensive way to learn it.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def _search_paths(key: str) -> list[str]:
    """Where a result may live: the write location first, then the repo's own.

    With RESULTS_DIR pointed at Drive, the pre-baked files that SHIP with the
    repo would otherwise become invisible -- you would gain cross-notebook
    persistence and lose every artifact the repo already provides.
    """
    default = os.path.join(DATA_DIR, "results")
    dirs = [RESULTS_DIR] + ([default] if default != RESULTS_DIR else [])
    return [os.path.join(d, f"{key}.json") for d in dirs]


def load_result(key: str, default=None):
    """Load a pre-baked result. Returns `default` if it hasn't been baked yet.

    Callers should treat a None return as "this chart can't be drawn here" and
    say so in the output -- never silently plot an empty series.
    """
    for path in _search_paths(key):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return default


def have_result(key: str) -> bool:
    return any(os.path.exists(p) for p in _search_paths(key))
