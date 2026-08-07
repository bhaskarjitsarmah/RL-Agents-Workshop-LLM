"""Run a local policy behind the SAME interface as `llm()`.

This is the hinge of the whole repo. `LocalLM.as_llm_fn()` returns a callable
with `llm()`'s exact signature, so:

    evaluate(make_agent(),                          split="test")  # gpt-4o-mini
    evaluate(make_local_agent(qwen_base),           split="test")  # Qwen zero-shot
    evaluate(make_local_agent(qwen_grpo),           split="test")  # Qwen after GRPO
    evaluate(make_local_agent(qwen_grpo, extra=SK), split="test")  # + repo 1's skills

Four rows of NB8's table, one scorer, zero goalpost-moving. The agent, the
prompts, the repair loop, and the parser are byte-identical across all of them;
only the weights behind `llm_fn` change.

Backends
--------
    unsloth  4-bit LoRA training path, also usable for generation
    hf       plain transformers + bitsandbytes -- the fallback when Unsloth's
             install fails on the day, which it sometimes does
    vllm     fast batched generation; attempted on Turing, not relied upon
    openai   any OpenAI-compatible endpoint, including ART's served model and a
             self-hosted vLLM server. Lets a GPU-less participant still run live
             rollouts in NB1.

Three details that silently ruin runs, handled here once
-------------------------------------------------------
* **Chat template.** Qwen2.5-Coder-Instruct is ChatML. Formatting prompts by
  hand in one place and via `apply_chat_template` in another produces a
  fine-tuned model that scores *worse* than the base -- the single most common
  silent failure in this kind of work. Everything goes through `_render`.
* **Padding side.** Left for generation, right for training. Getting this
  backwards corrupts batched generation in ways that look like a bad model.
* **temperature=0.** HF errors or warns on `temperature=0`; it must become
  `do_sample=False`. The eval harness calls with 0.0 by default.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from .agents import make_agent
from .config import base_model, base_model_4bit, gpu_report, torch_dtype

DEFAULT_STOP = ("```\n\n", "\n\nQuestion:", "<|im_end|>")


class LocalLM:
    """A local (or remote-OpenAI-compatible) policy with a uniform interface."""

    def __init__(self, model_id: str | None = None, adapter: str | None = None,
                 backend: str = "auto", max_seq_len: int = 2048,
                 load_in_4bit: bool = True, dtype=None,
                 base_url: str | None = None, api_model: str | None = None,
                 tokenizer=None, model=None):
        self.model_id = model_id
        self.adapter = adapter
        self.max_seq_len = max_seq_len
        self.load_in_4bit = load_in_4bit
        self.base_url = base_url
        self.api_model = api_model
        self.model = model
        self.tokenizer = tokenizer
        self._client = None
        self.backend = self._resolve_backend(backend)
        self._stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "wall_s": 0.0}
        if self.model is None and self.backend != "openai":
            self._load(dtype)

    # -- construction ------------------------------------------------------
    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if self.base_url:
            return "openai"
        rep = gpu_report()
        if not rep["gpu"]:
            # No GPU: an OpenAI-compatible endpoint is the only live option.
            return "openai" if os.environ.get("OPENAI_BASE_URL") or \
                os.environ.get("OPENAI_API_KEY") else "hf"
        return "unsloth" if rep.get("unsloth") else "hf"

    def _load(self, dtype=None) -> None:
        import torch

        dtype = dtype or torch_dtype()
        mid = self.model_id or (base_model_4bit() if self.load_in_4bit
                                else base_model())
        self.model_id = mid

        if self.backend == "unsloth":
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=mid, max_seq_length=self.max_seq_len,
                dtype=dtype,               # explicit: a T4 is fp16-only
                load_in_4bit=self.load_in_4bit,
                fast_inference=False,      # vLLM path is unreliable on Turing
            )
            FastLanguageModel.for_inference(self.model)
        elif self.backend == "vllm":
            from vllm import LLM

            self.model = LLM(model=mid, dtype="float16",
                             max_model_len=self.max_seq_len,
                             enable_lora=bool(self.adapter))
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model())
        else:  # hf
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)

            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            ) if self.load_in_4bit else None
            self.tokenizer = AutoTokenizer.from_pretrained(mid)
            self.model = AutoModelForCausalLM.from_pretrained(
                mid, quantization_config=quant, torch_dtype=dtype,
                device_map="auto", attn_implementation="sdpa",  # FA2 needs sm_80+
            )
            self.model.eval()

        if self.tokenizer is not None:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"   # generation; training flips it

        if self.adapter:
            self.load_adapter(self.adapter)

    def load_adapter(self, path_or_repo: str) -> None:
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, path_or_repo)
        self.model.eval()
        self.adapter = path_or_repo

    def unload(self) -> None:
        """Free the GPU. Call between notebooks; Colab will not do it for you."""
        from .config import empty_cache

        self.model = None
        self.tokenizer = None
        self._client = None
        empty_cache()

    # -- generation --------------------------------------------------------
    def _render(self, messages) -> str:
        """The ONE place a prompt becomes a string."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages, n: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 256, stop=None, seed=None) -> list[str]:
        t0 = time.time()
        try:
            if self.backend == "openai":
                out = self._generate_openai(messages, n, temperature, max_new_tokens)
            elif self.backend == "vllm":
                out = self._generate_vllm(messages, n, temperature, max_new_tokens, stop)
            else:
                out = self._generate_hf(messages, n, temperature, max_new_tokens, seed)
        finally:
            self._stats["wall_s"] += time.time() - t0
        self._stats["calls"] += n
        return out

    def _generate_hf(self, messages, n, temperature, max_new_tokens, seed):
        import torch

        prompt = self._render(messages)
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=self.max_seq_len)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        self._stats["prompt_tokens"] += int(enc["input_ids"].shape[-1]) * n
        if seed is not None:
            torch.manual_seed(seed)
        # temperature=0 is not a valid sampling temperature; it means greedy.
        do_sample = temperature > 0
        kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  num_return_sequences=n,
                  pad_token_id=self.tokenizer.pad_token_id)
        if do_sample:
            kw.update(temperature=temperature, top_p=1.0)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        gen = out[:, enc["input_ids"].shape[-1]:]
        self._stats["completion_tokens"] += int(gen.numel())
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def _generate_vllm(self, messages, n, temperature, max_new_tokens, stop):
        from vllm import SamplingParams

        params = SamplingParams(n=n, temperature=temperature,
                                max_tokens=max_new_tokens,
                                stop=list(stop or DEFAULT_STOP))
        outs = self.model.generate([self._render(messages)], params)
        return [o.text for o in outs[0].outputs]

    def _generate_openai(self, messages, n, temperature, max_new_tokens):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
            )
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resp = self._client.chat.completions.create(
            model=self.api_model or self.model_id or base_model(),
            messages=messages, temperature=temperature,
            max_tokens=max_new_tokens, n=n,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self._stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self._stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return [c.message.content or "" for c in resp.choices]

    def generate_batch(self, batch_messages, n: int = 1, temperature: float = 0.0,
                       max_new_tokens: int = 256) -> list[list[str]]:
        """True batched generation for the 200-task eval sets.

        Falls back to a loop on backends that cannot batch, so callers never
        need to branch.
        """
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(n=n, temperature=temperature,
                                    max_tokens=max_new_tokens)
            outs = self.model.generate([self._render(m) for m in batch_messages],
                                       params)
            self._stats["calls"] += len(batch_messages) * n
            return [[o.text for o in r.outputs] for r in outs]

        if self.backend in ("unsloth", "hf") and n == 1:
            import torch

            prompts = [self._render(m) for m in batch_messages]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_seq_len)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            self._stats["prompt_tokens"] += int(enc["input_ids"].numel())
            t0 = time.time()
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    **({"temperature": temperature} if temperature > 0 else {}),
                    pad_token_id=self.tokenizer.pad_token_id)
            self._stats["wall_s"] += time.time() - t0
            gen = out[:, enc["input_ids"].shape[-1]:]
            self._stats["completion_tokens"] += int(gen.numel())
            self._stats["calls"] += len(prompts)
            return [[t] for t in self.tokenizer.batch_decode(gen,
                                                             skip_special_tokens=True)]

        return [self.generate(m, n=n, temperature=temperature,
                              max_new_tokens=max_new_tokens)
                for m in batch_messages]

    # -- adapters ----------------------------------------------------------
    def as_llm_fn(self) -> Callable:
        """A drop-in replacement for `llm()`. THE integration point."""

        def _llm(messages, model=None, temperature: float = 0.0,
                 max_tokens: int = 800, **kw) -> str:
            outs = self.generate(messages, n=1, temperature=temperature,
                                 max_new_tokens=min(max_tokens, 512))
            return outs[0] if outs else ""

        return _llm

    def as_policy(self) -> Callable:
        """The `rollout.Policy` interface: n completions per call."""

        def _policy(messages, n: int = 1, temperature: float = 0.7,
                    max_new_tokens: int = 256, **kw) -> list[str]:
            return self.generate(messages, n=n, temperature=temperature,
                                 max_new_tokens=max_new_tokens)

        return _policy

    @property
    def stats(self) -> dict:
        """The cost-meter analogue. `print(lm.stats)` mirrors repo 1's ritual."""
        s = dict(self._stats)
        s["tok_per_s"] = (s["completion_tokens"] / s["wall_s"]) if s["wall_s"] else 0.0
        s["backend"] = self.backend
        s["model"] = self.model_id
        s["adapter"] = self.adapter
        return s

    def reset_stats(self) -> None:
        self._stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "wall_s": 0.0}

    def __repr__(self) -> str:
        return (f"LocalLM(backend={self.backend!r}, model={self.model_id!r}, "
                f"adapter={self.adapter!r})")


def load_policy(**kw) -> LocalLM:
    return LocalLM(**kw)


def make_local_agent(lm: LocalLM, extra: str = "", max_repairs: int = 2):
    """The vendored agent, driven by a local policy.

    One line, and it is the reason NB8's hybrid row is one line too.
    """
    return make_agent(extra=extra, max_repairs=max_repairs, llm_fn=lm.as_llm_fn())
