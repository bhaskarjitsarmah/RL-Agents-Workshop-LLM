"""NB7 - Deployment: Merge, Serve, Measure (AV Module 5)."""

from . import (COLAB_BADGE, EXERCISE, FOOTER_CELL, GAP, PREBAKE_HELPER,
               RESTART_WARNING, SETUP_CELL, TAKEAWAYS, code, md)

CELLS = [
    COLAB_BADGE("NB7_deployment.ipynb"),
    md(r"""
# NB7 - Deployment: Merge, Serve, Measure

An adapter in a Colab VM is not a system. This notebook turns it into one, and
then asks the question that decides whether any of this ships:

> **What does it cost per thousand queries, and how long does a user wait?**

Accuracy is one axis. Latency and cost are the other two, and the decision lives
on the Pareto front rather than at the top of a leaderboard.
"""),
    RESTART_WARNING(),
    SETUP_CELL(needs_gpu=True),
    PREBAKE_HELPER(),
    md(r"""
## 1. Merge the adapter

LoRA at `r=16` over seven projections on a 1.5B model is ~18M parameters - about
**36 MB in fp16**. That is why the pre-baked adapters download in seconds.

For serving we merge them into the base weights, which removes the adapter
indirection at inference time.
"""),
    code(r"""
from llm_utils.config import adapter_repo, base_model
print("base   :", base_model())
print("adapter:", adapter_repo("grpo"))

if CAP["gpu"]:
    from llm_utils.trainers import merge_and_save
    merged = merge_and_save(adapter_repo("grpo"), "out/merged-fp16")
else:
    merged = None
    print("\n(merge needs a GPU; the serving numbers below are pre-baked)")
"""),
    md(r"""
### Verify the merge before trusting it

A merge bug is silent: the merged model still produces fluent SQL, just slightly
different SQL. It shows up here or it does not show up at all.

**The merged model must reproduce the adapter's test-16 accuracy exactly.**
"""),
    code(r"""
from llm_utils import evaluate
from llm_utils.local_llm import LocalLM, make_local_agent

if CAP["gpu"] and merged:
    lm_adapter = LocalLM(adapter=adapter_repo("grpo"))
    lm_merged = LocalLM(model_id=merged, load_in_4bit=False)
    a = evaluate(make_local_agent(lm_adapter), split="test")
    b = evaluate(make_local_agent(lm_merged), split="test")
    print(report_number(a, "adapter"))
    print(report_number(b, "merged "))
    same = [x["correct"] for x in sorted(a["records"], key=lambda r: r["id"])] == \
           [x["correct"] for x in sorted(b["records"], key=lambda r: r["id"])]
    print(f"\nper-item identical: {same}")
    if not same:
        print("!! MERGE BUG. Do not serve this. Check dtype and adapter path.")
else:
    v = baked("nb7_merge_check",
                  "python scripts/bake_all.py --stage deploy")
    if v:
        print("adapter vs merged, per-item identical:", v["identical"])
"""),
    md(r"""
## 2. Serve it - three tiers, with fallback

| tier | when | note |
|---|---|---|
| **vLLM** | sm_80+ (A10, L4, A100) | fastest; unreliable on a Turing T4 |
| **FastAPI + transformers** | anywhere with a GPU | simple, adequate, what we usually get |
| **llama.cpp (GGUF)** | CPU only | slow but genuinely deployable on a laptop |

The notebook tries them in order and reports which one came up, rather than
assuming.
"""),
    code(r"""
served = baked("nb7_serving",
                  "python scripts/bake_all.py --stage deploy")
if served:
    print(f"tier that came up: {served['tier']}")
    print(f"reason: {served.get('reason', '-')}")
"""),
    md(r"""
### Then score the served endpoint with the *vendored* `evaluate()`

Point `OPENAI_BASE_URL` at whatever came up, and run the same function that
produced repo 1's 0.75. If the served number differs from the in-process number,
something in the serving path is wrong - and you want to know that now, not from
a user.
"""),
    code(r"""
import os
from llm_utils import reset_client

if served and served.get("base_url") and CAP["gpu"]:
    os.environ["OPENAI_BASE_URL"] = served["base_url"]
    reset_client()                       # or we keep talking to the old endpoint
    lm_served = LocalLM(backend="openai", base_url=served["base_url"],
                        api_model=served.get("model"))
    res_served = evaluate(make_local_agent(lm_served), split="test")
    print(report_number(res_served, "served endpoint"))
else:
    v = baked("nb7_served_eval",
                  "python scripts/bake_all.py --stage deploy")
    if v:
        print(report_number(tuple(v["test16"]), "served endpoint"))
        print("in-process vs served, identical:", v.get("matches_in_process"))
"""),
    md(r"""
## 3. Latency and throughput
"""),
    code(r"""
lat = baked("nb7_latency",
                  "python scripts/bake_all.py --stage deploy")
if lat:
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for name, xs in lat["latency_ms"].items():
        ax[0].hist(xs, bins=24, alpha=0.6, label=name)
    ax[0].set_xlabel("latency (ms)"); ax[0].set_title("per-query latency")
    ax[0].legend()
    bs = list(lat["throughput"])
    ax[1].plot(bs, [lat["throughput"][b] for b in bs], marker="o", color="#DD8452")
    ax[1].set_xlabel("batch size"); ax[1].set_ylabel("queries / second")
    ax[1].set_title("throughput vs batch size")
    plt.tight_layout(); plt.show()

    for name, xs in lat["latency_ms"].items():
        s = sorted(xs)
        print(f"  {name:<22} p50 {s[len(s)//2]:>7.0f} ms   "
              f"p95 {s[int(len(s)*0.95)]:>7.0f} ms")
"""),
    md(r"""
## 4. The number your manager will ask for

Self-hosting has a fixed hourly cost whether or not anyone is using it. An API
has a marginal cost per token and none when idle.

So self-hosting wins **above a break-even QPS** and loses below it. Compute it
rather than asserting it.
"""),
    code(r"""
from llm_utils.llm import GPU_HOURLY_USD, PRICING_PER_1M

if lat:
    qps = max(lat["throughput"].values())
    gpu_hr = GPU_HOURLY_USD["T4"]
    self_cost_1k = gpu_hr / (qps * 3600) * 1000

    in_tok, out_tok = lat.get("mean_prompt_tokens", 700), lat.get("mean_completion_tokens", 60)
    p = PRICING_PER_1M["gpt-4o-mini"]
    api_cost_1k = (in_tok * p["in"] + out_tok * p["out"]) / 1e6 * 1000

    print(f"self-hosted T4  : {qps:.2f} q/s at ${gpu_hr}/hr -> ${self_cost_1k:.3f} / 1k queries")
    print(f"gpt-4o-mini API : {in_tok} in + {out_tok} out tokens -> ${api_cost_1k:.3f} / 1k queries")
    breakeven = gpu_hr / (api_cost_1k / 1000 * 3600)
    print(f"\nbreak-even: ~{breakeven:.2f} queries/second sustained.")
    print("Below that, the API is cheaper. Above it, the GPU is.")
    print("This is the slide that decides the project, and it is one division.")
"""),
    code(r"""
pareto_pts = baked("nb7_pareto",
                  "python scripts/bake_all.py --stage deploy")
if pareto_pts:
    from llm_utils.plotting import pareto
    pareto(pareto_pts, title="Accuracy vs cost per 1k queries", prebaked=PREBAKED)
    plt.show()
"""),
    md(r"""
### What the Pareto front actually says

A fine-tuned 1.5B that matches a much larger API model is not interesting because
it is *better*. It is interesting because it is **the smallest thing that holds
that accuracy**, and small is what makes it cheap, fast, on-prem-able, and yours.

That, and not a leaderboard position, is the case for optimizing the weights.
"""),
    TAKEAWAYS([
        "**Verify the merge per-item.** A merge bug is silent - the model still "
        "writes fluent SQL, just subtly different SQL.",
        "Score the **served** endpoint with the vendored `evaluate()`. A serving "
        "path that changes the number is a bug you want to find before a user does.",
        "Serving has tiers and the fast one is not always available. Try, "
        "measure, and report which came up.",
        "**Compute the break-even QPS.** Self-hosting is a fixed cost and an API "
        "is marginal; that one division decides the architecture.",
        "The value of a fine-tuned small model is not that it wins, it is that it "
        "is **the smallest thing that holds the accuracy**.",
    ]),
    GAP("NB8", """
Two workshops. Two philosophies. One scoreboard that has been identical the
whole way through.

Time to settle it - and to find out whether the question "harness or weights?"
was even the right question.
"""),
    EXERCISE("""
1. Re-run the break-even calculation for an A10 at $1.00/hr. Does the answer
   change the architecture you would choose?
2. Quantise the merged model to 4-bit for serving and re-score. How much accuracy
   does the compression cost, and how much latency does it buy?
3. Your traffic is 0.2 QPS at midday and 0.001 QPS overnight. What do you deploy?
   (There is more than one defensible answer.)
"""),
    FOOTER_CELL(has_lm=False),
]
