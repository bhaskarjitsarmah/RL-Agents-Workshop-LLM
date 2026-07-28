"""Shared chart style, so both repos' decks look like one workshop.

Two conventions are enforced here rather than left to each notebook:

1. **The 0.75 line.** Repo 1's published baseline is drawn on almost every
   accuracy chart. The question this workshop asks is "did moving the weights
   beat moving the harness?", and the answer should be visible without reading
   an axis label.

2. **Error bars are not optional.** `bar_accuracy` takes records and draws
   Wilson intervals by default. At n=16 a bar chart without intervals is
   actively misleading -- it renders a 1-task difference as a visible step.

`prebaked=True` stamps a watermark on any figure built from replayed data, so a
participant with no GPU is never confused about whose run they are looking at.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

from .metrics import _k_n, wilson_ci

BASELINE_ACC = 0.75          # repo 1: gpt-4o-mini + repair loop, 16 test tasks

# Colour-blind-safe, consistent across both repos.
C_HARNESS = "#4C72B0"        # repo 1 / API models
C_WEIGHTS = "#DD8452"        # this repo / local policy
C_HYBRID = "#55A868"         # weights + skills
C_BAD = "#C44E52"            # ablations, pathologies, hacked runs
C_MUTED = "#8C8C8C"
PALETTE = [C_HARNESS, C_WEIGHTS, C_HYBRID, C_BAD, "#937860", "#8172B3"]


def use_house_style() -> None:
    matplotlib.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.size": 10,
    })


def _watermark(ax, prebaked: bool) -> None:
    if not prebaked:
        return
    ax.text(0.99, 0.02, "pre-baked replay", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=C_MUTED, alpha=0.8,
            style="italic")


def baseline_line(ax, value: float = BASELINE_ACC,
                  label: str = "repo 1 baseline (0.75)") -> None:
    ax.axhline(value, ls="--", lw=1.2, color=C_MUTED, zorder=0)
    ax.text(0.005, value + 0.015, label, transform=ax.get_yaxis_transform(),
            fontsize=8, color=C_MUTED)


def bar_accuracy(results: dict, title: str = "", baseline: float | None = BASELINE_ACC,
                 colors=None, prebaked: bool = False, ax=None,
                 show_ci: bool = True):
    """Accuracy bars WITH Wilson intervals.

    `results` maps label -> anything `report_number` understands (an evaluate()
    dict, a record list, or a (k, n) tuple).
    """
    use_house_style()
    if ax is None:
        _, ax = plt.subplots()
    labels = list(results)
    ks, ns = zip(*[_k_n(results[l]) for l in labels])
    accs = [k / n if n else 0.0 for k, n in zip(ks, ns)]
    cols = colors or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    err = None
    if show_ci:
        los, his = zip(*[wilson_ci(k, n) for k, n in zip(ks, ns)])
        err = [[a - lo for a, lo in zip(accs, los)],
               [hi - a for a, hi in zip(accs, his)]]

    ax.bar(labels, accs, color=cols, yerr=err, capsize=4,
           error_kw={"lw": 1, "ecolor": "#444"})
    for i, (a, k, n) in enumerate(zip(accs, ks, ns)):
        ax.text(i, a + 0.02, f"{a:.2f}\n({k}/{n})", ha="center", va="bottom",
                fontsize=8)
    if baseline is not None:
        baseline_line(ax, baseline)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("execution-match accuracy")
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _watermark(ax, prebaked)
    plt.tight_layout()
    return ax


def learning_curve(history: list[dict], keys=("reward",), x: str = "step",
                   title: str = "", prebaked: bool = False, ax=None,
                   baseline: float | None = None):
    """Training curves from a history list (live or replayed -- same code)."""
    use_house_style()
    if ax is None:
        _, ax = plt.subplots()
    xs = [h.get(x, i) for i, h in enumerate(history)]
    for i, k in enumerate(keys):
        ys = [h.get(k) for h in history]
        pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    label=k, color=PALETTE[i % len(PALETTE)], lw=1.6)
    if baseline is not None:
        baseline_line(ax, baseline)
    ax.set_xlabel(x)
    if title:
        ax.set_title(title)
    if len(keys) > 1:
        ax.legend()
    _watermark(ax, prebaked)
    plt.tight_layout()
    return ax


def scissors(history: list[dict], proxy_key: str = "proxy_reward",
             truth_key: str = "val_accuracy", x: str = "step",
             title: str = "Reward hacking: the proxy rises, the truth does not",
             prebaked: bool = False):
    """NB6's headline: proxy reward climbing while true accuracy falls.

    Two y-axes on purpose. The crossing shape is the entire lesson of Module 5,
    and it should be recognisable from the back of the room.
    """
    use_house_style()
    fig, ax = plt.subplots()
    xs = [h.get(x, i) for i, h in enumerate(history)]
    ax.plot(xs, [h.get(proxy_key) for h in history], color=C_BAD, lw=2,
            label="proxy reward (what we optimised)")
    ax.set_xlabel(x)
    ax.set_ylabel("proxy reward", color=C_BAD)
    ax.tick_params(axis="y", labelcolor=C_BAD)

    ax2 = ax.twinx()
    ax2.plot(xs, [h.get(truth_key) for h in history], color=C_HARNESS, lw=2,
             label="true val accuracy (what we wanted)")
    ax2.set_ylabel("true val accuracy", color=C_HARNESS)
    ax2.tick_params(axis="y", labelcolor=C_HARNESS)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax.set_title(title)
    _watermark(ax, prebaked)
    plt.tight_layout()
    return fig


def grpo_dashboard(history: list[dict], prebaked: bool = False,
                   baseline: float | None = BASELINE_ACC):
    """The six panels you actually need to diagnose a GRPO run.

    Panel 5 (`frac_zero_advantage`) is the one people forget and the one that
    explains most stalled runs: if every sample in a group scores the same, the
    step does nothing at all.
    """
    use_house_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    panels = [
        ("reward", "mean reward", None),
        ("kl", "KL to reference", None),
        ("completion_length", "mean completion length", None),
        ("frac_zero_advantage", "fraction of zero-advantage groups", None),
        ("grad_norm", "gradient norm", None),
        ("val_accuracy", "held-out val accuracy", baseline),
    ]
    xs = [h.get("step", i) for i, h in enumerate(history)]
    for ax, (key, label, base) in zip(axes.ravel(), panels):
        ys = [h.get(key) for h in history]
        pts = [(a, b) for a, b in zip(xs, ys) if b is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=C_WEIGHTS, lw=1.6)
        else:
            ax.text(0.5, 0.5, f"no '{key}' logged", ha="center", va="center",
                    transform=ax.transAxes, color=C_MUTED, fontsize=9)
        if base is not None:
            baseline_line(ax, base, "0.75")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step")
    _watermark(axes.ravel()[-1], prebaked)
    fig.suptitle("GRPO dashboard", fontweight="bold")
    plt.tight_layout()
    return fig


def reward_strip(groups: list[list[float]], labels=None, title: str = "",
                 prebaked: bool = False):
    """Per-group reward spread. NB1's chart.

    Flat columns are groups with zero advantage -- visibly wasted compute. This
    is the picture that makes the temperature sweep matter.
    """
    use_house_style()
    fig, ax = plt.subplots()
    for i, g in enumerate(groups):
        flat = len(set(round(r, 6) for r in g)) <= 1
        ax.scatter([i] * len(g), g, s=36, alpha=0.75,
                   color=C_BAD if flat else C_WEIGHTS,
                   label=None, zorder=3)
        if g:
            ax.hlines(sum(g) / len(g), i - 0.25, i + 0.25, color="#333", lw=1.5,
                      zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels or [f"t{i}" for i in range(len(groups))],
                       rotation=15, ha="right")
    ax.set_ylabel("reward")
    ax.set_title(title or "Reward spread within each sampled group "
                          "(red = zero advantage)")
    _watermark(ax, prebaked)
    plt.tight_layout()
    return fig


def pareto(points: list[dict], x: str = "cost_per_1k", y: str = "accuracy",
           label_key: str = "label", title: str = "Accuracy vs cost",
           prebaked: bool = False):
    """NB7/NB8: accuracy is one axis, $ and latency are the others."""
    use_house_style()
    fig, ax = plt.subplots()
    for i, p in enumerate(points):
        ax.scatter(p[x], p[y], s=90, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(p.get(label_key, ""), (p[x], p[y]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    baseline_line(ax, BASELINE_ACC)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y)
    ax.set_title(title)
    _watermark(ax, prebaked)
    plt.tight_layout()
    return fig


def correctness_heatmap(results: dict, title: str = "Which agent gets which task",
                        prebaked: bool = False):
    """Agents x tasks correctness grid.

    The argument for the hybrid, made visually: a fine-tuned small model and a
    large API model usually miss *different* tasks, and that is far more
    persuasive than two similar aggregate numbers.
    """
    import numpy as np

    use_house_style()
    labels = list(results)
    grids, ids = [], None
    for lab in labels:
        recs = sorted(results[lab]["records"], key=lambda r: str(r["id"]))
        ids = [r["id"] for r in recs]
        grids.append([1 if r["correct"] else 0 for r in recs])
    arr = np.array(grids)

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.55), 0.6 * len(labels) + 2))
    ax.imshow(arr, cmap=matplotlib.colors.ListedColormap(["#F2D7D5", "#A9DFBF"]),
              aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, fontsize=7, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("task id")
    ax.set_title(title)
    ax.grid(False)
    _watermark(ax, prebaked)
    plt.tight_layout()
    return fig
