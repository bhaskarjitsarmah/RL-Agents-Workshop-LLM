"""Notebook cell definitions, and the builder that turns them into .ipynb.

Same discipline as repo 1: notebooks are GENERATED from plain-text cell
definitions and never hand-edited. Prose changes show up as readable diffs,
outputs are never committed, and `python build_notebooks.py` is the one command.

Repo 1 kept all seven notebooks in a single 1,670-line file. Nine notebooks with
training configs, dashboards, ablation replays and serving code would be
2,600-3,400 lines in one module -- unpleasant to edit and worse to review. So
the cell lists live in `nbsrc/nb0.py` ... `nb8.py`, and the shared preamble
lives here exactly once.

That sharing is the real win. Nine notebooks would otherwise each carry their
own copy of the Colab clone-and-install cell, and changing the install line
would mean nine edits and one forgotten notebook on the morning of the workshop.
"""

from __future__ import annotations

import os

import nbformat as nbf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(REPO_ROOT, "notebooks")

GITHUB_USER = "bhaskarjitsarmah"
REPO_NAME = "RL-Agents-Workshop-LLM"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


def build(path: str, cells: list, colab: bool = True) -> str:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    if colab:
        # So Colab opens with a T4 already selected rather than on CPU -- a
        # participant who misses the runtime switch loses ten minutes.
        nb["metadata"]["accelerator"] = "GPU"
        nb["metadata"]["colab"] = {"provenance": [], "gpuType": "T4",
                                   "toc_visible": True}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        nbf.write(nb, f)
    return path


# ===========================================================================
# Shared cells
# ===========================================================================

def COLAB_BADGE(nb_filename: str):
    url = (f"https://colab.research.google.com/github/{GITHUB_USER}/"
           f"{REPO_NAME}/blob/main/notebooks/{nb_filename}")
    return md(f"[![Open In Colab](https://colab.research.google.com/assets/"
              f"colab-badge.svg)]({url})")


def SETUP_CELL(needs_gpu: bool = False, extra_keys: tuple = ()):
    """Clone + install on Colab, no-op locally, then report capability.

    Idempotent: safe to re-run after the runtime restart that installing
    Unsloth forces (it replaces torch).
    """
    keys = ", ".join(repr(k) for k in extra_keys)
    gpu_note = """
if not CAP["gpu"]:
    print()
    print("No GPU detected -> REPLAY MODE.")
    print("Training cells will load pre-baked runs; every chart still renders.")
    print("In Colab: Runtime -> Change runtime type -> T4 GPU, then re-run.")
""" if needs_gpu else ""
    return code(f'''
# --- Setup. Safe to re-run. ---------------------------------------------
import os, sys, subprocess

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("/content/{REPO_NAME}"):
        subprocess.run(["git", "clone", "-q",
                        "{REPO_URL}.git", "/content/{REPO_NAME}"], check=True)
    os.chdir("/content/{REPO_NAME}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements-colab.txt"], check=False)
else:
    # Run from the REPO ROOT in both environments, so every relative path in
    # every notebook ("data/...") means the same thing whether you are on Colab
    # (cwd = repo root) or opened the file locally from notebooks/.
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
sys.path.insert(0, os.getcwd())

from llm_utils import build_db, preflight, capability, load_result, report_number
from llm_utils.plotting import use_house_style
import matplotlib.pyplot as plt

CAP = preflight({keys})
use_house_style()
print("Database ready at:", build_db()){gpu_note}
'''.strip())


def PREBAKE_HELPER():
    """Defines `baked(key, how)` -- the honest replay idiom.

    Every training cell in this repo is of the form "run it if we can, otherwise
    replay a real recorded run". What it must NEVER do is fabricate a curve when
    neither is available: a chart drawn from nothing is worse than no chart.
    """
    return code('''
def baked(key, how):
    """Load a pre-baked run, or explain exactly how to produce it.

    Returns None when the artifact is missing. Callers must check -- we would
    rather show no chart than an invented one.
    """
    data = load_result(key)
    if data is None:
        print(f"[{key}] not baked yet.")
        print(f"  Produce it with:  {how}")
        print("  Then re-run this cell. (The pre-baked files ship with the repo;")
        print("   you only need this if you are rebuilding them yourself.)")
    return data


PREBAKED = not CAP["gpu"]   # charts get a watermark when we are replaying
'''.strip())


def FOOTER_CELL(has_lm: bool = False, has_wandb: bool = False):
    lm = '\nprint(lm.stats)' if has_lm else ""
    wb = '\ntry:\n    import wandb; wandb.finish()\nexcept Exception:\n    pass' if has_wandb else ""
    return code(f'''
# --- Cost / throughput meter -------------------------------------------
from llm_utils import METER, flush{lm}
print(METER)          # OpenAI spend (0 unless you ran the comparison rows){wb}
flush()
'''.strip())


def EXERCISE(text: str):
    return md("### Exercise\n\n" + text.strip())


def TAKEAWAYS(items: list[str]):
    body = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
    return md("## Takeaways\n\n" + body)


def GAP(nb: str, text: str):
    return md(f"### The gap this leaves (-> {nb})\n\n{text.strip()}")


def RESTART_WARNING():
    return md(
        "> **Restart the runtime before this notebook.** Colab does not free GPU "
        "memory between notebooks, and a leftover model from the previous one is "
        "the most common cause of an out-of-memory error halfway through a "
        "training run.\n>\n"
        "> *Runtime -> Restart session*, then run the setup cell below."
    )
