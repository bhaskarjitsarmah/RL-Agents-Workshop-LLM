"""The notebooks are generated artifacts, and these tests keep them that way.

Two failure modes this catches:

1. **A hand-edited .ipynb.** The prose lives in `nbsrc/`; a fix made directly in
   the notebook JSON survives until the next `python build_notebooks.py` and then
   vanishes, usually the morning of the workshop. Rebuilding and comparing makes
   that impossible to miss.

2. **A broken code cell.** An unbalanced quote inside a triple-quoted cell
   definition is invisible until it is a SyntaxError in front of a room.

Full execution (every cell, no GPU, no keys) is a separate, slower check:

    python scripts/run_notebook_cells.py notebooks/NB0_*.ipynb
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_notebooks import NOTEBOOKS  # noqa: E402
from nbsrc import NB_DIR, build  # noqa: E402

MODULES = [m for m, _ in NOTEBOOKS]


@pytest.mark.parametrize("mod_name,filename", NOTEBOOKS)
def test_notebook_exists_and_is_valid_json(mod_name, filename):
    path = os.path.join(NB_DIR, filename)
    assert os.path.exists(path), f"{filename} missing -- run python build_notebooks.py"
    nb = json.load(open(path, encoding="utf-8"))
    assert nb["cells"], f"{filename} has no cells"
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


@pytest.mark.parametrize("mod_name,filename", NOTEBOOKS)
def test_every_code_cell_parses(mod_name, filename):
    """Jupyter allows top-level `await`, so we compile the way it does."""
    mod = importlib.import_module(f"nbsrc.{mod_name}")
    for i, (kind, src) in enumerate(mod.CELLS):
        if kind != "code":
            continue
        try:
            compile(src, "<cell>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            pytest.fail(f"{mod_name} cell {i}: {e.msg} at line {e.lineno}")


@pytest.mark.parametrize("mod_name,filename", NOTEBOOKS)
def test_notebook_matches_its_source(mod_name, filename, tmp_path):
    """The checked-in .ipynb must be exactly what nbsrc/ generates.

    If this fails, someone edited the notebook directly. Re-apply the change in
    `nbsrc/{mod_name}.py` and run `python build_notebooks.py`.
    """
    mod = importlib.import_module(f"nbsrc.{mod_name}")
    regenerated = build(str(tmp_path / filename), mod.CELLS)
    current = os.path.join(NB_DIR, filename)
    a = json.load(open(regenerated, encoding="utf-8"))
    b = json.load(open(current, encoding="utf-8"))
    src_a = ["".join(c["source"]) for c in a["cells"]]
    src_b = ["".join(c["source"]) for c in b["cells"]]
    assert src_a == src_b, (
        f"{filename} differs from nbsrc/{mod_name}.py. Notebooks are generated: "
        f"edit the source module and re-run python build_notebooks.py")


def test_all_nine_notebooks_are_present():
    assert len(NOTEBOOKS) == 9
    for _, filename in NOTEBOOKS:
        assert os.path.exists(os.path.join(NB_DIR, filename))


@pytest.mark.parametrize("mod_name,filename", NOTEBOOKS)
def test_every_notebook_opens_with_badge_and_setup(mod_name, filename):
    """A participant must be able to click into Colab and run cell 1."""
    mod = importlib.import_module(f"nbsrc.{mod_name}")
    assert mod.CELLS[0][0] == "md"
    assert "colab.research.google.com" in mod.CELLS[0][1], "missing Colab badge"
    first_code = next(src for k, src in mod.CELLS if k == "code")
    for expected in ("IN_COLAB", "preflight", "build_db"):
        assert expected in first_code, (
            f"{mod_name}'s first code cell is not the shared setup cell "
            f"(missing {expected!r})")


def _baked_calls() -> list[tuple[str, str, str]]:
    """(module, key, command) for every baked(...) call across the notebooks."""
    import re

    out = []
    for mod_name, _ in NOTEBOOKS:
        mod = importlib.import_module(f"nbsrc.{mod_name}")
        for _kind, src in mod.CELLS:
            for key, cmd in re.findall(
                    r'baked\(\s*f?"([^"]+)"\s*,\s*\n?\s*"([^"]*)"', src):
                out.append((mod_name, key, cmd))
    return out


def test_every_baked_key_has_a_producer():
    """No notebook may tell a participant to run a command that does not exist.

    Every `baked()` call prints its command when the artifact is missing -- which
    is the normal state before the pre-bake run. Earlier drafts named seven
    scripts (run_grpo.py, run_sft.py, run_deploy.py, ...) that were never
    written, so the instruction on screen was a dead end. Worse than admitting
    the artifact is missing.
    """
    import re

    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "bake_all.py"),
        encoding="utf-8").read()
    stages = set(re.findall(r'\("([a-z]+)", stage_', src))
    assert stages, "could not parse the stage list out of bake_all.py"

    # The key each stage actually WRITES, not the key its description claims.
    # Checking the description is how the dead ends below shipped: the `sft`
    # stage's docstring advertised `nb2_ablations` and no code ever wrote it, so
    # the cell told participants to run a command that could not help them.
    # A stage name existing is not evidence of a producer.
    produced = {k.rstrip("/") for k in
                re.findall(r'save_result\(\s*f?"([^"{]*)', src)}

    def has_producer(key: str) -> bool:
        # f-string keys ("pathologies/{k}") match on their literal prefix.
        return key.split("{")[0].rstrip("/") in produced

    #: Keys the notebooks offer a replay for that nothing in bake_all.py writes
    #: yet. Listed rather than silently tolerated: each one is a cell that will
    #: print "not baked yet" no matter what a participant runs. Shrink this
    #: list by writing the producer; never grow it to make a test pass.
    KNOWN_UNPRODUCED = {
        "nb4_multiturn",      # needs a multi-turn training run, not just sweeps
        "nb5_art_eval",       # stage_art saves history but never scores it
        "nb6_human_labels",   # the HITL agreement sample is not collected
        "nb7_serving",        # serving tier is chosen live, never recorded
        "nb7_served_eval",
        "nb8_pareto",         # stage_deploy writes nb7_pareto, not this one
    }

    problems = []
    for mod_name, key, cmd in _baked_calls():
        if "--stage" not in cmd:
            problems.append(f"{mod_name}: {key!r} -> {cmd!r} (no --stage)")
            continue
        named = cmd.split("--stage ")[-1].strip().split()[0]
        unknown = [s for s in named.split(",") if s not in stages]
        if unknown:
            problems.append(f"{mod_name}: {key!r} -> unknown stage(s) {unknown}")
        elif not has_producer(key) and key not in KNOWN_UNPRODUCED:
            problems.append(
                f"{mod_name}: {key!r} -> `{cmd}` runs, but no save_result() in "
                f"bake_all.py ever writes {key!r}. Write the producer, or add "
                f"the key to KNOWN_UNPRODUCED with a reason.")
    assert not problems, "\n".join(problems)

    stale = sorted(k for k in KNOWN_UNPRODUCED if has_producer(k))
    assert not stale, (
        f"these now have producers -- remove them from KNOWN_UNPRODUCED: {stale}")


def test_referenced_scripts_exist():
    """Any scripts/*.py named in a notebook or a doc must actually be there."""
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = [os.path.join(root, "nbsrc", f"{m}.py") for m, _ in NOTEBOOKS]
    sources += [os.path.join(root, d) for d in
                ("README.md", "SETUP.md", "COLAB.md", "ARCHITECTURE.md")]

    referenced: set[str] = set()
    for path in sources:
        if os.path.exists(path):
            referenced |= set(re.findall(r"scripts/[a-z_]+\.py",
                                         open(path, encoding="utf-8").read()))
    missing = [s for s in sorted(referenced)
               if not os.path.exists(os.path.join(root, s))]
    assert not missing, f"referenced but absent: {missing}"


def test_diagram_exports_exist_and_match_the_docs():
    """Every Mermaid block in the docs has a slide-ready SVG + PNG export.

    Only checks presence and the source-block count, not pixels: re-rendering
    needs Node, and a test that silently skips when Node is absent is worse than
    one that checks what it can. `python scripts/export_diagrams.py --check`
    does the byte comparison when you have the toolchain.
    """
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from export_diagrams import DIAGRAMS, FENCE  # noqa: E402

    for stem, doc, idx, _title in DIAGRAMS:
        found = FENCE.findall(open(os.path.join(root, doc), encoding="utf-8").read())
        assert idx < len(found), (
            f"{stem}: {doc} has {len(found)} mermaid blocks, wanted #{idx}. "
            f"A diagram was removed or reordered -- update DIAGRAMS in "
            f"scripts/export_diagrams.py")
        for ext in (".svg", ".png"):
            path = os.path.join(root, "docs", "diagrams", stem + ext)
            assert os.path.exists(path), (
                f"missing {stem + ext} -- run python scripts/export_diagrams.py")
            assert os.path.getsize(path) > 2000, f"{stem + ext} looks truncated"

    index = os.path.join(root, "docs", "diagrams", "README.md")
    assert os.path.exists(index)
    body = open(index, encoding="utf-8").read()
    for stem, _doc, _idx, _t in DIAGRAMS:
        assert stem in body, f"{stem} missing from the diagram index"


@pytest.mark.parametrize("mod_name,filename", NOTEBOOKS)
def test_training_notebooks_never_hard_require_a_gpu(mod_name, filename):
    """The no-GPU contract: no cell may call preflight(require_gpu=True).

    A notebook that refuses to open without a GPU breaks the replay path, and
    the replay path is what lets a participant with no Colab quota follow the
    whole day.
    """
    mod = importlib.import_module(f"nbsrc.{mod_name}")
    for i, (kind, src) in enumerate(mod.CELLS):
        if kind == "code":
            assert "require_gpu=True" not in src, (
                f"{mod_name} cell {i} hard-requires a GPU")
