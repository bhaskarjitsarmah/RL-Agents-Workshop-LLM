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
