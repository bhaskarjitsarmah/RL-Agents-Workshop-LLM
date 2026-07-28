"""Generate NB0-NB8 as .ipynb files from the cell definitions in nbsrc/.

    python build_notebooks.py            # rebuild all
    python build_notebooks.py 3 5        # rebuild only NB3 and NB5

Notebooks are NEVER hand-edited: prose lives in `nbsrc/nb*.py`, so changes show
up as readable diffs and outputs are never committed. Regenerating CLEARS all
outputs; re-run the notebooks to repopulate them.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nbsrc import NB_DIR, build  # noqa: E402

NOTEBOOKS = [
    ("nb0", "NB0_two_agents_one_scoreboard.ipynb"),
    ("nb1", "NB1_the_mdp.ipynb"),
    ("nb2", "NB2_data_and_star_warm_start.ipynb"),
    ("nb3", "NB3_grpo.ipynb"),
    ("nb4", "NB4_multi_turn_rl.ipynb"),
    ("nb5", "NB5_openpipe_art.ipynb"),
    ("nb6", "NB6_reward_hacking_and_safety.ipynb"),
    ("nb7", "NB7_deployment.ipynb"),
    ("nb8", "NB8_capstone_weights_vs_harness.ipynb"),
]


def check_syntax(mod_name: str, cells: list) -> list[str]:
    """Every code cell must parse as Python.

    Cheap, and it catches the failure mode this generator is most prone to: a
    broken f-string or an unbalanced quote inside a triple-quoted cell, which
    would otherwise surface as a SyntaxError in front of a room of participants.
    """
    errors = []
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append(f"{mod_name} cell {i}: {e.msg} (line {e.lineno})\n"
                          f"    {(src.splitlines() or [''])[max(e.lineno - 1, 0)][:100]}")
    return errors


def main(argv: list[str]) -> int:
    wanted = {f"nb{a}" for a in argv} if argv else None
    os.makedirs(NB_DIR, exist_ok=True)

    all_errors: list[str] = []
    built = 0
    for mod_name, filename in NOTEBOOKS:
        if wanted and mod_name not in wanted:
            continue
        try:
            mod = importlib.import_module(f"nbsrc.{mod_name}")
        except ModuleNotFoundError:
            print(f"  {mod_name:<5} -- not written yet, skipping")
            continue
        cells = mod.CELLS
        errs = check_syntax(mod_name, cells)
        all_errors.extend(errs)
        path = build(os.path.join(NB_DIR, filename), cells)
        n_code = sum(1 for k, _ in cells if k == "code")
        print(f"  {mod_name:<5} -> {filename:<46} "
              f"{len(cells):>2} cells ({n_code} code)")
        built += 1

    if all_errors:
        print("\nSYNTAX ERRORS in generated code cells:")
        for e in all_errors:
            print("  " + e)
        return 1
    print(f"\n{built} notebook(s) written to {NB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
