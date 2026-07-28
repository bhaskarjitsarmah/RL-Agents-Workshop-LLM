"""Execute a notebook's code cells in one namespace, without Jupyter.

    python scripts/run_notebook_cells.py notebooks/NB0_two_agents_one_scoreboard.ipynb

This is the offline dry-run harness (plan Phase 12b): it proves every notebook
runs top-to-bottom on a machine with **no GPU, no API keys, and no network**,
which is the hard rule this repo commits to. Using a plain interpreter rather
than a Jupyter kernel keeps it usable in CI and avoids picking up whichever
kernel happens to be registered on the machine.

Charts are rendered to a headless Agg backend and discarded; we care that they
*render*, not what they look like.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(path: str, stop_on_error: bool = False) -> int:
    nb = json.load(open(path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    # Notebooks assume they are run from notebooks/, like a participant would.
    os.chdir(os.path.join(REPO_ROOT, "notebooks"))
    ns: dict = {"__name__": "__main__"}
    failures = 0

    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)   # noqa: S102
        except Exception:
            failures += 1
            print(f"\n--- CELL {i} FAILED " + "-" * 52)
            print(src[:400])
            print("---")
            traceback.print_exc(limit=3)
            if stop_on_error:
                break
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")

    name = os.path.basename(path)
    if failures:
        print(f"\n{name}: {failures}/{len(cells)} code cells FAILED")
    else:
        print(f"\n{name}: all {len(cells)} code cells ran clean (offline mode)")
    return 1 if failures else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    stop = "--stop" in sys.argv
    rc = 0
    for p in args:
        rc |= run(os.path.join(REPO_ROOT, p) if not os.path.isabs(p) else p, stop)
    raise SystemExit(rc)
