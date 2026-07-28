"""Export the Mermaid diagrams in the docs to slide-ready SVG + PNG.

    python scripts/export_diagrams.py            # render all
    python scripts/export_diagrams.py --check    # verify the exports are current

Needs Node (the renderer is `npx @mermaid-js/mermaid-cli`, downloaded on first
use). Mirrors repo 1's `docs/diagrams/` so the two decks look like one workshop.

The sources are the ```mermaid blocks in README.md and ARCHITECTURE.md -- never
a separate copy. A diagram maintained in two places diverges, and the version on
the slide is always the stale one.

`--check` re-renders to a temp directory and compares, so a doc edit that was
never re-exported fails loudly instead of shipping a slide that contradicts the
README.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "diagrams")

#: (output stem, source doc, index of the mermaid block in that doc, title)
#: Numbering mirrors repo 1's 00-05 so the two decks interleave cleanly.
DIAGRAMS = [
    ("00-overview", "README.md", 0,
     "Big-picture overview (frozen harness + learning policy)"),
    ("01-frozen-vs-learning", "ARCHITECTURE.md", 0,
     "What is frozen and what learns"),
    ("02-notebook-journey", "ARCHITECTURE.md", 1,
     "The NB0 -> NB8 journey"),
    ("03-training-loop", "ARCHITECTURE.md", 2,
     "The GRPO training loop (with the zero-advantage branch)"),
    ("04-data-flow", "ARCHITECTURE.md", 3,
     "Data flow: generator -> four splits -> one scorer"),
    ("05-no-gpu-contract", "ARCHITECTURE.md", 4,
     "The no-GPU replay contract"),
]

FENCE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def blocks(doc: str) -> list[str]:
    with open(os.path.join(REPO_ROOT, doc), encoding="utf-8") as f:
        return [b.strip() for b in FENCE.findall(f.read())]


def have_node() -> bool:
    return shutil.which("npx") is not None


def render(src: str, out_path: str, png: bool) -> bool:
    """Render one .mmd to SVG or PNG. Returns True on success."""
    cmd = ["npx", "-y", "@mermaid-js/mermaid-cli",
           "-i", src, "-o", out_path, "-t", "neutral", "-b", "white"]
    if png:
        cmd += ["-s", "3"]          # 3x scale: crisp when pasted into slides
    r = subprocess.run(cmd, capture_output=True, text=True, shell=(os.name == "nt"))
    if r.returncode != 0 or not os.path.exists(out_path):
        print(f"    !! render failed: {(r.stderr or r.stdout or '').strip()[:300]}")
        return False
    return True


def export(out_dir: str, quiet: bool = False) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cache: dict[str, list[str]] = {}
    written = []

    with tempfile.TemporaryDirectory() as tmp:
        for stem, doc, idx, title in DIAGRAMS:
            if doc not in cache:
                cache[doc] = blocks(doc)
            found = cache[doc]
            if idx >= len(found):
                print(f"  !! {stem}: {doc} has only {len(found)} mermaid blocks "
                      f"(wanted #{idx}). Update DIAGRAMS in this script.")
                continue
            mmd = os.path.join(tmp, f"{stem}.mmd")
            with open(mmd, "w", encoding="utf-8", newline="\n") as f:
                f.write(found[idx] + "\n")

            for ext, is_png in ((".svg", False), (".png", True)):
                dst = os.path.join(out_dir, stem + ext)
                if render(mmd, dst, is_png):
                    written.append(dst)
                    if not quiet:
                        kb = os.path.getsize(dst) / 1024
                        print(f"  {stem + ext:<32} {kb:>7.1f} KB   {title}")
    return written


def check() -> int:
    """Fail if a doc edit was never re-exported."""
    with tempfile.TemporaryDirectory() as tmp:
        export(tmp, quiet=True)
        stale, missing = [], []
        for stem, _doc, _idx, _t in DIAGRAMS:
            for ext in (".svg", ".png"):
                a = os.path.join(tmp, stem + ext)
                b = os.path.join(OUT_DIR, stem + ext)
                if not os.path.exists(b):
                    missing.append(stem + ext)
                elif os.path.exists(a) and not filecmp.cmp(a, b, shallow=False):
                    stale.append(stem + ext)
    if missing:
        print("missing exports:", missing)
    if stale:
        print("stale exports (the docs changed):", stale)
    if missing or stale:
        print("\nre-run: python scripts/export_diagrams.py")
        return 1
    print("diagram exports are current")
    return 0


def write_index() -> None:
    lines = [
        "# Slide-ready diagrams",
        "",
        "Exported from the Mermaid sources in "
        "[`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) and "
        "[`../../README.md`](../../README.md) — never maintained separately, "
        "because a diagram kept in two places diverges and the slide always ends "
        "up with the stale copy.",
        "",
        "**SVG** is vector (crisp at any size, best for slides); **PNG** is 3x "
        "scale for pasting anywhere.",
        "",
        "```bash",
        "python scripts/export_diagrams.py          # re-export after editing a doc",
        "python scripts/export_diagrams.py --check  # verify they are current",
        "```",
        "",
        "| # | Diagram | Files |",
        "|---|---|---|",
    ]
    for stem, _doc, _idx, title in DIAGRAMS:
        lines.append(f"| {stem[:2]} | {title} | `{stem}.svg` / `.png` |")
    lines.append("")
    lines.append("---")
    for stem, _doc, _idx, title in DIAGRAMS:
        lines += ["", f"### {stem[:2]} · {title}", "",
                  f"![{title}]({stem}.png)"]
    lines.append("")
    with open(os.path.join(OUT_DIR, "README.md"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not have_node():
        print("npx not found. The Mermaid renderer needs Node.js:")
        print("  https://nodejs.org  (then re-run this script)")
        print("\nThe diagrams still render on GitHub from the fenced blocks in")
        print("README.md and ARCHITECTURE.md -- only the slide exports need Node.")
        return 1

    if args.check:
        return check()

    print(f"rendering {len(DIAGRAMS)} diagrams -> docs/diagrams/")
    written = export(OUT_DIR)
    write_index()
    print(f"\n{len(written)} files written, plus docs/diagrams/README.md")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
