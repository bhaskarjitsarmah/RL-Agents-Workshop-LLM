"""Print a random sample of generated tasks for HUMAN review.

    python scripts/review_sample.py            # 25 tasks, one per family where possible
    python scripts/review_sample.py -n 40 --family revenue_by_group
    python scripts/review_sample.py --skeletons # one example per family (49 rows)

Why a human still has to look
-----------------------------
Everything else about this generator is machine-checked: the gold executes, it is
a SELECT, it returns rows, it does not leak a test task. None of that can tell
you whether the SQL answers the QUESTION.

`SELECT COUNT(*) FROM customers WHERE city='Pune'` is a perfectly valid,
non-leaking, non-empty gold for "List the names of customers in Pune" -- and it
is wrong. Only a person reading both halves catches that, and it only has to be
done once per skeleton, because slot substitution cannot break the correspondence.

So: read `--skeletons` once (49 rows), and spot-check a random sample after any
template change.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.db import DB_PATH, build_db  # noqa: E402
from llm_utils.gen_tasks import build_corpus  # noqa: E402
from llm_utils.sqlio import safe_run_sql  # noqa: E402


def show(i: int, t: dict, max_rows: int = 3) -> None:
    print(f"\n[{i}] {t['family']}  ({t['level']})")
    print("  Q: " + "\n     ".join(textwrap.wrap(t["question"], 92)))
    print("  G: " + "\n     ".join(textwrap.wrap(t["gold"], 92)))
    rows, err = safe_run_sql(t["gold"], DB_PATH)
    if err:
        print(f"  !! {err}")
        return
    preview = ", ".join(repr(r) for r in rows[:max_rows])
    more = f"  ... (+{len(rows) - max_rows} more)" if len(rows) > max_rows else ""
    print(f"  -> {len(rows)} row(s): {preview}{more}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default=None)
    ap.add_argument("--level", default=None, choices=["easy", "medium", "hard"])
    ap.add_argument("--skeletons", action="store_true",
                    help="one example per family -- the once-per-skeleton review")
    args = ap.parse_args()

    build_db()
    corpus, _ = build_corpus(seed=1234)
    rng = random.Random(args.seed)

    if args.skeletons:
        print("ONE EXAMPLE PER FAMILY -- read the Q and G of each and confirm the "
              "query answers the question.\n"
              "Slot substitution cannot break the correspondence, so this is a "
              "complete review of the generator.")
        i = 0
        for fam in sorted(corpus):
            items = corpus[fam]
            if not items:
                print(f"\n[--] {fam}: empty by construction (slot-free test pattern)")
                continue
            i += 1
            show(i, rng.choice(items))
        print(f"\n{i} skeletons shown.")
        return 0

    pool = [t for fam, items in corpus.items() for t in items
            if (args.family is None or fam == args.family)
            and (args.level is None or t["level"] == args.level)]
    if not pool:
        print("no tasks match that filter")
        return 1
    rng.shuffle(pool)
    for i, t in enumerate(pool[: args.n], 1):
        show(i, t)
    print(f"\n{min(args.n, len(pool))} of {len(pool)} matching tasks shown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
