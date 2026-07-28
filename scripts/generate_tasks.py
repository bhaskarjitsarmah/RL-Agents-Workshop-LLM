"""Regenerate the task sets and the leakage audit.

    python scripts/generate_tasks.py

Writes (all checked in, so the workshop never depends on regeneration):
    data/tasks_train_gen.jsonl         800  main training set
    data/tasks_val_gen.jsonl           200  gating / early stopping
    data/tasks_test_ext_gen.jsonl      169  the 16 test patterns, high-power
    data/tasks_train_noleak_gen.jsonl  800  memorization control (NB2 ablation)
    data/leakage_audit.json                 rejections by rule -> a chart in NB2
    data/test_family_map.json               which family each test task uses

test_ext is capped at 55% of the test-family instance pool (TEST_EXT_MAX_SHARE),
so 169 is the honest maximum, not a truncation. The rest of that pool stays
available to train: withholding the test *instances* is required, withholding the
test *patterns* would cripple the headline comparison for reasons unrelated to RL.

Deterministic given --seed: re-running produces byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.config import DATA_DIR  # noqa: E402
from llm_utils.db import build_db  # noqa: E402
from llm_utils.gen_tasks import (  # noqa: E402
    NEAR_VARIANT_OF, SLOT_FREE_TEST_FAMILIES, TEMPLATES, TEST_FAMILIES,
    TEST_TASK_FAMILY, generate_tasks, split_report, write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test-ext", type=int, default=169)
    args = ap.parse_args()

    build_db()
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"{len(TEMPLATES)} templates across "
          f"{len({t.family for t in TEMPLATES})} families\n")

    out = generate_tasks(n_train=args.train, n_val=args.val,
                         n_test_ext=args.test_ext, seed=args.seed)
    audit = out["audit"]

    write_jsonl(out["train"], os.path.join(DATA_DIR, "tasks_train_gen.jsonl"))
    write_jsonl(out["val"], os.path.join(DATA_DIR, "tasks_val_gen.jsonl"))
    write_jsonl(out["test_ext"], os.path.join(DATA_DIR, "tasks_test_ext_gen.jsonl"))

    # The memorization control: same generator, but the test PATTERNS are held
    # out of training entirely. NB2 trains on both and reports the gap, so the
    # question "how much of your gain is template memorization?" is answered
    # before anyone in the room asks it.
    noleak = generate_tasks(n_train=args.train, n_val=0, n_test_ext=0,
                            seed=args.seed, exclude_families=TEST_FAMILIES)
    write_jsonl(noleak["train"],
                os.path.join(DATA_DIR, "tasks_train_noleak_gen.jsonl"))

    audit["noleak_train"] = len(noleak["train"])
    audit["near_variant_of"] = NEAR_VARIANT_OF
    audit["slot_free_test_families"] = list(SLOT_FREE_TEST_FAMILIES)
    audit["note_slot_free"] = (
        "These families have zero usable instances BY CONSTRUCTION: they are "
        "slot-free, so their only possible gold is the held-out test task "
        "itself. The leakage rules correctly reject every candidate. test_ext "
        "covers those five patterns via the near-variant families instead."
    )
    with open(os.path.join(DATA_DIR, "leakage_audit.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    with open(os.path.join(DATA_DIR, "test_family_map.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"test_task_family": {str(k): v for k, v in
                                        TEST_TASK_FAMILY.items()},
                   "test_families": list(TEST_FAMILIES),
                   "near_variant_of": NEAR_VARIANT_OF}, f, indent=2, sort_keys=True)

    for name, split in (("train", out["train"]), ("val", out["val"]),
                        ("test_ext", out["test_ext"]),
                        ("train_noleak", noleak["train"])):
        r = split_report(split)
        print(f"{name:<13} n={r['n']:<5} unique_q={r['unique_questions']:<5} "
              f"families={r['n_families']:<3} {r['by_level']}")

    print(f"\nleakage rejections: {audit['rejected_total']} "
          f"{audit['rejected_by_rule']}")
    print(f"signature-only collisions allowed (not leakage): "
          f"{audit['signature_only_collisions_allowed']}")
    short = {k: v for k, v in audit["shortfall"].items() if v > 0}
    print(f"shortfall vs requested: {short or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
