"""Build data/test_perturbed.json -- the frozen robustness suite for NB6.

    python scripts/make_perturbations.py        # deterministic, no GPU, no keys

Four perturbations of each of the 16 held-out questions. The **gold SQL never
changes**: a paraphrase asks the same question, so the same query answers it.
That is what makes this a robustness test rather than a different eval set.

    paraphrase   same question, different words
    typo         realistic keyboard slips (adjacent keys, transposition)
    distractor   an irrelevant but plausible clause bolted on
    rename       schema entities referred to by synonyms ("clients", "purchases")

Why constructed rather than LLM-generated
-----------------------------------------
The plan originally called for gpt-4o-mini paraphrases plus a hand check. Rule-
based construction is better here for one reason: **determinism**. NB6 compares
robustness *across checkpoints*, and a perturbation set that regenerated per run
would make that comparison meaningless -- you could not tell a brittle model from
an unlucky sample. This runs offline, produces byte-identical output every time,
and the result is checked in.

The trade-off is honest and worth stating in the notebook: these paraphrases are
less varied than a model's would be. They are a floor on robustness, not a
ceiling.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.config import DATA_DIR  # noqa: E402
from llm_utils.db import build_db  # noqa: E402
from llm_utils.sqlio import safe_run_sql  # noqa: E402
from llm_utils.tasks import TASKS  # noqa: E402

# --- paraphrase: whole-phrase rewrites, applied in order --------------------
PARAPHRASE = [
    (r"^List the names of all ", "Give me every "),
    (r"^List the names of ", "Give me the names of "),
    (r"^Show all ", "Return all "),
    (r"^Show the ", "Return the "),
    (r"^Show ", "Return "),
    (r"^List ", "Return "),
    (r"^How many ", "What is the count of "),
    (r"^What is the ", "Tell me the "),
    (r"^Which ", "What "),
    (r"^Find ", "Locate "),
    (r"^Return only the ", "Give just the "),
    (r"\bReturn the ", "Give the "),
    (r"\bproducts\b", "items"),
    (r"\bcustomers\b", "clients"),
]

# --- rename: schema synonyms (the model must still map them to the schema) --
RENAME = [
    (r"\bcustomers?\b", "buyers"),
    (r"\bproducts?\b", "SKUs"),
    (r"\borders?\b", "purchases"),
    (r"\bcategory\b", "product group"),
    (r"\bsegment\b", "tier"),
]

DISTRACTORS = [
    " Ignore any rows that do not exist.",
    " Note that the database is small.",
    " Please be careful with the join.",
    " This is for a quarterly report.",
    " The answer should be computed from the tables above.",
]

_ADJACENT = {"a": "s", "e": "r", "i": "o", "o": "p", "u": "y", "n": "m",
             "t": "y", "s": "d", "r": "t", "l": "k", "c": "v", "m": "n"}


def paraphrase(q: str) -> str:
    out = q
    for pat, rep in PARAPHRASE:
        out = re.sub(pat, rep, out, count=1)
    return out[0].upper() + out[1:] if out else out


def typo(q: str, rng: random.Random) -> str:
    """Two realistic slips: one adjacent-key substitution, one transposition.

    Deliberately confined to words longer than four characters and never to a
    quoted literal -- corrupting 'Mumbai' would change the question's meaning
    rather than testing robustness to sloppy typing.
    """
    words = q.split()
    idxs = [i for i, w in enumerate(words)
            if len(w.strip(".,?'")) > 4 and "'" not in w and w[0].islower()]
    if not idxs:
        return q
    for _ in range(min(2, len(idxs))):
        i = rng.choice(idxs)
        w = words[i]
        j = rng.randrange(1, len(w) - 1)
        if w[j].lower() in _ADJACENT and rng.random() < 0.5:
            words[i] = w[:j] + _ADJACENT[w[j].lower()] + w[j + 1:]
        else:
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        idxs = [k for k in idxs if k != i]
        if not idxs:
            break
    return " ".join(words)


def distractor(q: str, rng: random.Random) -> str:
    return q.rstrip() + rng.choice(DISTRACTORS)


def rename(q: str) -> str:
    out = q
    for pat, rep in RENAME:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def main() -> int:
    build_db()
    rng = random.Random(20260808)
    test = [t for t in TASKS if t["split"] == "test"]

    data: dict[str, list[dict]] = {"clean": [], "paraphrase": [], "typo": [],
                                   "distractor": [], "rename": []}
    for t in test:
        q, gold = t["question"], t["gold"]
        data["clean"].append({"id": t["id"], "question": q, "gold": gold})
        data["paraphrase"].append({"id": t["id"], "question": paraphrase(q),
                                   "gold": gold})
        data["typo"].append({"id": t["id"], "question": typo(q, rng), "gold": gold})
        data["distractor"].append({"id": t["id"], "question": distractor(q, rng),
                                   "gold": gold})
        data["rename"].append({"id": t["id"], "question": rename(q), "gold": gold})

    # The gold must still be the gold: every one has to execute, and the
    # perturbation must actually have changed the question.
    unchanged = []
    for kind, items in data.items():
        for it in items:
            rows, err = safe_run_sql(it["gold"])
            assert err is None and rows, f"{kind} #{it['id']}: gold broke"
        if kind != "clean":
            same = [it["id"] for it, c in zip(items, data["clean"])
                    if it["question"] == c["question"]]
            if same:
                unchanged.append((kind, same))

    path = os.path.join(DATA_DIR, "test_perturbed.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    print(f"wrote {path}")
    for kind, items in data.items():
        print(f"  {kind:<12} {len(items)} questions")
    if unchanged:
        print("\nWARNING -- these perturbations left the question identical, so "
              "they test nothing:")
        for kind, ids in unchanged:
            print(f"  {kind}: task ids {ids}")
    print("\nexamples:")
    for kind in ("paraphrase", "typo", "distractor", "rename"):
        print(f"\n  [{kind}]")
        print(f"    clean: {data['clean'][0]['question']}")
        print(f"    ->     {data[kind][0]['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
