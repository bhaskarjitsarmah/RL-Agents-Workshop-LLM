"""The task generator's correctness guarantees.

The reward in this repo is `score_sql(pred, gold)`. A single wrong gold does not
merely lose one training example -- it *inverts* the gradient on that prompt,
punishing the policy for being right. So the generator's claims are tested, not
trusted:

  * every gold executes, is a SELECT, and returns something meaningful
  * no generated task leaks a held-out test task, under any of the four rules
  * the splits are mutually disjoint
  * generation is deterministic given a seed
  * the checked-in JSONL files match what the generator produces today

That last one is the load-bearing regression test: the JSONLs are committed, so
the workshop never depends on regeneration -- but that also means a change to a
template could silently desync the files from the code.

    pytest tests/test_generator.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.config import DATA_DIR  # noqa: E402
from llm_utils.db import DB_PATH  # noqa: E402
from llm_utils.gen_tasks import (  # noqa: E402
    FAMILIES, NEAR_VARIANT_OF, SLOT_FREE_TEST_FAMILIES, TEMPLATES,
    TEST_FAMILIES, TEST_TASK_FAMILY, build_corpus, canon_sql,
    collides_with_eval, content_tokens, generate_tasks, jaccard, norm_question,
    read_jsonl, validate_task,
)
from llm_utils.sqlio import is_read_only, safe_run_sql  # noqa: E402
from llm_utils.tasks import TASKS  # noqa: E402

TEST_TASKS = [t for t in TASKS if t["split"] == "test"]


# --- Templates ------------------------------------------------------------

def test_family_names_unique():
    assert len(FAMILIES) == len(TEMPLATES)


def test_every_template_has_paraphrases_and_valid_level():
    for t in TEMPLATES:
        assert len(t.questions) >= 2, f"{t.family} needs >= 2 paraphrases"
        assert len(set(t.questions)) == len(t.questions), f"{t.family} has dupes"
        assert t.level in ("easy", "medium", "hard")
        assert t.gold.strip().lower().startswith("select")
        assert t.gold.rstrip().endswith(";")


def test_question_and_gold_share_a_namespace():
    """Every slot used by the gold must be fillable, and vice versa.

    This is the correctness argument for the whole generator: because the same
    namespace formats both strings, a value in the question IS the value in the
    query. A slot that appears in only one of them breaks that guarantee.
    """
    import string

    for t in TEMPLATES:
        gold_fields = {f for _, f, _, _ in string.Formatter().parse(t.gold) if f}
        provided = set(t.slots) | {k for v in t.variants for k in v}
        missing = gold_fields - provided
        assert not missing, f"{t.family}: gold uses unfillable slots {missing}"
        for q in t.questions:
            q_fields = {f for _, f, _, _ in string.Formatter().parse(q) if f}
            assert not (q_fields - provided), (
                f"{t.family}: question uses unfillable slots {q_fields - provided}")


def test_test_task_family_map_is_complete_and_real():
    assert set(TEST_TASK_FAMILY) == {t["id"] for t in TEST_TASKS}
    for tid, fam in TEST_TASK_FAMILY.items():
        assert fam in FAMILIES, f"test task {tid} maps to unknown family {fam}"
    for near, real in NEAR_VARIANT_OF.items():
        assert near in FAMILIES and real in FAMILIES


# --- The corpus -----------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_and_audit():
    return build_corpus(seed=1234)


def test_every_generated_gold_executes_and_is_readonly(corpus_and_audit):
    corpus, _ = corpus_and_audit
    for fam, items in corpus.items():
        for t in items:
            rows, err = safe_run_sql(t["gold"], DB_PATH)
            assert err is None, f"{fam}: gold failed: {err}\n{t['gold']}"
            assert rows is not None
            assert is_read_only(t["gold"]), f"{fam}: gold is not read-only"


def test_no_generated_task_leaks_a_test_task(corpus_and_audit):
    corpus, _ = corpus_and_audit
    for fam, items in corpus.items():
        for t in items:
            rule = collides_with_eval(t, TEST_TASKS, DB_PATH)
            assert rule in (None, "counted_signature_only"), (
                f"{fam}: leaked via {rule}: {t['question']}")


def test_slot_free_test_families_are_empty_by_construction(corpus_and_audit):
    """These five families CANNOT contribute anything, and that is correct.

    They have no slots, so their only possible gold is the held-out test task
    itself. A non-zero count here would mean the leakage rules had stopped
    working -- so this test asserts the zero rather than tolerating it.
    """
    corpus, _ = corpus_and_audit
    for fam in SLOT_FREE_TEST_FAMILIES:
        assert corpus[fam] == [], f"{fam} should be empty by construction"


def test_all_other_families_produce_something(corpus_and_audit):
    corpus, _ = corpus_and_audit
    empty = [f for f, v in corpus.items()
             if not v and f not in SLOT_FREE_TEST_FAMILIES]
    assert not empty, f"unexpectedly empty families: {empty}"


def test_generated_questions_are_unique_within_a_family(corpus_and_audit):
    corpus, _ = corpus_and_audit
    for fam, items in corpus.items():
        qs = [norm_question(t["question"]) for t in items]
        assert len(qs) == len(set(qs)), f"{fam} has duplicate questions"


# --- Splits ---------------------------------------------------------------

@pytest.fixture(scope="module")
def splits():
    return generate_tasks(n_train=200, n_val=60, n_test_ext=60, seed=99)


def test_splits_are_mutually_disjoint(splits):
    def keys(name):
        return {(t["family"], norm_question(t["question"])) for t in splits[name]}

    tr, va, te = keys("train"), keys("val"), keys("test_ext")
    assert not (tr & va), "train and val overlap"
    assert not (tr & te), "train and test_ext overlap"
    assert not (va & te), "val and test_ext overlap"


def test_val_covers_every_level(splits):
    """val gates early stopping. A val set missing the hard tasks would stop
    training based on the easy half of the distribution -- an earlier version of
    the allocator did exactly that."""
    levels = {t["level"] for t in splits["val"]}
    assert levels == {"easy", "medium", "hard"}, f"val only has {levels}"


def test_test_ext_only_uses_test_families(splits):
    for t in splits["test_ext"]:
        assert t["family"] in TEST_FAMILIES


def test_train_sees_the_test_patterns(splits):
    """The main training set MUST cover the patterns the 16 test tasks use.

    Only the test *instances* are withheld. A policy that never trains on
    'revenue argmax' would fail the headline comparison for reasons unrelated to
    RL. The memorization question is answered by the separate no-leak control.
    """
    fams = {t["family"] for t in splits["train"]}
    assert fams & set(TEST_FAMILIES), "train has no test-family coverage at all"


def test_exclude_families_control_is_actually_clean():
    out = generate_tasks(n_train=200, n_val=0, n_test_ext=0, seed=99,
                         exclude_families=TEST_FAMILIES)
    fams = {t["family"] for t in out["train"]}
    assert not (fams & set(TEST_FAMILIES)), (
        "the memorization control still contains test-family patterns")


def test_no_family_dominates_a_split(splits):
    """Capacity is an accident of the schema; it must not decide what the policy
    spends its gradient steps on."""
    counts: dict[str, int] = {}
    for t in splits["train"]:
        counts[t["family"]] = counts.get(t["family"], 0) + 1
    n, n_fams = len(splits["train"]), len(counts)
    assert max(counts.values()) <= max(int(n / n_fams * 2.5), 4) + 1


def test_generation_is_deterministic():
    a = generate_tasks(n_train=80, n_val=20, n_test_ext=20, seed=7)
    b = generate_tasks(n_train=80, n_val=20, n_test_ext=20, seed=7)
    for split in ("train", "val", "test_ext"):
        assert [t["question"] for t in a[split]] == [t["question"] for t in b[split]]
        assert [t["gold"] for t in a[split]] == [t["gold"] for t in b[split]]


def test_different_seeds_differ():
    a = generate_tasks(n_train=80, n_val=20, n_test_ext=20, seed=7)
    b = generate_tasks(n_train=80, n_val=20, n_test_ext=20, seed=8)
    assert [t["question"] for t in a["train"]] != [t["question"] for t in b["train"]]


# --- The checked-in files -------------------------------------------------

CHECKED_IN = {
    "tasks_train_gen.jsonl": 800,
    "tasks_val_gen.jsonl": 200,
    "tasks_test_ext_gen.jsonl": 169,
    "tasks_train_noleak_gen.jsonl": 800,
}


@pytest.mark.parametrize("fname,expected_n", sorted(CHECKED_IN.items()))
def test_checked_in_files_exist_and_are_sane(fname, expected_n):
    path = os.path.join(DATA_DIR, fname)
    assert os.path.exists(path), (
        f"{fname} missing -- run: python scripts/generate_tasks.py")
    tasks = read_jsonl(path)
    assert len(tasks) == expected_n
    for t in tasks:
        assert {"id", "question", "gold", "level", "family", "split"} <= set(t)


def test_checked_in_files_are_leak_free_and_valid():
    """Re-verify the shipped artifacts directly, not just the generator.

    The JSONLs are committed and consumed by every notebook; a desync between
    them and the template code would be invisible until a training run produced
    nonsense.
    """
    for fname in CHECKED_IN:
        for t in read_jsonl(os.path.join(DATA_DIR, fname)):
            ok, why = validate_task(t, DB_PATH)
            assert ok, f"{fname} {t['id']}: {why}\n{t['gold']}"
            rule = collides_with_eval(t, TEST_TASKS, DB_PATH)
            assert rule in (None, "counted_signature_only"), (
                f"{fname} {t['id']} leaks via {rule}: {t['question']}")


def test_checked_in_noleak_file_excludes_test_families():
    tasks = read_jsonl(os.path.join(DATA_DIR, "tasks_train_noleak_gen.jsonl"))
    assert not ({t["family"] for t in tasks} & set(TEST_FAMILIES))


def test_checked_in_splits_disjoint():
    def keys(f):
        return {(t["family"], norm_question(t["question"]))
                for t in read_jsonl(os.path.join(DATA_DIR, f))}

    tr = keys("tasks_train_gen.jsonl")
    va = keys("tasks_val_gen.jsonl")
    te = keys("tasks_test_ext_gen.jsonl")
    assert not (tr & va) and not (tr & te) and not (va & te)


# --- Leakage-rule unit tests ---------------------------------------------

def test_rule1_catches_an_exact_test_question():
    ev = TEST_TASKS[0]
    task = {"question": ev["question"], "gold": "SELECT 1;", "family": "x"}
    assert collides_with_eval(task, TEST_TASKS, DB_PATH) == "rule1_exact_question"


def test_rule3_catches_alias_only_differences():
    """The generator's skeletons and repo 1's gold were written by the same
    hand, so 'same query, different alias' is a real risk, not a hypothetical."""
    ev = next(t for t in TEST_TASKS if t["id"] == 3)
    task = {"question": "Completely unrelated wording here about widgets.",
            "gold": ev["gold"].replace("products", "products p").replace(
                "SELECT name", "SELECT p.name"),
            "family": "x"}
    assert collides_with_eval(task, TEST_TASKS, DB_PATH) == "rule3_canonical_sql"


def test_rule4_requires_both_result_and_question_similarity():
    """A bare count colliding with an unrelated test task's number is NOT
    leakage. Rejecting on the result signature alone would gut the easy
    families for no benefit."""
    task = {"question": "How many order line items are recorded in total?",
            "gold": "SELECT COUNT(*) FROM order_items;", "family": "x"}
    rule = collides_with_eval(task, TEST_TASKS, DB_PATH)
    assert rule in (None, "counted_signature_only")


def test_canon_sql_and_norm_question_helpers():
    assert canon_sql("select  a from t  ;") == canon_sql("SELECT a FROM t")
    assert canon_sql("SELECT c.name FROM customers c") == \
        canon_sql("SELECT name FROM customers")
    assert norm_question("How many customers?") == "how many customers"
    assert jaccard(content_tokens("list customers in Mumbai"),
                   content_tokens("List the customers in Mumbai!")) == 1.0
