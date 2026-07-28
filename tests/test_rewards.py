"""Reward and MDP guarantees.

Two properties here are load-bearing for the whole workshop:

1. **Separation.** No wrong answer may score as well as a right one. If shaping
   ever inverts that, the policy will find the exploit and NB3's curves become
   a story about a bug.
2. **The hackable reward must actually be hackable.** NB6's entire lesson is a
   scissors chart -- proxy reward rising while true accuracy falls. If
   `r_hackable_rowcount` cannot be gamed, that notebook has no demo, and we
   would rather learn that here than on the day.

    pytest tests/test_rewards.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.gen_tasks import read_jsonl  # noqa: E402
from llm_utils.config import DATA_DIR  # noqa: E402
from llm_utils.metrics import (  # noqa: E402
    mcnemar, paired_bootstrap, report_number, wilson_ci, zero_advantage_fraction,
)
from llm_utils.rewards import (  # noqa: E402
    DEFAULT_WEIGHTS, composite_reward, detect_reward_hacks, make_trl_reward_fns,
    r_exec_match, r_executes, r_format, r_hackable_rowcount, r_nonempty,
    r_safety, reward_bounds,
)
from llm_utils.rollout import (  # noqa: E402
    SQLEnv, Trajectory, advantages, rollout_group, rollout_multi_turn,
    rollout_single_turn, summarize_group,
)
from llm_utils.sqlio import fast_score_sql, is_write_statement  # noqa: E402
from llm_utils.tasks import TASKS  # noqa: E402

GOLD = "SELECT name FROM customers WHERE city='Mumbai';"
WRAPPED = f"```sql\n{GOLD}\n```"


def fenced(sql: str) -> str:
    return f"```sql\n{sql}\n```"


# --- Components -----------------------------------------------------------

def test_format_reward_targets_the_vendored_parser():
    """The format reward must reward what `extract_sql` actually parses.

    Training a policy to emit `<sql>...</sql>` would look perfect in training
    and score zero through the real harness.
    """
    assert r_format(WRAPPED) == 1.0
    assert r_format("```\nSELECT 1;\n```") == 1.0
    assert r_format("<sql>SELECT 1;</sql>") == 0.0
    assert r_format("SELECT 1;") == 0.0
    assert r_format("") == 0.0
    # Chatty but correctly shaped: partial credit.
    assert r_format("Sure, here is the query you asked for and some more words "
                    "of padding:\n" + WRAPPED) == 0.5
    # Two fences: `extract_sql` silently takes the first, so hedging must not pay.
    assert r_format(WRAPPED + "\nor maybe\n```sql\nSELECT 1;\n```") == 0.0


def test_exec_match_is_the_real_reward():
    assert r_exec_match(GOLD, GOLD) == 1.0
    assert r_exec_match("SELECT name FROM customers WHERE city='Delhi';", GOLD) == 0.0
    assert r_exec_match("SELCT broken FROM", GOLD) == 0.0


def test_executes_and_nonempty():
    assert r_executes(GOLD) == 1.0
    assert r_executes("SELCT broken FROM") == 0.0
    assert r_nonempty(GOLD) == 1.0
    assert r_nonempty("SELECT name FROM customers WHERE city='Atlantis';") == 0.0


def test_safety_component():
    assert r_safety(GOLD) == 1.0
    assert r_safety("DROP TABLE customers;") == 0.0
    assert is_write_statement("DROP TABLE customers")
    assert is_write_statement("SELECT 1; DELETE FROM orders")
    assert not is_write_statement(GOLD)
    # A typo is a mistake, not an attack -- it must NOT be treated as a write,
    # or the policy loses the formatting gradient that gets it off the floor.
    assert not is_write_statement("SELCT nope FROM")


# --- The separation property ---------------------------------------------

def test_reward_bounds_are_separated():
    b = reward_bounds()
    assert b["separated"], b
    assert b["margin"] > 0.5, "separation margin is uncomfortably thin"


def test_gold_scores_max_and_garbage_scores_low():
    r_gold, parts = composite_reward(WRAPPED, GOLD)
    assert parts["correct"] == 1.0
    assert r_gold == pytest.approx(1.30, abs=1e-6)
    r_bad, _ = composite_reward(fenced("SELECT name FROM customers WHERE city='Delhi';"), GOLD)
    assert r_bad <= 0.30
    assert composite_reward("just some prose", GOLD)[0] == 0.0


def test_no_wrong_answer_outscores_a_right_one_on_real_tasks():
    """The inequality that makes the reward trustworthy, over real data.

    Worst case for a correct answer (unfenced, after 3 turns) must still beat
    the best case for an incorrect one (perfectly fenced, runs, returns rows).
    """
    tasks = read_jsonl(os.path.join(DATA_DIR, "tasks_train_gen.jsonl"))[:150]
    worst_correct, best_wrong = 9e9, -9e9
    plausible_wrong = "SELECT name FROM customers;"
    for t in tasks:
        r, _ = composite_reward(t["gold"], t["gold"], n_llm_calls=3,
                                already_sql=True)
        worst_correct = min(worst_correct, r)
        if not fast_score_sql(plausible_wrong, t["gold"]):
            r2, _ = composite_reward(fenced(plausible_wrong), t["gold"])
            best_wrong = max(best_wrong, r2)
    assert worst_correct > best_wrong, (
        f"reward inverted: worst correct {worst_correct} <= best wrong {best_wrong}")


def test_write_statements_are_clamped():
    r, parts = composite_reward(fenced("DROP TABLE customers;"), GOLD)
    assert r == 0.0 and parts.get("unsafe") == 1.0


def test_efficiency_penalty_costs_turns():
    r1, _ = composite_reward(WRAPPED, GOLD, n_llm_calls=1)
    r3, _ = composite_reward(WRAPPED, GOLD, n_llm_calls=3)
    assert r3 < r1
    assert r1 - r3 == pytest.approx(2 * DEFAULT_WEIGHTS["efficiency"], abs=1e-6)


# --- TRL glue -------------------------------------------------------------

def test_trl_reward_fns_shape_and_gold_requirement():
    fns = make_trl_reward_fns()
    assert [f.__name__ for f in fns] == ["exec_match", "format", "executes", "nonempty"]
    completions = [WRAPPED, fenced("SELECT 1;")]
    golds = [GOLD, GOLD]
    for f in fns:
        out = f(completions, gold=golds)
        assert len(out) == 2 and all(isinstance(x, float) for x in out)
    assert fns[0](completions, gold=golds) == [DEFAULT_WEIGHTS["exec_match"], 0.0]


def test_trl_reward_fn_fails_loudly_without_gold():
    """`remove_unused_columns=True` silently drops the gold column; without this
    guard the run would train against nothing and merely look disappointing."""
    with pytest.raises(ValueError, match="remove_unused_columns"):
        make_trl_reward_fns()[0]([WRAPPED])


def test_trl_reward_fns_accept_conversational_completions():
    fns = make_trl_reward_fns()
    conv = [[{"role": "assistant", "content": WRAPPED}]]
    assert fns[0](conv, gold=[GOLD]) == [DEFAULT_WEIGHTS["exec_match"]]


# --- Module 5: the proxy must genuinely be gameable ------------------------

def test_hackable_reward_is_actually_hackable():
    """If this fails, NB6 has no demo.

    The exploit: keep a degenerate query that answers nothing, and alias the
    question's own nouns as column names to farm the overlap term. It must be
    able to MATCH OR BEAT the true gold under the proxy while scoring zero on
    real correctness -- that gap is the scissors chart.
    """
    tasks = read_jsonl(os.path.join(DATA_DIR, "tasks_train_gen.jsonl"))[:60]
    beats = 0
    for t in tasks:
        q = t["question"]
        nouns = [w.strip("?.,'") for w in q.lower().split() if len(w) > 4][:3]
        stuffed = fenced("SELECT " + ", ".join(f"order_id AS {w}" for w in nouns)
                         + " FROM orders LIMIT 5;") if nouns \
            else fenced("SELECT * FROM orders LIMIT 5;")
        if r_hackable_rowcount(stuffed, q) >= r_hackable_rowcount(fenced(t["gold"]), q):
            beats += 1
    assert beats >= len(tasks) // 4, (
        f"the 'hackable' proxy was only gamed on {beats}/{len(tasks)} tasks -- "
        "NB6's reward-hacking demo needs a real exploit, not a theoretical one")


def test_hackable_reward_never_looks_at_the_gold():
    """The defect, stated as a test.

    A proxy that could see the gold would score the true answer perfectly and be
    unhackable by construction -- which is precisely the bug an earlier draft
    had. The proxy takes the QUESTION.
    """
    wrong = fenced("SELECT * FROM orders LIMIT 5;")
    question = "List the names of all customers in Mumbai."
    assert r_hackable_rowcount(wrong, question) > 0.5
    assert r_exec_match("SELECT * FROM orders LIMIT 5;", GOLD) == 0.0
    # Passing the gold text as the "question" must not be a back door to a
    # perfect score for the gold: the term is generic word overlap, nothing more.
    import inspect
    assert "question" in inspect.signature(r_hackable_rowcount).parameters


def test_detect_reward_hacks_fires_on_degenerates_and_is_quiet_on_honest_output():
    hacked = [fenced("SELECT * FROM orders LIMIT 5;")] * 10
    rep = detect_reward_hacks(hacked)
    assert rep["suspicious"]
    assert "answer_collapse" in rep["flags"]
    assert "select_star" in rep["flags"]

    tasks = read_jsonl(os.path.join(DATA_DIR, "tasks_train_gen.jsonl"))[:10]
    honest = [fenced(t["gold"]) for t in tasks]
    rep2 = detect_reward_hacks(honest)
    assert not rep2["suspicious"], rep2


def test_detect_reward_hacks_catches_hardcoded_constants():
    rep = detect_reward_hacks([fenced("SELECT 'Mumbai';")] * 5)
    assert "no_from_clause" in rep["flags"]


# --- The MDP --------------------------------------------------------------

def scripted_policy(*replies):
    """A deterministic stand-in for a real policy, so the env is testable on CPU."""
    queue = list(replies)

    def policy(messages, n=1, temperature=0.0, max_new_tokens=256, **kw):
        out = queue.pop(0) if queue else ""
        return [out] * n

    return policy


def test_env_reset_and_direct_answer():
    task = next(t for t in TASKS if t["id"] == 1)
    env = SQLEnv()
    obs = env.reset(task)
    assert obs[0]["role"] == "system" and task["question"] in obs[1]["content"]
    _, _, done, info = env.step(fenced(task["gold"]))
    assert done and info["reason"] == "submitted" and info["sql"] == task["gold"]


def test_env_tool_call_then_answer():
    task = next(t for t in TASKS if t["id"] == 1)
    env = SQLEnv()
    env.reset(task)
    _, _, done, info = env.step('<tool>{"name": "list_tables", "args": {}}</tool>')
    assert not done and info["tool_calls"] == 1
    assert "customers" in env.steps[-1].content
    _, _, done, info = env.step(fenced(task["gold"]))
    assert done and info["sql"] == task["gold"]


def test_env_refuses_writes_through_the_tool():
    env = SQLEnv()
    env.reset(TASKS[0])
    env.step('<tool>{"name": "run_query", "args": {"sql": "DROP TABLE customers"}}</tool>')
    assert "Refused" in env.steps[-1].content


def test_env_gives_a_malformed_action_one_warning_before_terminating():
    """A parser mismatch must not masquerade as reward collapse."""
    env = SQLEnv(max_turns=3)
    env.reset(TASKS[0])
    _, _, done, info = env.step("I am not going to follow the format.")
    assert not done and info["reason"] == "malformed"
    assert "```sql" in env.steps[-1].content


def test_env_terminates_at_max_turns():
    env = SQLEnv(max_turns=2)
    env.reset(TASKS[0])
    env.step('<tool>{"name": "list_tables", "args": {}}</tool>')
    _, _, done, _ = env.step('<tool>{"name": "list_tables", "args": {}}</tool>')
    assert done


def test_env_handles_bad_json_tool_calls():
    env = SQLEnv()
    env.reset(TASKS[0])
    env.step("<tool>{not json at all}</tool>")
    assert "not valid JSON" in env.steps[-1].content


def test_single_turn_rollout_scores_a_correct_answer():
    task = next(t for t in TASKS if t["id"] == 1)
    traj = rollout_single_turn(scripted_policy(fenced(task["gold"])), task)
    assert traj.correct and traj.reward > 1.0 and traj.n_llm_calls == 1
    assert traj.final_sql == task["gold"]


def test_multi_turn_rollout_masks_tool_output_from_training():
    task = next(t for t in TASKS if t["id"] == 1)
    policy = scripted_policy('<tool>{"name": "list_tables", "args": {}}</tool>',
                             fenced(task["gold"]))
    traj = rollout_multi_turn(policy, task, max_turns=4)
    assert traj.correct and traj.n_tool_calls == 1 and traj.n_llm_calls == 2
    roles = [traj.steps[i].role for i in traj.assistant_spans()]
    assert roles == ["assistant", "assistant"], "only assistant turns are trainable"
    assert any(s.role == "tool" for s in traj.steps)


def test_multi_turn_terminates_on_every_original_task():
    """A policy that never answers must still terminate, on all 40 tasks."""
    for task in TASKS:
        traj = rollout_multi_turn(scripted_policy(), task, max_turns=3)
        assert traj.n_llm_calls <= 3
        assert traj.reward <= 0.30


# --- Advantages -----------------------------------------------------------

def _traj(reward: float) -> Trajectory:
    return Trajectory(task={"id": "x", "level": "easy"}, reward=reward,
                      correct=reward > 0.5)


def test_advantages_are_zero_for_a_flat_group():
    """The diagnostic that matters most in GRPO, asserted rather than assumed."""
    assert advantages([_traj(1.0)] * 8) == [0.0] * 8
    assert advantages([_traj(0.0)] * 8) == [0.0] * 8


def test_advantages_centre_and_scale():
    adv = advantages([_traj(1.0), _traj(0.0)])
    assert adv[0] > 0 > adv[1]
    assert abs(sum(adv)) < 1e-6, "advantages must be mean-zero within the group"


def test_unscaled_advantages_are_dr_grpo():
    adv = advantages([_traj(1.0), _traj(0.0)], scale=False)
    assert adv == [0.5, -0.5]


def test_group_summary_flags_zero_advantage():
    s = summarize_group([_traj(1.0)] * 8)
    assert s["zero_advantage"] and s["pass_rate"] == 1.0
    s2 = summarize_group([_traj(1.0), _traj(0.0)] * 4)
    assert not s2["zero_advantage"] and s2["pass_rate"] == 0.5


def test_rollout_group_returns_g_trajectories():
    task = next(t for t in TASKS if t["id"] == 1)
    g = rollout_group(scripted_policy(fenced(task["gold"])), task, G=4)
    assert len(g) == 4


def test_zero_advantage_fraction_metric():
    assert zero_advantage_fraction([[1, 1, 1], [0, 1, 0]]) == 0.5


# --- Metrics --------------------------------------------------------------

def test_wilson_matches_published_values():
    lo, hi = wilson_ci(12, 16)
    assert round(lo, 3) == 0.505 and round(hi, 3) == 0.898
    lo, hi = wilson_ci(13, 16)
    assert round(lo, 3) == 0.570 and round(hi, 3) == 0.934


def test_wilson_does_not_claim_certainty_at_the_boundary():
    lo, hi = wilson_ci(16, 16)
    assert hi == 1.0 and lo < 0.85, "16/16 must not imply p=1.0"


def test_report_number_always_carries_an_interval():
    s = report_number((12, 16))
    assert "0.750" in s and "[0.505, 0.898]" in s and "(12/16)" in s


def test_paired_tests_call_a_one_task_difference_inconclusive():
    """The pedagogical point of NB0, as a test."""
    a = [{"id": i, "correct": i < 12} for i in range(16)]
    b = [{"id": i, "correct": i < 13} for i in range(16)]
    pb = paired_bootstrap(a, b)
    assert pb["delta"] == pytest.approx(1 / 16)
    assert pb["p_two_sided"] > 0.10, "a single task must not look significant"
    assert mcnemar(a, b)["n_discordant"] == 1


def test_paired_tests_reject_unaligned_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap([{"id": 1, "correct": True}],
                         [{"id": 1, "correct": True}, {"id": 2, "correct": False}])
