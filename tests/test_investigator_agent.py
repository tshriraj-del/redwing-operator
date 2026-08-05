"""
Tests for the LLM investigator agent.

Why this file exists. The agent's whole claim is that it is GRADED rather than asserted, and
that claim rests on two things being true: it drives the same environment the reference
policies drive, and its scorecard is assembled from the environment's own verifiers. Both are
easy to break with a well-meaning refactor, and neither failure is visible in a transcript,
which will keep reading like a competent investigation either way.

The model is faked throughout. A test that needed an API key would not run, and the thing under
test is the loop and the scoring, not the model's judgement.

Runs under pytest or standalone (python3 tests/test_investigator_agent.py).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fraud_env                             # noqa: E402
from core import investigator_agent as A     # noqa: E402


def _case(fraud=True, high_card_signal=True, dispute=False, ring=False, risk="Low"):
    """A case shaped like case_file.assemble()'s output, carrying only what the env reads."""
    return {
        "case_id": "case_1", "transaction_id": "txn_1",
        "alert": {"ground_truth_label": "fraud" if fraud else "legit", "score": 0.8},
        "transaction": {"amount": 4200.0, "payment_rail": "Zelle"},
        "card_fraud_signals": ([{"severity": "high", "name": "cnp_velocity"}]
                               if high_card_signal else []),
        "dispute": ({"active": True, "first_party_fraud_risk": 0.8} if dispute else {}),
        "device_network": {"ring_flag": bool(ring)},
        "customer": {"risk_rating": risk},
        "timeline": [{"ts": "2026-01-01", "what": "login"}],
        "instrument": {"last4": "4242"},
    }


def _fake_model(script):
    """Replace the network call with a scripted sequence of replies."""
    it = iter(script)

    def _post(key, messages):
        try:
            return next(it)
        except StopIteration:
            return '{"reasoning": "done", "action": "clear_false_positive"}'
    return _post


def _with_model(monkeyish_script, key="test-key"):
    A._post = _fake_model(monkeyish_script)
    A._api_key = lambda: key


# ------------------------------------------------------------------ the honesty guarantees

def test_without_a_key_it_refuses_rather_than_substituting_a_policy():
    """THE one that matters for the claim. Silently falling back to a scripted policy and
    reporting it as an agent run would make every comparison in this module a fiction, and the
    transcript would look completely normal."""
    A._api_key = lambda: ""
    r = A.run_episode(_case())
    assert r["available"] is False
    assert "scorecard" not in r, "a refusal must not carry a score"
    assert "no ANTHROPIC_API_KEY" in r["reason"]


def test_the_scorecard_comes_from_the_environment_not_from_the_agent():
    """The agent must not be able to grade itself. Running the identical action sequence
    through fraud_env's own reference machinery must produce the identical numbers."""
    # The trajectory deliberately includes a REDUNDANT inspection and an irrelevant one, so
    # the process score is not 1.0. An earlier version used a clean trajectory whose true
    # process score happened to BE 1.0, and a mutation hardcoding 1.0 sailed straight through
    # the assertion: the test could not tell the env's answer from a constant.
    case = _case(fraud=True)
    seq = ["inspect_card_signals", "inspect_card_signals", "inspect_timeline"]
    _with_model([json.dumps({"reasoning": "r", "action": a}) for a in seq]
                + [json.dumps({"reasoning": "fraud", "action": "confirm_fraud"})])
    agent = A.run_episode(case)

    env_view = fraud_env.step(case, seq, "confirm_fraud")["info"]
    assert env_view["process"]["score"] != 1.0, (
        "the fixture no longer produces a non-trivial process score, so this test cannot "
        "distinguish the env's verdict from a hardcoded constant")
    assert agent["scorecard"]["outcome_reward"] == env_view["outcome_reward"]
    assert agent["scorecard"]["process_reward"] == env_view["process"]["score"]
    assert agent["scorecard"]["total_reward"] == env_view["total_reward"]
    assert agent["gold_disposition"] == env_view["gold_disposition"]


def test_deciding_blind_is_penalised_even_when_the_answer_is_right():
    """The failure the process verifier exists to catch. An agent that guesses 'confirm_fraud'
    on a fraud case gets the outcome right and has investigated nothing, and an evaluation that
    only checked the label would score it perfect."""
    case = _case(fraud=True)
    _with_model(['{"reasoning": "looks bad", "action": "confirm_fraud"}'])
    r = A.run_episode(case)
    assert r["scorecard"]["correct"] is True, "the label is right"
    assert r["scorecard"]["process_detail"]["guessed"] is True
    assert r["scorecard"]["process_reward"] < 0, "guessing right must still score badly"
    assert r["scorecard"]["total_reward"] < 1.0


def test_investigating_first_scores_above_guessing_on_the_same_answer():
    """Same case, same terminal action, different process. If these two ever tie, the process
    verifier has stopped doing anything and the agent's headline becomes a coin flip."""
    case = _case(fraud=True)
    _with_model(['{"reasoning": "x", "action": "confirm_fraud"}'])
    blind = A.run_episode(case)
    _with_model(['{"reasoning": "check card", "action": "inspect_card_signals"}',
                 '{"reasoning": "confirmed", "action": "confirm_fraud"}'])
    worked = A.run_episode(case)
    assert worked["scorecard"]["total_reward"] > blind["scorecard"]["total_reward"]


# ----------------------------------------------------------------- driving the environment

def test_it_stops_at_the_terminal_action():
    """A model that keeps talking after committing would keep stepping a finished episode."""
    _with_model(['{"reasoning": "a", "action": "inspect_card_signals"}',
                 '{"reasoning": "b", "action": "confirm_fraud"}',
                 '{"reasoning": "c", "action": "inspect_customer"}'])
    r = A.run_episode(_case())
    assert [t["action"] for t in r["trajectory"]] == ["inspect_card_signals", "confirm_fraud"]


def test_an_agent_that_never_decides_is_scored_as_a_failure_not_omitted():
    """Investigating forever and committing to nothing is a real failure mode, and the one most
    likely to read as diligence in a transcript."""
    _with_model(['{"reasoning": "more", "action": "inspect_customer"}'] * 20)
    r = A.run_episode(_case(), max_steps=3)
    assert r["available"] is True
    assert r["scorecard"]["terminal_action"] is None
    assert r["scorecard"]["correct"] is False
    assert r["scorecard"]["total_reward"] == -1.0


def test_a_reply_wrapped_in_prose_still_yields_its_action():
    """Models add preambles and code fences. Failing on that would measure formatting rather
    than investigation, and would quietly depress every score."""
    _with_model(['Sure! Here is my choice:\n```json\n'
                 '{"reasoning": "check the card", "action": "inspect_card_signals"}\n```',
                 'I will now decide: {"reasoning": "fraud", "action": "confirm_fraud"}'])
    r = A.run_episode(_case())
    assert [t["action"] for t in r["trajectory"]] == ["inspect_card_signals", "confirm_fraud"]


def test_an_unparseable_reply_abandons_the_episode_rather_than_inventing_an_action():
    """Picking a default action here would put words in the model's mouth and score them."""
    _with_model(["I am not going to answer in that format."])
    r = A.run_episode(_case())
    assert r["scorecard"]["terminal_action"] is None
    assert any(t.get("note") for t in r["trajectory"])


def test_a_failed_model_call_produces_no_score():
    """A network failure must not be scored as a bad investigation, which would silently
    depress the agent's measured performance for a reason that has nothing to do with it."""
    import urllib.error

    def _boom(key, messages):
        raise urllib.error.URLError("no network")
    A._post = _boom
    A._api_key = lambda: "test-key"
    r = A.run_episode(_case())
    assert r["available"] is False and "scorecard" not in r


# --------------------------------------------------------------------------- the comparison

def test_the_comparison_ranks_the_agent_against_the_reference_policies():
    """The reference policies are the honest part. Without them a mediocre agent looks fine,
    because there is nothing on the page saying what a disciplined policy already achieves."""
    case = _case(fraud=True, high_card_signal=True)
    _with_model(['{"reasoning": "check card", "action": "inspect_card_signals"}',
                 '{"reasoning": "fraud", "action": "confirm_fraud"}'])
    c = A.compare(case)
    assert set(c["runs"]) == {"investigator", "trigger_happy", "cautious", "llm_agent"}
    assert c["ranking"] and c["ranking"][0][1] >= c["ranking"][-1][1]
    assert c["best_reference"] in ("investigator", "trigger_happy", "cautious")
    assert isinstance(c["beats_best_reference"], bool)


def test_the_comparison_reports_a_loss_as_a_loss():
    """If the agent cannot beat a hand-written policy, that is the finding. An artifact that
    can only report wins is marketing."""
    case = _case(fraud=True, high_card_signal=True)
    _with_model(['{"reasoning": "clear it", "action": "clear_false_positive"}'])
    c = A.compare(case)
    assert c["beats_best_reference"] is False
    assert c["runs"]["llm_agent"]["scorecard"]["correct"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed ({len(tests)} total)")
    sys.exit(1 if failed else 0)
