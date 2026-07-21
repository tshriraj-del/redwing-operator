"""
Tests for core/replay.py - the Phase 2 substrate filler.

These protect the honesty guarantees, not the plumbing. A replay over a fully-labeled dataset
is trivially easy to make flattering: label everything, and the trained model looks great
because it was handed outcomes production would never have seen. The tests below pin the
three properties that stop that:

  - a HELD decision is recorded and left unlabelled (censored by our own enforcement)
  - an ALLOWED decision gets its outcome label
  - a holdout RELEASE converts a would-be-held case into an observed one, which is the entire
    reason the holdout costs money

Runs under pytest or standalone (python3 tests/test_replay.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
if OP not in sys.path:
    sys.path.insert(0, OP)

from core.replay import BASELINE_RULE, _as_bool, baseline_rule, replay, replay_row
from core.store import Store


def _store():
    return Store(os.path.join(tempfile.mkdtemp(), "replay.db"))


def _row(tid, fraud, fires):
    """A source row. `fires` controls whether the baseline rule triggers."""
    return {
        "transaction_id": tid, "user_id": "u1", "amount": "100.0", "payment_rail": "wire",
        "is_fraud": "True" if fraud else "False",
        "amount_vs_max": "0.95" if fires else "0.10",
        "recipient_familiarity": "0.05" if fires else "0.90",
        "amount_zscore": "1.0", "velocity_1h": "0.1", "hour_risk": "0.2", "rail_risk": "0.3",
        "device_familiarity": "0.5", "velocity_4h": "0.1", "velocity_24h": "0.1",
        "new_recipient_streak": "0", "is_crypto": "0", "is_instant_rail": "1", "is_p2p": "0",
    }


# -- the rule ------------------------------------------------------------------

def test_baseline_rule_matches_its_documented_definition():
    """The rule is fixed in advance and quoted in the module docstring; if it drifts, the
    measured precision/recall recorded alongside it becomes a lie."""
    assert baseline_rule({"amount_vs_max": 0.95, "recipient_familiarity": 0.05}) is True
    assert baseline_rule({"amount_vs_max": 0.95, "recipient_familiarity": 0.50}) is False
    assert baseline_rule({"amount_vs_max": 0.50, "recipient_familiarity": 0.05}) is False
    assert "amount_vs_max" in BASELINE_RULE["description"]


def test_as_bool_handles_the_forms_the_source_data_uses():
    for v in ("True", "true", True, "1", 1, "yes"):
        assert _as_bool(v) is True, v
    for v in ("False", "false", False, "0", 0, "", None, "no"):
        assert _as_bool(v) is False, v


# -- the censoring guarantee ---------------------------------------------------

def test_held_decision_is_recorded_but_left_unlabelled():
    """The core honesty property. We blocked it, so we never learned the outcome, so it must
    carry no label even though the dataset knows the answer."""
    s = _store()
    # rate 0 so the rule's HOLD is always enforced, never released
    r = replay_row(s, _row("held_1", fraud=True, fires=True), holdout_config={"rate": 0.0})
    assert r["enforced"] == "HOLD" and r["observed"] is False

    dec = s.latest_decision_for_subject("held_1")
    assert dec is not None, "the decision itself must still be recorded"
    assert dec.features, "and must carry its point-in-time feature snapshot"

    gold = [l for l in s.current_labels(subject_ref="held_1")
            if l.source == "confirmed_loss"]
    assert gold == [], "a held decision must not receive an outcome label"


def test_allowed_decision_receives_its_outcome_label():
    s = _store()
    r = replay_row(s, _row("allowed_1", fraud=True, fires=False), holdout_config={"rate": 0.0})
    assert r["enforced"] == "ALLOW" and r["observed"] is True

    gold = [l for l in s.current_labels(subject_ref="allowed_1")
            if l.source == "confirmed_loss"]
    assert len(gold) == 1
    assert gold[0].label_value == "True"       # the ledger observed the fraud


def test_holdout_release_turns_a_would_be_block_into_an_observed_outcome():
    """This is what the holdout buys: a case the rule wanted to block, released and observed,
    so its true outcome enters training instead of being censored away."""
    s = _store()
    # rate 1.0 releases every eligible would-be-block
    r = replay_row(s, _row("released_1", fraud=True, fires=True),
                   holdout_config={"rate": 1.0, "max_liability": 1e9})
    assert r["released"] is True
    assert r["enforced"] == "ALLOW"
    assert r["observed"] is True

    gold = [l for l in s.current_labels(subject_ref="released_1")
            if l.source == "confirmed_loss"]
    assert len(gold) == 1 and gold[0].label_value == "True"


def test_liability_ceiling_still_blocks_an_expensive_case():
    """The holdout must never release a case too expensive to be worth the counterfactual,
    however high the release rate is set."""
    s = _store()
    row = _row("expensive_1", fraud=True, fires=True)
    row["amount"] = "500000.0"
    r = replay_row(s, row, holdout_config={"rate": 1.0, "max_liability": 100.0})
    assert r["released"] is False
    assert r["enforced"] == "HOLD" and r["observed"] is False


# -- the summary ---------------------------------------------------------------

def test_replay_summary_accounting_is_consistent():
    s = _store()
    rows = [_row(f"t{i}", fraud=(i % 10 == 0), fires=(i % 3 == 0)) for i in range(60)]
    out = replay(s, iter(rows), holdout_config={"rate": 0.0})

    assert out["replayed"] == 60
    assert out["allowed"] + out["enforced_hold"] == out["replayed"]
    assert out["observed_labeled"] == out["allowed"]     # every allow is observed
    assert out["censored"] == out["enforced_hold"]
    assert out["rule_fired"] == out["enforced_hold"]     # with no holdout, fired == held
    assert 0.0 < out["censored_fraction"] < 1.0


def test_replay_respects_its_limit():
    s = _store()
    rows = [_row(f"lim{i}", fraud=False, fires=False) for i in range(50)]
    out = replay(s, iter(rows), limit=10)
    assert out["replayed"] == 10


# -- standalone runner (no pytest needed) --------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed ({len(tests)} total)")
    sys.exit(1 if failed else 0)
