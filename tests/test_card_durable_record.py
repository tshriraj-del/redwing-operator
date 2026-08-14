"""
Tests for the durable record on the card authorization path.

WHY THIS FILE EXISTS. `/authorize` was the one decision path that wrote nothing. The card rail
therefore produced no labels, no outcome-ledger entries and no holdout membership, and its model
could never be measured for decay or graduated. That was the most consequential entry in the
conformance test's KNOWN_GAPS and it is what this closes.

Two properties carry the whole design and both are easy to get wrong under a network deadline:

  THE JOIN KEY IS THE ARN. A chargeback arrives months later referencing the acquirer reference
  number, because that is the only identifier the issuer and acquirer both hold. File the
  decision under an internal id and the outcome can never be joined back to the decision that
  caused it, which makes the record worthless for exactly the purpose it exists for.

  HOLDOUT IS DECIDED IN THE DECISION, NOT AT THE WRITE. The write happens off the response path
  because a card auth answers against a deadline. If membership were computed by the writer, a
  slow or dropped write would silently change which cases were sampled, and a holdout whose
  membership depends on write success is not randomised.

Runs under pytest or standalone (python3 tests/test_card_durable_record.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_RECOVERY_SECRET", "card-record-test")

from core.authorization import authorize, card_subject_ref, durable_record  # noqa: E402
from core.holdout import holdout_decision  # noqa: E402

MSG = {"amount": 900.0, "merchant_name": "Acme Supplies", "cardholder_name": "Jane Roe",
       "entry_mode": "ecom", "mcc_code": 5999, "account_age_days": 30,
       "available_balance": 9000.0, "bin": "400000", "merchant_id": "m_1",
       "arn": "74537501234567890123456"}


def _decide(**over):
    msg = {**MSG, **over}
    return msg, authorize(msg, score_fn=lambda m: (0.42, {"p_fraud": 0.42, "features": {"a": 1}}))


# ------------------------------------------------------------------ the join key

def test_the_subject_ref_is_the_acquirer_reference_number():
    """THE join. A dispute file references the ARN; nothing else is held by both sides."""
    assert card_subject_ref(MSG) == MSG["arn"]


def test_network_identifiers_outrank_internal_ones():
    """If an internal transaction_id won, the record would be filed under a key no dispute file
    will ever carry, and every outcome would arrive unjoinable."""
    m = {**MSG, "transaction_id": "internal_999"}
    assert card_subject_ref(m) == MSG["arn"]
    assert card_subject_ref({"transaction_id": "internal_999"}) == "internal_999"


def test_a_message_with_no_reference_at_all_is_not_recorded():
    """A decision nobody can join an outcome to is not worth a row, and writing it would inflate
    the coverage denominator with rows that can never be labelled."""
    msg, dec = _decide(arn="")
    msg.pop("arn", None)
    assert durable_record(msg, dec, holdout_fn=holdout_decision) == {}


# ------------------------------------------------------------------ holdout timing

def test_holdout_membership_is_decided_from_the_message_alone():
    """No store, no clock, no write. The record is fully determined by the message and the
    decision, which is what makes it safe to persist after the response has gone out."""
    msg, dec = _decide()
    rec = durable_record(msg, dec, holdout_fn=holdout_decision)
    assert "holdout" in rec and isinstance(rec["holdout"], bool)


def test_the_same_authorization_always_lands_in_the_same_holdout_bucket():
    """Deterministic on the ARN. If a retry could re-roll membership, a fraudster could retry
    into a release, and the sample would no longer be random."""
    msg, dec = _decide()
    a = durable_record(msg, dec, holdout_fn=holdout_decision)
    b = durable_record(msg, dec, holdout_fn=holdout_decision)
    assert a["holdout"] == b["holdout"] and a["subject_ref"] == b["subject_ref"]


def test_a_released_holdout_reports_the_enforced_action_not_the_proposed_one():
    """The record has to say what actually happened. If it stored the proposed DECLINE while the
    payment was really allowed through, the outcome would be attributed to a decision that was
    never enforced, and the holdout would be measuring a fiction."""
    msg, dec = _decide()
    released = lambda ref, action, liab: {          # noqa: E731
        "release": True, "enforced_action": "ALLOW", "holdout": True, "reason": "forced"}
    rec = durable_record(msg, dec, holdout_fn=released)
    assert rec["released"] is True and rec["action"] == "ALLOW"


def test_without_a_holdout_function_nothing_is_silently_sampled():
    """Absent the injection the record must not invent membership. A default-true here would put
    cases in the holdout that no policy selected."""
    msg, dec = _decide()
    rec = durable_record(msg, dec)
    assert rec["holdout"] is False and rec["released"] is False


# ------------------------------------------------------------------ what the record carries

def test_the_record_carries_what_the_dispute_rail_needs_to_compute_maturity():
    """entry_mode and mcc travel with the decision because the dispute clock and the CP/CNP base
    rate both depend on them, and re-deriving them from a chargeback file months later is not
    possible: the file describes the dispute, not the original terminal."""
    msg, dec = _decide()
    r = durable_record(msg, dec, holdout_fn=holdout_decision)["rationale"]
    assert r["entry_mode"] == "ecom" and r["mcc_code"] == 5999 and r["bin"] == "400000"
    assert r["rail"] == "card" and r["path"] == "authorize"


def test_the_recorded_score_is_the_one_that_decided():
    """Recording a different number from the one the policy acted on makes every later analysis
    of that decision wrong, and it is invisible without this assertion."""
    msg, dec = _decide()
    rec = durable_record(msg, dec, holdout_fn=holdout_decision)
    assert abs(rec["score"] - 0.42) < 1e-9


def test_a_screening_block_is_still_recorded():
    """A blocked payment is a decision, and the most censored population in the book. Dropping
    it is how the training set quietly becomes only the payments that were allowed."""
    msg = {**MSG, "merchant_name": "OFAC SDN TEST ENTITY"}
    dec = authorize(msg, score_fn=lambda m: (0.9, {"p_fraud": 0.9}))
    rec = durable_record(msg, dec, holdout_fn=holdout_decision)
    assert rec, "a screening block produced no durable record"
    assert rec["subject_ref"] == MSG["arn"]


def test_a_hostile_or_empty_message_degrades_rather_than_raising():
    """This sits on a live decision path behind a deadline."""
    for bad in ({}, {"arn": ""}, {"arn": "x"}):
        durable_record(bad, {"action": "ALLOW"}, holdout_fn=holdout_decision)


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
