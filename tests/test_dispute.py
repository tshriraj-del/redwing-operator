"""
Tests for the card dispute rail.

The load-bearing tests here are the ones that REFUSE to emit a label. Emitting is easy and every
system in this space does it; the value is entirely in the four cases where a naive pipeline
writes a fraud label and this one does not:

    a chargeback that has not settled          -> a claim is not an adjudication
    a service dispute (13.1, 4853)             -> not fraud, and training on it poisons the model
    a fraud claim the merchant won             -> the label INVERTS, it does not simply vanish
    a monitoring-program chargeback (10.5)     -> says something about the merchant, not the card

Each has a named test and each is mutation-verified against the module it guards.

Runs under pytest or standalone (python3 tests/test_dispute.py).
"""

import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import dispute as D  # noqa: E402

T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _iso(d):
    return d.isoformat().replace("+00:00", "Z")


def _cb(code="10.4", days=0, amount=250.0):
    return {"stage": "chargeback", "reason_code": code, "amount": amount,
            "ts": _iso(T0 + timedelta(days=days))}


def _stage(name, days, **kw):
    return {"stage": name, "ts": _iso(T0 + timedelta(days=days)), **kw}


def _end(terminal, days):
    return {"terminal": terminal, "ts": _iso(T0 + timedelta(days=days))}


# ------------------------------------------------------- rule 1: a claim is not an outcome

def test_an_open_chargeback_emits_no_label():
    """THE most common mistake in this domain: a chargeback lands and the row is flipped to
    fraud on the spot. Nobody has adjudicated anything yet. This is the immature-window error
    in a different costume, and label_maturity refuses the same thing for the same reason."""
    st = D.advance([_cb("10.4")])
    d = D.derive_outcome(st)
    assert d["emit"] is False, "labelled an unadjudicated claim"
    assert "not an adjudication" in d["reason"] or "claim" in d["reason"]


def test_a_dispute_mid_representment_still_emits_nothing():
    st = D.advance([_cb("10.4"), _stage("representment", 20)])
    assert st["stage"] == "representment"
    assert D.derive_outcome(st)["emit"] is False


# ------------------------------------------------------- rule 2: family gates the label space

def test_a_service_dispute_never_reaches_the_fraud_label_space():
    """13.1 is 'merchandise not received'. The cardholder is not claiming fraud, they are
    claiming the parcel never arrived. Writing that into outcome.is_fraud trains the model that
    late shipping is fraud, and it is a large share of all chargebacks."""
    st = D.advance([_cb("13.1"), _end("issuer_won", 60)])
    d = D.derive_outcome(st)
    assert d["emit"] is False, "a service dispute was written into the fraud label space"
    assert "consumer" in d["reason"]


def test_a_mastercard_cardholder_dispute_is_also_withheld():
    """4853 is the Mastercard equivalent. Same rule, different network, and a taxonomy that
    only knew one network would silently mislabel the other."""
    st = D.advance([_cb("4853"), _end("issuer_won", 60)])
    assert D.derive_outcome(st)["emit"] is False


def test_a_settled_fraud_claim_does_emit():
    """The other half. If the gate is so tight nothing gets through, the rail is useless."""
    st = D.advance([_cb("4837"), _end("issuer_won", 45)])
    d = D.derive_outcome(st)
    assert d["emit"] is True and d["label_value"] == "fraud"
    assert d["confidence"] >= 0.85, "an unambiguous fraud code should settle at high confidence"


# ------------------------------------------------------- rule 3: a won representment inverts

def test_a_fraud_claim_the_merchant_won_inverts_to_legit():
    """Not 'no label'. The network adjudicated that the charge was VALID, which is positive
    evidence the transaction was not fraud. A pipeline that only ever writes fraud labels
    accumulates every dispute the cardholder lost as a permanent false positive in training."""
    st = D.advance([_cb("10.4"), _stage("representment", 20, compelling_evidence=True),
                    _end("merchant_won", 55)])
    d = D.derive_outcome(st)
    assert d["emit"] is True
    assert d["label_value"] == "legit", f"expected an inverted label, got {d['label_value']}"


def test_a_defeated_fraud_claim_raises_the_first_party_flag():
    """A cardholder who disputes a charge they made, and loses on the merchant's evidence, is
    the canonical friendly-fraud shape. It requires a different action from third-party fraud
    and is completely invisible to a binary fraud label."""
    st = D.advance([_cb("10.4"), _stage("representment", 20, compelling_evidence=True),
                    _end("merchant_won", 55)])
    assert D.derive_outcome(st)["first_party_signal"] is True


def test_a_withdrawn_fraud_claim_is_weaker_evidence_than_a_contested_win():
    """Both point at first party, but a withdrawal was never tested by anyone. The confidences
    must not be equal, or the ledger cannot tell an adjudicated fact from an abandoned one."""
    won = D.derive_outcome(D.advance([_cb("10.4"), _stage("representment", 20, compelling_evidence=True),
                                      _end("merchant_won", 55)]))
    wd = D.derive_outcome(D.advance([_cb("10.4"), _end("withdrawn", 30)]))
    assert wd["label_value"] == won["label_value"] == "legit"
    assert wd["confidence"] < won["confidence"], "a withdrawal was scored as strongly as a win"


# ------------------------------------------------------- the ambiguity that pays for itself

def test_an_ambiguous_code_settles_at_lower_confidence_than_an_unambiguous_one():
    """Visa's own definition of 10.4 spans true fraud, friendly fraud AND merchant error. An
    uncontested win on 10.4 does not separate them, so it cannot carry the same weight as a win
    on 4837 ('no cardholder authorization'), which asserts one thing only."""
    amb = D.derive_outcome(D.advance([_cb("10.4"), _end("issuer_won", 60)]))
    clear = D.derive_outcome(D.advance([_cb("4837"), _end("issuer_won", 60)]))
    assert amb["emit"] and clear["emit"]
    assert amb["confidence"] < clear["confidence"], (
        "an ambiguous code was trusted as much as an unambiguous one")


def test_a_monitoring_program_chargeback_is_not_evidence_about_the_cardholder():
    """10.5 is raised by the Visa Fraud Monitoring Program against a merchant with an excessive
    fraud ratio. No cardholder disputed anything. Treating it as a cardholder fraud claim
    attributes a merchant's portfolio problem to one card."""
    st = D.advance([_cb("10.5"), _end("issuer_won", 60)])
    d = D.derive_outcome(st)
    assert d["emit"] is False, "a program artifact was recorded as cardholder-asserted fraud"
    assert "monitoring program" in d["reason"]


def test_an_unknown_reason_code_is_never_guessed_at():
    st = D.advance([_cb("99.9"), _end("issuer_won", 60)])
    d = D.derive_outcome(st)
    assert d["emit"] is False and "not in the taxonomy" in d["reason"]


def test_an_expired_dispute_produces_no_adjudicated_fact():
    """Nobody contested and nobody prevailed. A deadline lapsing is not evidence."""
    st = D.advance([_cb("10.4"), _end("expired", 130)])
    assert D.derive_outcome(st)["emit"] is False


# ------------------------------------------------------- the state machine

def test_a_resent_stage_does_not_look_like_escalation():
    """Acquirers re-send. If a repeated representment advanced the pointer to pre-arbitration,
    the dispute would appear to escalate on a duplicate and its maturity floor would grow."""
    st = D.advance([_cb("10.4"), _stage("representment", 20), _stage("representment", 21)])
    assert st["stage"] == "representment"


def test_a_stage_arriving_out_of_order_does_not_move_the_dispute_backwards():
    """A late-delivered chargeback event after a representment must not reopen the dispute. A
    dispute that appears to regress would re-derive a label that was already emitted."""
    st = D.advance([_cb("10.4"), _stage("pre_arbitration", 40), _stage("representment", 20)])
    assert st["stage"] == "pre_arbitration"


def test_the_opening_reason_code_wins_over_later_ones():
    """The dispute is defined by what was originally claimed. A code appearing on a later event
    must not silently reclassify the whole dispute into a different label space."""
    st = D.advance([_cb("10.4"), {"stage": "representment", "reason_code": "13.1",
                                  "ts": _iso(T0 + timedelta(days=20))}])
    assert st["reason_code"] == "10.4"


# ------------------------------------------------------- maturity, the card advantage

def test_the_maturity_floor_grows_with_the_stage_reached():
    """The point of the card rail: unlike push, the clock is defined by the rulebook, so a
    cohort's floor is computable rather than estimated. A dispute that went to arbitration has
    been open longer than one that settled at chargeback."""
    assert D.maturity_floor_days("10.4") < D.maturity_floor_days("10.4", "arbitration")


def test_settled_by_returns_a_real_date_and_degrades_on_junk():
    assert D.settled_by(_iso(T0), "10.4") > T0
    assert D.settled_by("not-a-date", "10.4") is None


# ------------------------------------------------------- the ledger hand-off

def test_the_ledger_record_carries_the_settlement_time_not_the_ingest_time():
    """The maturity curve measures the lag between a fact becoming true and us learning it.
    Stamping effective_ts with the moment we processed the file makes that lag read as zero and
    the curve becomes a measure of our own batch schedule."""
    st = D.advance([_cb("4837"), _end("issuer_won", 45)])
    rec = D.to_ledger_record(st, "txn_1", transaction_ts=_iso(T0))
    assert rec is not None
    assert rec["effective_ts"].startswith("2026-04-15"), rec["effective_ts"]
    assert rec["source"] == "chargeback" and rec["outcome"] == "fraud"


def test_a_withheld_outcome_produces_no_ledger_record_at_all():
    """Withholding must not leak through as a record with a null label; the ledger would count
    it as coverage."""
    assert D.to_ledger_record(D.advance([_cb("13.1"), _end("issuer_won", 60)]), "txn_2") is None
    assert D.to_ledger_record(D.advance([_cb("10.4")]), "txn_3") is None


def test_an_empty_or_junk_event_list_degrades_rather_than_raising():
    """This sits behind an ingest endpoint, so hostile input must not take it down."""
    for bad in ([], None, [{}], [{"stage": "nonsense"}], [{"ts": None}]):
        st = D.advance(bad)
        assert D.derive_outcome(st)["emit"] is False


# ------------------------------------------------------- the ledger actually keeps the nuance

def test_the_confidence_gradation_survives_the_write_to_the_ledger():
    """FOUND BY RUNNING IT, NOT BY READING IT. Every test above passed while the ledger wrote a
    hardcoded 0.95 for every source and every record, so the whole gradation this module computes
    was flattened the moment it left. A withdrawn dispute trained exactly as hard as an
    adjudicated fraud. The unit tests could not see it because they stopped at to_ledger_record.
    """
    import tempfile
    from core.outcome_ledger import record_outcome
    from core.store import Store

    s = Store(os.path.join(tempfile.mkdtemp(), "t.db"))
    try:
        cases = {
            "weak":   D.advance([_cb("10.4"), _end("withdrawn", 30)]),
            "strong": D.advance([_cb("4837"), _end("issuer_won", 45)]),
        }
        got = {}
        for name, st in cases.items():
            rec = D.to_ledger_record(st, f"ref_{name}")
            assert record_outcome(s, rec)["ok"], name
            lbl = [x for x in s.current_labels(subject_ref=f"ref_{name}")
                   if x.label_space == "outcome"][0]
            got[name] = lbl.confidence
        assert got["weak"] < got["strong"], (
            f"the ledger flattened the gradation: withdrawn={got['weak']} "
            f"settled-fraud={got['strong']}")
    finally:
        s.close()


def test_the_reason_code_reaches_the_ledger_as_its_own_field():
    """It was being buried inside the reference string, so the ledger could not be queried by
    dispute type without splitting text."""
    rec = D.to_ledger_record(D.advance([_cb("4837"), _end("issuer_won", 45)]), "ref_rc")
    assert rec["reason_code"] == "4837"


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
