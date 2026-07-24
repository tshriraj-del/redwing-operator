"""
Tests for core/authorization_iq.py - the push-rail Authorization IQ pack.

These protect the properties that make it network intelligence rather than a repackaged local
feature vector. The theme: a field is only worth returning if the NETWORK saw something the
querying bank could not, so most tests assert on the network delta and the reveal, not on a
raw score.

Runs under pytest or standalone (python3 tests/test_authorization_iq.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
if OP not in sys.path:
    sys.path.insert(0, OP)

from core import authorization_iq as A
from core import consortium as C


def _senders_for(inst, n, salt=""):
    """n distinct sender ids that all map to the given institution (deterministic)."""
    out, i = [], 0
    while len(out) < n:
        uid = f"user_{salt}_{i}"
        if C.institution_of(uid) == inst:
            out.append(uid)
        i += 1
    return out


def _edge(recipient, sender, fraud=0, amount=100.0, rail="Zelle"):
    return {"recipient": recipient, "sender": sender, "is_fraud": fraud,
            "amount": amount, "rail": rail}


# -- the index ------------------------------------------------------------------

def test_index_counts_distinct_senders_and_rail_norms():
    edges = []
    for s in _senders_for("inst_neobank", 5, "a"):
        edges.append(_edge("recipient:R", s, amount=100.0, rail="Zelle"))
    # a sender paying twice must count once toward fan-in
    edges.append(_edge("recipient:R", edges[0]["sender"], amount=100.0, rail="Zelle"))
    idx = A.build_index(edges)
    assert idx.fanin["recipient:R"] == 5, "fan-in must be DISTINCT senders, not tx count"
    assert idx.recipient_tx["recipient:R"] == 6
    assert idx.rail_norm["Zelle"]["n"] == 6


# -- fan-in: the merchant vs mule distinction -----------------------------------

def test_fanin_does_not_fire_when_concentrated_at_one_institution():
    """High fan-in at a SINGLE institution is a merchant, not a mule. The multi-institution
    test is what stops Authorization IQ flagging every popular payee."""
    edges = [_edge("recipient:MERCH", s) for s in _senders_for("inst_neobank", 25, "m")]
    idx = A.build_index(edges)
    # send from the same institution that banks all those senders
    pack = A.authorize({"sender": "user_probe", "recipient": "recipient:MERCH",
                        "amount": 100.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    fanin = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_fanin")
    assert fanin["institutions_seeing"] == 1
    assert not fanin["fired"], "a single-institution payee is a merchant, must not fire fan-in"


def test_fanin_fires_across_institutions_with_a_positive_network_delta():
    edges = ([_edge("recipient:MULE", s) for s in _senders_for("inst_neobank", 10, "n")]
             + [_edge("recipient:MULE", s) for s in _senders_for("inst_crypto", 10, "c")])
    idx = A.build_index(edges)
    pack = A.authorize({"sender": _senders_for("inst_neobank", 1, "q")[0],
                        "recipient": "recipient:MULE", "amount": 100.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    fanin = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_fanin")
    assert fanin["institutions_seeing"] == 2
    assert fanin["fired"], "cross-bank fan-in above the alert must fire"
    # the delta is the whole point: the network sees more senders than the querying bank
    assert fanin["network_delta"] == fanin["value"] - fanin["local_value"] > 0


# -- the reveal: the field a single bank structurally cannot produce ------------

def _mule_edges():
    """A payee that is clean at the neobank and fraudulent at the crypto off-ramp, so the
    neobank's own view is below the alert line while the network's is above it."""
    edges = [_edge("recipient:CASH", s, fraud=0) for s in _senders_for("inst_neobank", 30, "clean")]
    for i, s in enumerate(_senders_for("inst_crypto", 30, "dirty")):
        edges.append(_edge("recipient:CASH", s, fraud=1 if i < 20 else 0, rail="crypto", amount=8000.0))
    return edges


def test_network_reveal_fires_for_the_bank_that_is_locally_blind():
    idx = A.build_index(_mule_edges())
    pack = A.authorize({"sender": _senders_for("inst_neobank", 1, "victim")[0],
                        "recipient": "recipient:CASH", "amount": 5000.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    assert pack["network_reveal"], "neobank is locally clean but the network flags the mule"
    codes = {c["code"] for c in pack["reason_codes"]}
    assert "AIQ06_NETWORK_REVEAL" in codes
    rep = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_network_rep")
    assert rep["network_delta"] > 0, "the network knows more than the neobank's own book"


def test_no_reveal_for_the_bank_that_could_see_it_alone():
    """The crypto exchange already sees the fraud in its own book, so the network reveals it
    nothing new - only_visible_via_network must be False there."""
    idx = A.build_index(_mule_edges())
    pack = A.authorize({"sender": _senders_for("inst_crypto", 1, "q")[0],
                        "recipient": "recipient:CASH", "amount": 5000.0, "rail": "crypto"}, idx,
                       querying_institution="inst_crypto")
    assert not pack["network_reveal"], "a bank that catches it locally gains no reveal"


# -- amount vs the NETWORK norm, not the sender's own history -------------------

def test_amount_over_norm_uses_the_rails_network_distribution():
    # a tight network norm on the wire rail (~$500), then a $50k wire
    edges = [_edge("recipient:X", f"user_w_{i}", amount=400 + (i % 5) * 50, rail="wire")
             for i in range(40)]
    idx = A.build_index(edges)
    big = A.authorize({"sender": "user_s", "recipient": "recipient:Y",
                       "amount": 50000.0, "rail": "wire"}, idx)
    amt = next(i for i in big["insights"] if i["field"] == "aiq_amount_over_norm")
    assert amt["fired"] and amt["z_vs_network"] >= A.Z_ALERT
    normal = A.authorize({"sender": "user_s", "recipient": "recipient:Y",
                          "amount": 520.0, "rail": "wire"}, idx)
    amt2 = next(i for i in normal["insights"] if i["field"] == "aiq_amount_over_norm")
    assert not amt2["fired"], "an on-norm amount must not fire"


def test_amount_insight_silent_without_a_trusted_rail_norm():
    idx = A.build_index([_edge("recipient:X", "user_a", rail="RTP")])   # 1 obs, below MIN_RAIL_NORM_N
    pack = A.authorize({"sender": "user_s", "recipient": "recipient:Z",
                        "amount": 99999.0, "rail": "RTP"}, idx)
    amt = next(i for i in pack["insights"] if i["field"] == "aiq_amount_over_norm")
    assert not amt["fired"], "no trusted norm -> no insight, not a guess"


# -- newness: established-elsewhere is REASSURING, not suspicious ---------------

def test_new_to_you_but_established_on_network_does_not_raise_risk():
    edges = [_edge("recipient:KNOWN", s) for s in _senders_for("inst_crypto", 20, "e")]
    idx = A.build_index(edges)
    # a neobank customer paying a payee neobank has never seen, but the network has
    pack = A.authorize({"sender": _senders_for("inst_neobank", 1, "new")[0],
                        "recipient": "recipient:KNOWN", "amount": 5000.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    new = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_network_newness")
    assert new["established_elsewhere"] and not new["new_to_network"]
    assert new["risk"] == 0.0, "the network knowing the payee should calm, not alarm"


def test_new_to_the_whole_network_with_a_large_amount_fires():
    idx = A.build_index([_edge("recipient:SEED", "user_seed")])   # network barely knows anyone
    pack = A.authorize({"sender": "user_s", "recipient": "recipient:GHOST",
                        "amount": 9000.0, "rail": "wire"}, idx)
    new = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_network_newness")
    assert new["new_to_network"] and new["fired"]


# -- honesty floor + composition ------------------------------------------------

def test_evidence_floor_keeps_reputation_silent_on_thin_data():
    # a recipient with only a couple of tx: even if both are fraud, the network stays silent
    edges = [_edge("recipient:THIN", _senders_for("inst_neobank", 1, "t1")[0], fraud=1),
             _edge("recipient:THIN", _senders_for("inst_crypto", 1, "t2")[0], fraud=1)]
    idx = A.build_index(edges)
    pack = A.authorize({"sender": "user_s", "recipient": "recipient:THIN",
                        "amount": 1000.0, "rail": "Zelle"}, idx)
    rep = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_network_rep")
    assert not rep["sufficient_evidence"] and not rep["fired"]


def test_network_risk_is_bounded_and_codes_are_valid():
    idx = A.build_index(_mule_edges())
    pack = A.authorize({"sender": _senders_for("inst_neobank", 1, "z")[0],
                        "recipient": "recipient:CASH", "amount": 40000.0, "rail": "crypto"}, idx,
                       querying_institution="inst_neobank")
    assert 0.0 <= pack["network_risk"] <= 1.0
    for c in pack["reason_codes"]:
        assert c["code"] in A.AIQ_CODES


def test_authorize_is_deterministic():
    idx = A.build_index(_mule_edges())
    p = {"sender": "user_det", "recipient": "recipient:CASH", "amount": 5000.0, "rail": "Zelle"}
    a = A.authorize(p, idx, querying_institution="inst_neobank")
    b = A.authorize(p, idx, querying_institution="inst_neobank")
    assert a == b


def test_clean_payment_adds_no_network_risk():
    """A payee the network knows as clean, on-norm amount, single institution: the pack should
    say plainly that it adds nothing, rather than manufacturing risk."""
    edges = [_edge("recipient:GOOD", s, fraud=0, amount=100.0)
             for s in _senders_for("inst_neobank", 40, "g")]
    idx = A.build_index(edges)
    pack = A.authorize({"sender": "user_probe2", "recipient": "recipient:GOOD",
                        "amount": 100.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    assert not pack["network_reveal"]
    assert pack["network_risk"] < 0.1
    assert "adds no risk" in pack["explanation"]


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
