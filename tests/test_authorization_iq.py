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
    """High fan-in whose senders all bank in one place is an ordinary payee serving that
    institution's customer base, not a collector. Concentration is what stops Authorization IQ
    flagging every popular payee."""
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


def test_fanin_does_not_fire_on_a_universal_merchant():
    """THE case the previous discriminator got wrong, and the reason it was replaced.

    The old gate fired on any payee seen by two or more institutions, justified by the claim
    that a legitimate merchant "concentrates within the acquirer that banks it". That confuses
    where a merchant banks with where its CUSTOMERS bank. A supermarket is paid by customers of
    every institution, so under the old rule every large merchant fired, and measured across the
    reference ledger the gate separated fraud from legitimate payees at a lift of 1.00x.

    A universal merchant has an even split AND overwhelming fan-in. What marks a collector is an
    even split at a fan-in that no ordinary payee of that kind would have, so this asserts the
    merchant stays quiet while remaining plainly visible to the network."""
    edges = ([_edge("recipient:TESCO", s) for s in _senders_for("inst_neobank", 200, "n")]
             + [_edge("recipient:TESCO", s) for s in _senders_for("inst_crypto", 200, "c")])
    idx = A.build_index(edges)
    pack = A.authorize({"sender": _senders_for("inst_neobank", 1, "q")[0],
                        "recipient": "recipient:TESCO", "amount": 40.0, "rail": "card"}, idx,
                       querying_institution="inst_neobank")
    fanin = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_fanin")
    assert fanin["institutions_seeing"] == 2, "the merchant IS seen by both, as expected"
    assert fanin["value"] >= A.FANIN_ALERT, "and its fan-in IS above the raw alert line"
    # ... yet it must not fire, because an even split is normal for something this universal.
    assert not fanin["fired"], (
        "a universal merchant fired the collector insight; this is the exact false positive "
        "the concentration test exists to remove")


def test_fanin_risk_needs_both_properties_not_either_one():
    """Multiplied, not added. Either property alone is ordinary: plenty of payees have high
    fan-in, plenty have an even split. Only together do they lack an innocent explanation."""
    # high fan-in, fully concentrated -> no risk
    conc = A.build_index([_edge("recipient:A", s)
                          for s in _senders_for("inst_neobank", 40, "a")])
    # even split but fan-in below the band -> no risk (one sender each side)
    tiny = A.build_index([_edge("recipient:B", s) for s in _senders_for("inst_neobank", 1, "b")]
                         + [_edge("recipient:B", s) for s in _senders_for("inst_crypto", 1, "c")])
    for idx, rid in ((conc, "recipient:A"), (tiny, "recipient:B")):
        pack = A.authorize({"sender": "user_probe", "recipient": rid,
                            "amount": 100.0, "rail": "Zelle"}, idx,
                           querying_institution="inst_neobank")
        fi = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_fanin")
        assert fi["risk"] == 0.0 and not fi["fired"], f"{rid} should carry no fan-in risk"


def test_fanin_reports_concentration_and_never_fires_on_an_unseen_payee():
    """Absence is not evidence. A payee with no senders at all has an undefined split, and the
    insight must default to fully concentrated rather than to maximally suspicious, or every
    brand-new payee would fire the collector code. AIQ04 is what covers new-to-network."""
    idx = A.build_index([_edge("recipient:OTHER", s)
                         for s in _senders_for("inst_neobank", 5, "o")])
    pack = A.authorize({"sender": "user_probe", "recipient": "recipient:GHOST",
                        "amount": 100.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    fi = next(i for i in pack["insights"] if i["field"] == "aiq_recipient_fanin")
    assert fi["concentration"] == 1.0, "an unseen payee must default to concentrated, not split"
    assert not fi["fired"] and fi["risk"] == 0.0


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


# -- the scoring contract: how the network may move a live decision -------------
#
# These pin the rules _network_view enforces in main.py. They are expressed against the same
# authorize() output that helper consumes, so they hold wherever it is wired.

def _apply(local_score, pack):
    """The escalate-only composition, imported rather than reimplemented. A test that keeps
    its own copy of the rule cannot catch the rule changing."""
    return A.apply_escalate_only(local_score, pack)


def test_network_escalates_a_payment_the_local_model_would_have_missed():
    """THE point of wiring this into scoring. A payment that looks ordinary to the local model
    must still be raised when the network knows the payee is a mule - that lift is money the
    bank would otherwise have sent."""
    idx = A.build_index(_mule_edges())
    pack = A.authorize({"sender": _senders_for("inst_neobank", 1, "v2")[0],
                        "recipient": "recipient:CASH", "amount": 180.0, "rail": "card"}, idx,
                       querying_institution="inst_neobank")
    local = 0.20
    assert _apply(local, pack) > local, "the network must be able to raise a missed payment"


def test_network_never_lowers_a_score_the_local_book_earned():
    """ESCALATE-ONLY. A clean network view must not talk the local model down off a signal it
    found in its own data: the consortium adds evidence, it does not grant absolution."""
    edges = [_edge("recipient:GOOD", s, fraud=0) for s in _senders_for("inst_neobank", 40, "g2")]
    idx = A.build_index(edges)
    pack = A.authorize({"sender": "user_probe9", "recipient": "recipient:GOOD",
                        "amount": 100.0, "rail": "Zelle"}, idx,
                       querying_institution="inst_neobank")
    local = 0.93
    assert _apply(local, pack) == local, "a clean network view must never reduce a local score"


def test_thin_evidence_cannot_move_a_live_decision():
    """Below the consortium's evidence floor the combined rate is noise. It may be reported,
    but it must not move a real decision."""
    edges = [_edge("recipient:THIN2", _senders_for("inst_neobank", 1, "x1")[0], fraud=1),
             _edge("recipient:THIN2", _senders_for("inst_crypto", 1, "x2")[0], fraud=1)]
    idx = A.build_index(edges)
    pack = A.authorize({"sender": "user_s", "recipient": "recipient:THIN2",
                        "amount": 5000.0, "rail": "wire"}, idx)
    assert not pack["sufficient_evidence"]
    local = 0.10
    assert _apply(local, pack) == local, "noise must not escalate a live decision"


# -- escalate-only as a checkable guarantee, not a design intention -------------

def test_escalate_only_never_lowers_a_score():
    """Claim 1, DIRECTION. The original rule, and the only one that was ever stated."""
    for local, nr, suf in ((0.9, 0.1, True), (0.4, 0.4, True), (0.7, 0.0, True),
                           (0.3, 0.99, False), (0.0, 0.5, True)):
        out = A.apply_escalate_only(local, {"sufficient_evidence": suf, "network_risk": nr})
        assert out >= local - 1e-12, f"escalate-only lowered {local} to {out}"


def test_direction_alone_does_not_make_the_guarantee_hold():
    """Claim 2, BUDGET, and THE lesson. Folding the network into the live score degraded
    detection at every alert budget while direction held on every single payment: the network
    simply raised 56.9% of scores, and a floor over most of the book destroys the local
    ranking as effectively as lowering scores would. A rule that cannot fail on the case that
    actually hurt is not a guarantee, it is a slogan."""
    flood = [(0.001, 0.02)] * 570 + [(0.001, 0.001)] * 430   # 57% raised, none lowered
    a = A.audit_escalate_only(flood)
    assert a["direction_violations"] == 0, "the flooding case never lowered a score"
    assert not a["holds"], (
        "the audit passed a composition that raised 57% of payments; the budget claim is "
        "what makes this guarantee mean something")
    assert a["escalation_rate"] > a["escalation_budget"]


def test_a_sparse_escalation_holds():
    """The shape the reweighted pack actually produces: a small number of payments raised."""
    ok = [(0.001, 0.9)] * 20 + [(0.001, 0.001)] * 980       # 2% raised
    a = A.audit_escalate_only(ok)
    assert a["holds"] and a["within_budget"]
    assert a["direction_violations"] == 0


def test_the_audit_catches_an_actual_de_escalation():
    """If the composition is ever changed to blend rather than take a maximum, direction goes
    first and this is what notices."""
    a = A.audit_escalate_only([(0.8, 0.4), (0.1, 0.1), (0.2, 0.9)])
    assert a["direction_violations"] == 1 and not a["holds"]


def test_displacement_is_reported_so_the_cost_is_visible():
    """Claim 3. Reported rather than bounded, because the honest value depends on how many
    cases the team can actually review."""
    a = A.audit_escalate_only([(0.9, 0.9)] * 10 + [(0.1, 0.99)] * 10, top_n=10)
    assert 0.0 <= a["top_n_displaced"] <= 1.0
    assert a["top_n_displaced"] > 0, "raising ten low scores above the top ten displaced none?"


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
