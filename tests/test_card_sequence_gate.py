"""
Tests for the card sequence gate at SERVING time, and the card key it depends on.

WHY THIS FILE EXISTS. The gate was measured in redwing-ml against a streamed ledger. Serving is
the mirror image: a live authorization has no future to stream and must ask the substrate. Two
things can silently go wrong in that translation and both are guarded here.

  THE PAN MUST NEVER BE STORED. A stored PAN puts the decision substrate in PCI scope. The key
  is a salted hash, the salt comes from the environment, and an unsalted deployment is REPORTED
  rather than silently accepted, because it looks identical to a correct one until the DB leaks.

  THE WINDOW MUST BE STRICTLY PRIOR. The authorization being scored must not appear in its own
  history. That was trivially true when the write blocked the response; it stopped being obvious
  the moment the card write moved off the deadline path, so it is asserted rather than assumed.

Runs under pytest or standalone (python3 tests/test_card_sequence_gate.py).
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_CARD_SALT", "test-salt-not-a-real-one")

from core import card_identity as CI          # noqa: E402
from core.card_history import sequence_view   # noqa: E402
from core.store import Store, eid             # noqa: E402

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
PAN = "4111111111111111"


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def _store_with(card, n, amount=100.0, hours_ago=1.0, base=NOW):
    s = Store(os.path.join(tempfile.mkdtemp(), "t.db"))
    for i in range(n):
        s.log_decision(subject_ref=f"arn_{i}", entity_id=eid("card", card), action="ALLOW",
                       module="card_scorer", score=0.1, expected_liability=amount * 0.01,
                       features={"amount": amount},
                       ts=_iso(base - timedelta(hours=hours_ago)))
    return s


# ------------------------------------------------------------------ the card key

def test_the_pan_never_appears_in_the_key():
    """THE compliance property. If the raw PAN, or any recognisable slice of it, survives into
    the key then the decisions table is cardholder data and the whole substrate is in PCI scope.
    """
    k = CI.card_key({"pan": PAN})
    assert k and PAN not in k
    for i in range(0, len(PAN) - 5):
        assert PAN[i:i + 6] not in k, f"a 6-digit run of the PAN survived into {k}"


def test_the_key_is_salted_so_a_leaked_table_is_not_a_card_list():
    """A bare SHA-256 of a PAN is reversible in practice: ~10^15 candidates with a known checksum
    and known BIN ranges is hours of work. Different salt must give a different key."""
    a = CI.card_key({"pan": PAN})
    os.environ["REDWING_CARD_SALT"] = "a-different-salt"
    try:
        b = CI.card_key({"pan": PAN})
    finally:
        os.environ["REDWING_CARD_SALT"] = "test-salt-not-a-real-one"
    assert a != b, "the key did not change with the salt, so it is effectively unsalted"


def test_an_absent_salt_is_reported_rather_than_silently_accepted():
    """A system hashing with no salt looks exactly like one doing it properly, right up until
    the database leaks. It has to be inspectable."""
    saved = os.environ.pop("REDWING_CARD_SALT", None)
    try:
        assert CI.salt_configured() is False
    finally:
        if saved is not None:
            os.environ["REDWING_CARD_SALT"] = saved
    assert CI.salt_configured() is True


def test_a_network_token_is_preferred_over_the_pan():
    """A token is the industry's own answer to this problem and is safe without the salt. If
    both are present the token wins, so the PAN is not touched at all."""
    both = CI.card_key({"card_token": "tok_abc", "pan": PAN})
    assert both == CI.card_key({"card_token": "tok_abc"})
    assert both != CI.card_key({"pan": PAN})


def test_the_same_card_always_yields_the_same_key():
    assert CI.card_key({"pan": PAN}) == CI.card_key({"pan": "4111-1111-1111-1111"})


def test_a_message_with_no_card_identifier_yields_no_key():
    """AND THIS MUST NOT BE SYNTHESISED. A shared fallback key would give every unidentified card
    one history, the gate would see a single card bursting constantly, and it would fire on all
    of them."""
    assert CI.card_key({}) == ""
    assert CI.card_key({"bin": "411111", "cardholder_name": "Jane Roe"}) == ""


def test_last4_is_not_the_key():
    """Last four is safe to display and useless as an identifier: many cards share any four
    digits. Using it as the key would merge unrelated cards into one history."""
    assert CI.last4({"pan": PAN}) == "1111"
    assert CI.last4({"pan": PAN}) not in CI.card_key({"pan": PAN})


# ------------------------------------------------------------------ the trailing window

def test_the_window_counts_only_this_card():
    s = _store_with("cardA", 4)
    for i in range(9):
        s.log_decision(subject_ref=f"other_{i}", entity_id=eid("card", "cardB"), action="ALLOW",
                       module="card_scorer", features={"amount": 100.0},
                       ts=_iso(NOW - timedelta(hours=1)))
    try:
        assert sequence_view(s, eid("card", "cardA"), 100.0, now=NOW)["seq_count_24h"] == 4.0
    finally:
        s.close()


def test_authorizations_older_than_the_window_are_not_counted():
    s = _store_with("cardA", 3, hours_ago=30)
    try:
        v = sequence_view(s, eid("card", "cardA"), 100.0, now=NOW)
        assert v["seq_count_24h"] == 0.0 and v["card_known"] is False
    finally:
        s.close()


def test_the_window_is_strictly_prior_and_excludes_the_present_authorization():
    """The row being scored must not appear in its own history. Obvious while the write blocked
    the response; not obvious once the card write moved off the deadline path."""
    s = _store_with("cardA", 2)
    s.log_decision(subject_ref="arn_now", entity_id=eid("card", "cardA"), action="ALLOW",
                   module="card_scorer", features={"amount": 100.0}, ts=_iso(NOW))
    try:
        assert sequence_view(s, eid("card", "cardA"), 100.0, now=NOW)["seq_count_24h"] == 2.0, (
            "an authorization at exactly `now` was counted in its own trailing window")
    finally:
        s.close()


def test_escalation_is_measured_against_the_card_s_own_recent_amounts():
    s = _store_with("cardA", 5, amount=50.0)
    try:
        v = sequence_view(s, eid("card", "cardA"), 500.0, now=NOW)
        assert abs(v["seq_amount_vs_recent"] - 10.0) < 1e-9
    finally:
        s.close()


def test_an_unknown_card_reads_as_no_history_not_as_a_quiet_one():
    """Zero burst and neutral escalation is a CLAIM about a card. 'We have never seen this card'
    is a different statement and must not be dressed up as the first."""
    s = _store_with("cardA", 3)
    try:
        v = sequence_view(s, eid("card", "never_seen"), 100.0, now=NOW)
        assert v["card_known"] is False and v["seq_count_24h"] == 0.0
        assert v["seq_amount_vs_recent"] == 1.0
    finally:
        s.close()


def test_a_zero_value_history_does_not_divide_by_zero():
    s = _store_with("cardA", 3, amount=0.0)
    try:
        assert sequence_view(s, eid("card", "cardA"), 500.0, now=NOW)["seq_amount_vs_recent"] == 1.0
    finally:
        s.close()


def test_a_broken_substrate_degrades_to_no_history_rather_than_failing_the_auth():
    """This runs inside a network window. A database problem must cost the gate, never the
    authorization."""
    class Broken:
        class _c:
            @staticmethod
            def execute(*a, **k):
                raise RuntimeError("db gone")
        _conn = _c()
    v = sequence_view(Broken(), eid("card", "x"), 100.0, now=NOW)
    assert v["card_known"] is False and v["seq_count_24h"] == 0.0
    assert sequence_view(None, "x", 100.0)["card_known"] is False


# ------------------------------------------------------------------ the deadline

def test_the_lookup_is_fast_enough_for_an_authorization_window():
    """MEASURED, not assumed. The gate sits in front of a network deadline, so a lookup that is
    correct but slow is a timeout, and a timeout is the network deciding instead of us."""
    import time
    s = _store_with("cardA", 150)
    try:
        sequence_view(s, eid("card", "cardA"), 100.0, now=NOW)   # warm
        t0 = time.perf_counter()
        for _ in range(200):
            sequence_view(s, eid("card", "cardA"), 100.0, now=NOW)
        per_call_ms = (time.perf_counter() - t0) * 1000 / 200
        assert per_call_ms < 15.0, f"{per_call_ms:.2f}ms per lookup is too slow for the window"
        print(f"        (measured {per_call_ms:.3f}ms per lookup over a 150-row window)")
    finally:
        s.close()


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
