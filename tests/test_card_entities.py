"""
Tests that a card authorization writes ENTITIES and EDGES into the backbone.

WHY THIS FILE EXISTS. MEASURED 2026-08-15 on the live store: `entities` held recipient 9,818,
user 2,005, device 1,891, and **card 0, merchant 0, bin 0**. The push path
(`record_scored_event`) upserts three entities per transaction and appends a multi-entity event,
which is what creates edges. The card path wrote `eid("card", ckey)` into `decisions.entity_id`
and stopped: no upsert, no merchant, no device, no linking event. The card rail was a graph
orphan, so campaign and actor detection could not run on it at all.

WHY THIS IS THE PRIORITY, and it is not "campaigns would be nice". Measured on the challenge
ledger, per-typology novelty-gate recall:

    invoice_redirection  90.0%      card_testing_bot   1.6%
    mule_account         52.3%      first_party_dispute 2.9%

Nothing in the system sees card testing: the model scores it 0.0008 and the gate escalates 9 of
577. That is structural, not a threshold. A card-testing authorization is unremarkable in
isolation (small amount, ordinary merchant, a card with no history because each card is used
once), so a per-transaction detector has nothing to flag. The signal exists ONLY in aggregate:
one merchant, many first-use cards, minutes apart. Which requires the merchant to be an entity
with edges to those cards, which is what this writes.

THE BIN IS DELIBERATELY NOT AN ENTITY. Millions of cards share a BIN, so making it a node would
create a supernode that every card transaction links to, wrecking traversal. It travels as an
ATTRIBUTE on the card entity, so "distinct BINs at this merchant in the last hour" is still
answerable by aggregating over the merchant's linked cards, without the degree explosion.

Runs under pytest or standalone (python3 tests/test_card_entities.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_RECOVERY_SECRET", "card-entities-test")
os.environ.setdefault("REDWING_CARD_SALT", "card-entities-salt")

from core.card_identity import card_key            # noqa: E402
from core.record import record_card_authorization  # noqa: E402
from core.store import Store, eid                  # noqa: E402

MSG = {"amount": 900.0, "merchant_name": "Acme Supplies", "merchant_id": "m_acme",
       "cardholder_name": "Jane Roe", "entry_mode": "ecom", "mcc_code": 5999,
       "bin": "400000", "arn": "74537501234567890123456", "card_token": "tok_entities_1",
       "device_id": "dev_ent_1", "user_id": "u_ent_1"}

DEC = {"action": "ALLOW", "score": 0.42, "expected_liability": 12.5}


def _store():
    return Store(os.path.join(tempfile.mkdtemp(), "ent.db"))


def _types(s):
    return {r[0]: r[1] for r in
            s._conn.execute("SELECT type, COUNT(*) FROM entities GROUP BY type").fetchall()}


# ── the entities that did not exist ──────────────────────────────────────────

def test_a_card_authorization_creates_a_card_entity():
    """The whole point. `decisions.entity_id` already carried the card key, but nothing ever
    upserted it into `entities`, so the card was invisible to every graph query."""
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC)
        assert _types(s).get("card", 0) == 1
    finally:
        s.close()


def test_the_merchant_becomes_an_entity():
    """THE one that matters for card testing. Fan-in of distinct cards at one merchant is the
    only place that signal exists, and it needs the merchant to be a node."""
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC)
        assert _types(s).get("merchant", 0) == 1
    finally:
        s.close()


def test_the_device_is_linked_when_the_message_carries_one():
    """CNP authorizations commonly do. This is the join key between a card-testing phase and the
    ATO that follows it, which is the campaign shape the whole rail exists to detect."""
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC)
        assert _types(s).get("device", 0) == 1
    finally:
        s.close()


def test_the_bin_is_an_attribute_and_not_an_entity():
    """Millions of cards share a BIN. As a node it is a supernode every card links to."""
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC)
        assert _types(s).get("bin", 0) == 0, "the BIN was written as an entity; it is a class"
    finally:
        s.close()


# ── the edges, which are the actual deliverable ──────────────────────────────

def test_the_authorization_links_its_entities_in_one_event():
    """Entities without a linking event are isolated nodes. The EDGE is the deliverable: it is
    what makes two authorizations on one merchant reachable from each other."""
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC)
        ids = record_card_authorization(s, {**MSG, "arn": "arn_2", "card_token": "tok_2"}, DEC)
        assert len(ids) >= 2, f"expected several linked entities, got {ids}"
        merchant = eid("merchant", "m_acme")
        rows = s._conn.execute(
            "SELECT COUNT(*) FROM event_entities WHERE entity_id = ?", (merchant,)).fetchone()
        assert rows[0] >= 2, "two authorizations at one merchant produced no shared edges"
    finally:
        s.close()


def test_card_testing_is_visible_as_merchant_fan_in():
    """THE measurement this whole change exists to enable. Twenty distinct cards, one merchant,
    one window. Invisible per-card (each card has exactly one authorization, so the sequence
    gate reads card_known=False), and unmistakable per-merchant."""
    s = _store()
    try:
        for i in range(20):
            record_card_authorization(
                s, {**MSG, "arn": f"arn_burst_{i}", "card_token": f"tok_burst_{i}"}, DEC)
        merchant = eid("merchant", "m_acme")
        distinct_cards = s._conn.execute(
            "SELECT COUNT(DISTINCT ee2.entity_id) FROM event_entities ee1 "
            "JOIN event_entities ee2 ON ee1.event_id = ee2.event_id "
            "WHERE ee1.entity_id = ? AND ee2.entity_id LIKE 'card:%'", (merchant,)).fetchone()[0]
        assert distinct_cards == 20, (
            f"merchant fan-in shows {distinct_cards} distinct cards, expected 20")
    finally:
        s.close()


def test_the_same_card_twice_is_one_entity():
    """Idempotent by construction, or a repeat customer looks like a fan-out."""
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC)
        record_card_authorization(s, {**MSG, "arn": "arn_again"}, DEC)
        assert _types(s).get("card", 0) == 1
    finally:
        s.close()


# ── it must not break the authorization ──────────────────────────────────────

def test_a_message_with_no_card_identifier_still_records_what_it_can():
    """No token and no PAN is a real case. The merchant is still worth linking."""
    s = _store()
    try:
        msg = {k: v for k, v in MSG.items() if k != "card_token"}
        record_card_authorization(s, msg, DEC)
        t = _types(s)
        assert t.get("card", 0) == 0 and t.get("merchant", 0) == 1
    finally:
        s.close()


def test_a_broken_store_does_not_raise():
    """This runs behind a network deadline. A substrate failure must cost the backbone, never
    the authorization."""
    class Broken:
        def upsert_entity(self, *a, **k):
            raise RuntimeError("db gone")

        def append_event(self, *a, **k):
            raise RuntimeError("db gone")
    assert record_card_authorization(Broken(), MSG, DEC) == []
    assert record_card_authorization(None, MSG, DEC) == []


def test_a_failed_write_is_reported_rather_than_swallowed():
    """The silent-degradation class, measured three times in this codebase on 2026-08-15. A
    write that did not happen must be distinguishable from one that had nothing to write."""
    class Broken:
        def upsert_entity(self, *a, **k):
            raise RuntimeError("db gone")

        def append_event(self, *a, **k):
            raise RuntimeError("db gone")
    out = {}
    record_card_authorization(Broken(), MSG, DEC, report=out)
    assert out.get("ok") is False and out.get("error"), (
        f"a failed backbone write reported nothing: {out}")
    ok = {}
    s = _store()
    try:
        record_card_authorization(s, MSG, DEC, report=ok)
        assert ok.get("ok") is True
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
