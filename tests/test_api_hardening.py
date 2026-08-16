"""
Security hardening: the three data-exposure holes found in the 2026-08-15 pre-assessment.

WHY THIS FILE EXISTS. An external offensive-security run is imminent, and a pre-assessment found
that the API's protections were shaped the same way its authentication was: correct when
configured, absent by default. These are the three that leak DATA rather than merely allowing
traffic, so they are guarded together.

  A SOURCE CONNECTOR MUST NOT READ ARBITRARY FILES. `/connectors/db/poll` took `db_path` from the
  request body with no validation, opened it, and republished its rows through the ingestion
  pipeline where they are readable from `/monitor/stream` and the backbone. That is a file-read
  primitive aimed at whatever the process can reach, and what it could reach was the 1.3GB
  decision store. NOTE THE TRAP: the obvious allowlist root is MODELS_DIR, which is the directory
  redwing.db lives in, so defaulting to it would leave the exploit fully intact.

  AN UNSALTED PAN HASH IS A REVERSIBLE PAN. `salt_configured()` existed, was tested, and had ZERO
  production callers. Nothing stopped the system hashing PANs with an empty salt and writing them
  to the substrate as a global constant function of the card number, recoverable from a leaked
  database in hours. Tokens are exempt: a network token is already a surrogate and is safe
  unsalted, which is why the fix must distinguish them rather than refusing both.

  THE PAN MUST DIE AT THE INGESTION BOUNDARY. `/authorize` was clean because `card_message.
  normalise` allowlists its output. Every OTHER surface funnels through `validate_event`, which
  does `ev = dict(raw)` to preserve passthrough fields, so a `pan` rode into stream.db in
  cleartext and back out through `/stream/dead_letter`.

Runs under pytest or standalone (python3 tests/test_api_hardening.py).
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_RECOVERY_SECRET", "hardening-test")

PAN = "4111111111111111"


# ── the source connector may not read arbitrary files ────────────────────────

def _connector_root_case(tmp_root, db_path):
    """Build a DBConnector under an explicit root and report whether read() yielded anything."""
    from core.connectors import DBConnector
    os.environ["REDWING_CONNECTOR_ROOT"] = str(tmp_root)
    c = DBConnector(connector_id="t", transport=None, checkpoints=None,
                    db_path=str(db_path), table="t")
    return list(c.read(0))


def _make_db(path, rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, amount REAL, secret TEXT)")
    for i in range(rows):
        conn.execute("INSERT INTO t (amount, secret) VALUES (?, ?)", (100.0 + i, "s3cret"))
    conn.commit()
    conn.close()
    return path


def test_a_source_inside_the_configured_root_is_readable():
    """The control must not break the feature it guards."""
    tmp = Path(tempfile.mkdtemp())
    inside = _make_db(tmp / "sources" / "src.db")
    try:
        assert len(_connector_root_case(tmp / "sources", inside)) == 3
    finally:
        os.environ.pop("REDWING_CONNECTOR_ROOT", None)


def test_a_source_outside_the_configured_root_is_refused():
    """THE exploit. `db_path` pointed at the decision store and its rows came back out through
    /monitor/stream four columns at a time."""
    tmp = Path(tempfile.mkdtemp())
    outside = _make_db(tmp / "elsewhere" / "victim.db")
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    try:
        assert _connector_root_case(tmp / "sources", outside) == [], (
            "a database outside the connector root was read")
    finally:
        os.environ.pop("REDWING_CONNECTOR_ROOT", None)


def test_traversal_out_of_the_root_is_refused():
    """Confinement has to survive `..`, or it is decoration."""
    tmp = Path(tempfile.mkdtemp())
    outside = _make_db(tmp / "elsewhere" / "victim.db")
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    sneaky = tmp / "sources" / ".." / "elsewhere" / "victim.db"
    try:
        assert _connector_root_case(tmp / "sources", sneaky) == [], (
            f"traversal escaped the root: {sneaky}")
        assert outside.exists()
    finally:
        os.environ.pop("REDWING_CONNECTOR_ROOT", None)


def test_a_symlink_out_of_the_root_is_refused():
    """A path that RESOLVES outside must be judged on where it resolves, not how it is spelled."""
    tmp = Path(tempfile.mkdtemp())
    outside = _make_db(tmp / "elsewhere" / "victim.db")
    (tmp / "sources").mkdir(parents=True, exist_ok=True)
    link = tmp / "sources" / "innocent.db"
    try:
        link.symlink_to(outside)
    except OSError:
        return                                            # symlinks unavailable; nothing to assert
    try:
        assert _connector_root_case(tmp / "sources", link) == [], "a symlink escaped the root"
    finally:
        os.environ.pop("REDWING_CONNECTOR_ROOT", None)


def test_with_no_root_configured_the_db_connector_reads_nothing():
    """FAIL CLOSED, and the default must not be MODELS_DIR. redwing.db lives in MODELS_DIR, so
    defaulting there would confine the connector to exactly the file it must never read."""
    tmp = Path(tempfile.mkdtemp())
    db = _make_db(tmp / "src.db")
    saved = os.environ.pop("REDWING_CONNECTOR_ROOT", None)
    try:
        from core.connectors import DBConnector
        c = DBConnector(connector_id="t", transport=None, checkpoints=None,
                        db_path=str(db), table="t")
        assert list(c.read(0)) == [], "the connector read a file with no root configured"
    finally:
        if saved is not None:
            os.environ["REDWING_CONNECTOR_ROOT"] = saved


def test_unmapped_source_columns_do_not_ride_into_the_event():
    """`_map` started from `dict(row)`, so EVERY column of the attacker's table survived into the
    published event regardless of field_map. Projection is what makes the field_map a whitelist
    rather than a rename."""
    from core.connectors import DBConnector
    c = DBConnector(connector_id="t", transport=None, checkpoints=None,
                    db_path="/nonexistent", table="t",
                    field_map={"amount": "amount"})
    mapped = c._map({"amount": 100.0, "secret": "s3cret", "another": "leak"})
    assert mapped.get("amount") == 100.0
    assert "secret" not in mapped and "another" not in mapped, (
        f"unmapped columns survived into the event: {mapped}")


# ── an unsalted PAN hash must never be written ───────────────────────────────

def test_an_unsalted_pan_yields_no_key_rather_than_a_reversible_one():
    """A bare sha256 of a PAN is a global constant function of the card number. With the BIN
    stored in the clear beside it, a leaked table is enumerable in hours. Returning "" is the
    path already handled everywhere ("no card identifier"), so the gate degrades instead of the
    substrate filling with reversible digests."""
    import importlib
    saved = os.environ.pop("REDWING_CARD_SALT", None)
    try:
        import core.card_identity as ci
        importlib.reload(ci)
        assert ci.salt_configured() is False
        assert ci.card_key({"pan": PAN}) == "", "an unsalted PAN produced a key"
    finally:
        if saved is not None:
            os.environ["REDWING_CARD_SALT"] = saved
        import core.card_identity as ci2
        importlib.reload(ci2)


def test_a_network_token_still_works_without_a_salt():
    """Tokens are exempt and the distinction matters: a token is already a per-merchant surrogate,
    not an enumerable 16-digit space. Refusing both would disable the card rail on a deployment
    that is doing the RIGHT thing."""
    import importlib
    saved = os.environ.pop("REDWING_CARD_SALT", None)
    try:
        import core.card_identity as ci
        importlib.reload(ci)
        assert ci.card_key({"card_token": "tok_abc"}), "a token was refused without a salt"
    finally:
        if saved is not None:
            os.environ["REDWING_CARD_SALT"] = saved
        import core.card_identity as ci2
        importlib.reload(ci2)


def test_a_salted_pan_still_yields_a_key():
    """The guard must not break the configured path."""
    import importlib
    os.environ["REDWING_CARD_SALT"] = "a-real-salt"
    import core.card_identity as ci
    importlib.reload(ci)
    k = ci.card_key({"pan": PAN})
    assert k and PAN not in k


# ── the PAN dies at the ingestion boundary ───────────────────────────────────

def _no_pan_anywhere(obj) -> bool:
    """No 12-to-19 digit run survives anywhere in the serialised structure."""
    import re
    return not re.search(r"\d{12,19}", json.dumps(obj, default=str))


def test_validate_event_strips_cardholder_data():
    """`/authorize` was clean because card_message.normalise allowlists. Every OTHER ingestion
    surface funnels through validate_event, which preserved passthrough fields, so a PAN reached
    stream.db in cleartext and came back out of /stream/dead_letter."""
    from core.ingest_schema import validate_event
    ev = validate_event({"transaction_id": "t1", "amount": 100.0, "payment_rail": "card",
                         "pan": PAN, "cvv": "123", "expiry": "12/29"}, source="test")
    body = ev.get("event") if isinstance(ev.get("event"), dict) else ev

    # Assert on the FIELDS and on the digits, not on the substring "pan". The card key is
    # deliberately prefixed `pan_` to record which input produced it, so a naive substring scan
    # flags a correctly-hashed identifier as a leaked card number. The thing that must not
    # survive is a readable PAN, not the letters p-a-n.
    for f in ("pan", "card_number", "primary_account_number", "cvv", "expiry"):
        assert f not in body, f"cardholder field {f!r} survived validation"
    assert _no_pan_anywhere(body), f"a 12-19 digit run survived: {json.dumps(body)[:300]}"


def test_every_pan_spelling_is_stripped():
    from core.ingest_schema import validate_event
    for field in ("pan", "card_number", "primary_account_number"):
        ev = validate_event({"transaction_id": "t", "amount": 1.0, field: PAN}, source="test")
        body = ev.get("event") if isinstance(ev.get("event"), dict) else ev
        assert _no_pan_anywhere(body), f"{field} survived validation"


def test_stripping_the_pan_leaves_a_usable_card_key():
    """Removing the PAN must not remove the card's IDENTITY, or the sequence gate goes blind on
    every ingested card event."""
    os.environ["REDWING_CARD_SALT"] = "a-real-salt"
    from core.ingest_schema import validate_event
    ev = validate_event({"transaction_id": "t", "amount": 1.0,
                         "payment_rail": "card", "pan": PAN}, source="test")
    body = ev.get("event") if isinstance(ev.get("event"), dict) else ev
    assert body.get("card_key"), "the card lost its identity along with its PAN"
    assert _no_pan_anywhere(body)


def test_a_non_card_event_is_untouched():
    """The strip must not disturb push traffic."""
    from core.ingest_schema import validate_event
    ev = validate_event({"transaction_id": "t", "amount": 100.0,
                         "payment_rail": "faster_payments", "user_id": "u1"}, source="test")
    body = ev.get("event") if isinstance(ev.get("event"), dict) else ev
    assert body.get("user_id") == "u1"
    assert float(body.get("amount")) == 100.0


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
