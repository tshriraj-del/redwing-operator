"""
core/card_history.py - the serving-side answer to "how has this card behaved lately".

WHY THIS IS A SEPARATE FILE FROM THE GATE. redwing-ml computed the sequence view by streaming a
ledger in timestamp order, which is correct for TRAINING and impossible at serving: a live
authorization has no future rows to stream and must ask the substrate instead. This is the same
view, derived the other way, and the two must agree or the gate is being trained on one thing and
served another. The features it produces are named identically to the training ones for exactly
that reason.

STRICTLY PRIOR BY CONSTRUCTION. The query reads decisions with `ts < now`, so the authorization
being scored cannot be in its own history. That is trivially true at serving because the row has
not been written yet, but it is asserted rather than assumed, because the write moved off the
response path when the card path went durable and "not yet written" stopped being obvious.

THE DEADLINE IS THE DESIGN CONSTRAINT. A card authorization answers inside a network window, so
this does one indexed read and nothing else: no joins, no aggregation in Python over a wide scan,
no second query. `idx_dec_entity` already exists, so filing card decisions under
`eid("card", card_key)` makes the lookup an index seek that was free to obtain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Matches redwing-ml/card_sequence.py. If these drift the gate is served a different view from
# the one its thresholds were priced against.
RECENT_N = 5
_DAY_SECONDS = 86_400

# Above this many rows the trailing window is truncated. A card with thousands of authorizations
# in 24h is either a test harness or an incident, and neither is worth spending the deadline on.
MAX_WINDOW_ROWS = 200


def _parse(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def sequence_view(store, card_key: str, amount: float, now=None) -> dict:
    """The sequence features for one authorization, from this card's strictly-prior decisions.

    Returns the same keys `card_sequence.gate` reads at training time. An unknown card yields
    the explicit no-history view rather than zeros, because zero burst and zero escalation is a
    claim about a card, and "we have never seen this card" is not that claim.
    """
    empty = {"seq_count_24h": 0.0, "seq_amount_vs_recent": 1.0,
             "card_known": False, "window_rows": 0}
    if store is None or not card_key:
        return empty

    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(seconds=_DAY_SECONDS)).isoformat().replace("+00:00", "Z")
    cutoff = now.isoformat().replace("+00:00", "Z")

    try:
        # ONE indexed read. `ts < cutoff` is what makes the window strictly prior; without it a
        # concurrent write of this very authorization could join its own history.
        #
        # AMOUNT, NOT LIABILITY. The first version read `expected_liability` and compared this
        # authorization's amount against it, which is dollars-transacted over dollars-at-risk:
        # a units error producing a meaningless ratio. Training compares amount to amount, so
        # serving must, or the gate is served a view its thresholds were never priced against.
        #
        # json_extract keeps the parse in SQLite rather than looping json.loads in Python, which
        # matters because this sits inside a network window.
        rows = store._conn.execute(
            "SELECT ts, json_extract(features, '$.amount') AS amt FROM decisions "
            "WHERE entity_id = ? AND ts >= ? AND ts < ? "
            "ORDER BY ts DESC LIMIT ?",
            (card_key, since, cutoff, MAX_WINDOW_ROWS),
        ).fetchall()
    except Exception:                                             # noqa: BLE001
        # A substrate failure must not fail an authorization. No history reads as no evidence,
        # which leaves the model's score exactly as it was.
        return empty

    if not rows:
        return empty

    amounts = [float(r["amt"] or 0.0) for r in rows]
    recent = [a for a in amounts[:RECENT_N] if a > 0]
    mean_recent = sum(recent) / len(recent) if recent else 0.0

    return {
        "seq_count_24h": float(len(rows)),
        # Guarded against a zero baseline: a card whose recent authorizations were all zero-value
        # would otherwise divide by zero, and 1.0 is the honest neutral for "nothing to compare".
        "seq_amount_vs_recent": (float(amount) / mean_recent) if mean_recent > 0 else 1.0,
        "card_known": True,
        "window_rows": len(rows),
    }
