"""
core/record.py - write a scored transaction into the backbone (Phase 1, WS1).

Kept out of main.py deliberately: main.py imports the ML stack (numpy/xgboost),
so anything living there is untestable without it. This function is pure Python
over the store, so the score-path integration can be tested on its own.

Contract:
  * Idempotent by transaction_id. Re-scoring the same transaction (historical
    replay, the agent loop) updates the same rows instead of duplicating them,
    so the store stabilises in size.
  * Reputation counts are NOT bumped here. Those come from history (the seed
    importer) and from confirmed analyst dispositions (the feedback loop, WS2),
    so replay can never inflate a recipient's fraud rate.
  * Never raises. The backbone is a non-critical side-effect of scoring; a store
    failure must not break the score path.
"""

from __future__ import annotations

from .store import Store, eid

_ENTITY_COLS = (("user", "user_id"), ("device", "device_id"), ("recipient", "recipient_id"))


def record_scored_event(store, event: dict, row: dict) -> list:
    """Persist one scored transaction as entities + a transaction event (+ an alert
    event when it fired). Returns the entity ids touched (for callers/tests); [] on
    any failure or when the store is absent."""
    if store is None:
        return []
    try:
        tid  = str(event.get("transaction_id", "") or "")
        inst = str(row.get("institution_id", "") or "")
        ids  = []
        for kind, col in _ENTITY_COLS:
            raw = str(row.get(col, "") or "").strip()
            if raw and raw != "nan":
                store.upsert_entity(eid(kind, raw), kind, institution_id=inst)
                ids.append(eid(kind, raw))
        store.append_event(
            "transaction", entities=ids, institution_id=inst,
            event_id=tid or None,
            payload={"amount": event.get("amount"), "rail": event.get("rail"),
                     "typology": str(row.get("fraud_typology", "") or "")},
            derived={"ml_score": event.get("ml_score"),
                     "combined_score": event.get("combined_score"),
                     "is_alert": bool(event.get("is_alert")),
                     "expected_liability": event.get("expected_liability"),
                     "is_fraud": int(bool(row.get("is_fraud", False)))},
        )
        if event.get("is_alert"):
            store.append_event(
                "alert", entities=ids, institution_id=inst,
                event_id=(f"alert:{tid}" if tid else None),
                payload={"amount": event.get("amount"), "rail": event.get("rail")},
                derived={"combined_score": event.get("combined_score"),
                         "expected_liability": event.get("expected_liability")})
        return ids
    except Exception:
        return []   # backbone is a non-critical side-effect; scoring must never break on it
