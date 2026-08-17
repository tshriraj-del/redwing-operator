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

# The card rail's equivalent. `card` comes from the salted key rather than a column, so it is
# resolved separately below; these are the plain columns.
_CARD_ENTITY_COLS = (("merchant", "merchant_id"), ("device", "device_id"), ("user", "user_id"))

# NOT an entity, on purpose. See record_card_authorization.
_CARD_CLASS_FIELDS = ("bin", "mcc_code", "entry_mode", "card_type")


def record_card_authorization(store, msg: dict, decision: dict, report: dict | None = None):
    """Write a card authorization into the backbone as ENTITIES plus a linking EVENT.

    WHY THIS EXISTS. MEASURED 2026-08-15 on the live store: `entities` held recipient 9,818,
    user 2,005, device 1,891, and card 0, merchant 0. The card path wrote its key into
    `decisions.entity_id` and stopped, so a card authorization was an isolated row. No node, no
    edge, nothing for a graph query, a campaign detector, or an investigator to traverse. The
    card rail could not participate in actor detection because the substrate it would read was
    never written.

    WHY IT IS THE PRIORITY. Per-typology recall on the challenge ledger: the novelty gate catches
    90.0% of invoice redirection and 1.6% of card testing, while the model scores card testing at
    0.0008. Nothing sees it, and that is structural: a card-testing authorization is unremarkable
    in isolation (small amount, ordinary merchant, a card with no history because each card is
    used once), so no per-transaction detector can flag it. The signal exists only as MERCHANT
    FAN-IN, which needs the merchant to be a node with edges to those cards. That is this
    function.

    THE BIN IS AN ATTRIBUTE, NOT AN ENTITY, and the distinction is load-bearing. Millions of
    cards share a BIN, so a `bin:` node would be a supernode every card authorization links to,
    and traversal from any card would reach most of the graph in two hops. Kept as an attribute
    on the card entity, "distinct BINs at this merchant in the last hour" is still answerable by
    aggregating over the merchant's linked cards, with no degree explosion.

    NEVER RAISES, and never silently succeeds either. A substrate failure must cost the backbone
    and not the authorization, but `report` receives `{"ok": bool, "error": str}` so a caller can
    tell a write that FAILED from one that had nothing to write. That distinction is the
    silent-degradation defect class this codebase hit three times in one day.

    Returns the entity ids linked, or [] on failure.
    """
    if report is not None:
        report.clear()
        report.update({"ok": False, "entities": 0})
    if store is None:
        if report is not None:
            report["error"] = "no store"
        return []

    try:
        from .card_identity import card_key
    except Exception:                                             # noqa: BLE001
        card_key = lambda _m: ""                                  # noqa: E731

    try:
        inst = str(msg.get("institution_id", "") or "")
        ids = []

        ckey = card_key(msg)
        if ckey:
            # The card's class fields ride as ATTRIBUTES, which is what keeps the BIN out of the
            # node set while leaving it aggregatable.
            attrs = {k: msg.get(k) for k in _CARD_CLASS_FIELDS if msg.get(k) not in (None, "")}
            cid = eid("card", ckey)
            store.upsert_entity(cid, "card", institution_id=inst, attributes=attrs)
            ids.append(cid)

        for kind, col in _CARD_ENTITY_COLS:
            raw = str(msg.get(col, "") or "").strip()
            if raw and raw.lower() != "nan":
                e = eid(kind, raw)
                store.upsert_entity(e, kind, institution_id=inst)
                ids.append(e)

        if not ids:
            if report is not None:
                report.update({"ok": True, "entities": 0, "note": "nothing identifiable to link"})
            return []

        # THE EDGE, which is the actual deliverable. Entities without a linking event are
        # isolated nodes and answer no question that a plain column could not.
        ref = str(msg.get("arn") or msg.get("rrn") or msg.get("transaction_id") or "")
        store.append_event(
            "card_authorization", entities=ids, institution_id=inst,
            event_id=(f"auth:{ref}" if ref else None),
            payload={"amount": msg.get("amount"), "mcc_code": msg.get("mcc_code"),
                     "entry_mode": msg.get("entry_mode"), "bin": msg.get("bin")},
            derived={"score": decision.get("score"),
                     "action": decision.get("action"),
                     "expected_liability": decision.get("expected_liability")},
        )
        if report is not None:
            report.update({"ok": True, "entities": len(ids)})
        return ids
    except Exception as e:                                        # noqa: BLE001
        if report is not None:
            report.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"})
        return []


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


def row_from_backbone(store, transaction_id: str):
    """Rebuild a transaction row from the backbone, or None if it was never recorded.

    Everything the ingestion pipeline brings in (a file drop, a webhook push, a polled
    source table, /ingest, /stream/publish) is scored and persisted here, but it never
    enters the historical dataset. Without this, an analyst cannot open any of it as a
    case: the case file would only ever resolve rows that shipped with the CSV. This
    reconstructs enough of the original row (amount, rail, typology, and the party ids
    from the linked entities) for the case assembler to re-score and build the file."""
    if store is None or not transaction_id:
        return None
    try:
        ev = store.get_event(str(transaction_id))
        if ev is None or ev.event_type != "transaction":
            return None
        row = {
            "transaction_id": str(transaction_id),
            "amount": ev.payload.get("amount"),
            "payment_rail": ev.payload.get("rail"),
            "fraud_typology": ev.payload.get("typology") or "",
            "institution_id": ev.institution_id or "",
            "is_fraud": bool(ev.derived.get("is_fraud")),
        }
        for ent in ev.entities or []:
            kind, _, raw = str(ent).partition(":")
            for k, col in _ENTITY_COLS:
                if kind == k and raw:
                    row[col] = raw
        return row
    except Exception:
        return None
