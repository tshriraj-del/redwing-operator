"""
core/loop.py - close the analyst-feedback loop on the backbone (Phase 1, WS2).

The existing feedback path (feedback.py) already does the real-time online update:
a confirmed-fraud disposition raises the recipient's reputation in the live feature
engine, so the very next payment to that counterparty scores higher. What it does
NOT do is leave a durable, inspectable trail or tell the analyst what their one
decision just accomplished.

This closes that gap. Given a disposition, it:
  1. emits a `disposition` event (what the analyst decided),
  2. moves the recipient ENTITY's reputation in the durable store,
  3. emits a `feedback` event (the labeled example) and, for a fraud/legit label,
     a `model_update` event onto the retrain queue,
  4. returns a RECEIPT: how many pending payments to this recipient exist, their
     total exposure, the reputation before/after, and the retrain-queue depth.

The receipt is the point. It makes the compounding visible: one human decision,
several downstream effects, shown back to the person who made it. That is the
property no incumbent surfaces and the whole reason an ecosystem beats a dashboard.

Pure Python over the store - unit-testable without the ML stack.
"""

from __future__ import annotations

from .store import eid


def close_loop(store, transaction_id: str, recipient_id: str, label: str,
               is_fraud, rep_rate=None, source: str = "investigator") -> dict:
    """Mirror one disposition onto the backbone and return the analyst receipt.
    `is_fraud` is True / False / None (unknown). `rep_rate`, when provided, is the
    authoritative empirical-Bayes fraud rate from the live reputation layer, so the
    store agrees with the feature engine rather than recomputing the math."""
    if store is None:
        return {}

    tid     = str(transaction_id or "")
    rid_raw = str(recipient_id or "").strip()
    rec_id  = eid("recipient", rid_raw) if rid_raw and rid_raw != "nan" else ""
    ents    = [rec_id] if rec_id else []

    # 1. disposition event (idempotent per transaction)
    store.append_event(
        "disposition", entities=ents,
        event_id=(f"disp:{tid}" if tid else None),
        payload={"label": label, "source": source, "recipient_id": rid_raw},
        derived={"is_fraud": (int(is_fraud) if is_fraud is not None else None)},
    )

    # 2. move the recipient entity's reputation in the durable store
    before = after = None
    pending = 0
    exposure = 0.0
    liability = 0.0
    if rec_id:
        ent = store.get_entity(rec_id)
        if ent is None:
            store.upsert_entity(rec_id, "recipient", reputation={"tx": 0, "fraud": 0})
            before = {"tx": 0, "fraud": 0}
        else:
            before = dict(ent.reputation)
        patch = {}
        if is_fraud is not None:
            patch["tx"]    = int(before.get("tx", 0)) + 1
            patch["fraud"] = int(before.get("fraud", 0)) + (1 if is_fraud else 0)
        if rep_rate is not None:
            patch["fraud_rate"] = round(float(rep_rate), 6)
        after = store.update_reputation(rec_id, patch).reputation if patch else before

        # pending payments to this recipient = transaction events on the backbone.
        # One aggregate query (not a materialised scan) so /feedback stays sub-second
        # even against an 880k-event store.
        summ = store.entity_event_summary(rec_id, "transaction")
        pending   = summ["count"]
        exposure  = summ["exposure"]
        liability = summ["liability"]   # WS4: reimbursement dollars this payee exposes

    # 3. feedback event + retrain-queue model_update
    store.append_event(
        "feedback", entities=ents, event_id=(f"fb:{tid}" if tid else None),
        payload={"label": label, "recipient_id": rid_raw},
        derived={"is_fraud": (int(is_fraud) if is_fraud is not None else None)},
    )
    if is_fraud is not None:
        store.append_event(
            "model_update", entities=ents,
            payload={"reason": "analyst_label", "transaction_id": tid},
            derived={"is_fraud": int(is_fraud)},
        )

    labels_queued = store.count_events("model_update")

    return {
        "recipient_id":      rid_raw,
        "reputation_before": before,
        "reputation_after":  after,
        "pending_payments":  pending,
        "exposure_usd":      exposure,
        "liability_at_risk": liability,
        "labels_queued":     labels_queued,
        "events_emitted":    ["disposition", "feedback"] + (["model_update"] if is_fraud is not None else []),
    }
