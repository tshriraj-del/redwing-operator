"""
core/loop.py - close the analyst-feedback loop AND feed the training substrate.

The existing feedback path (feedback.py) does the real-time online update: a confirmed
disposition raises the recipient's reputation so the next payment scores higher. This
module does two more things:

  1. It leaves a durable, inspectable trail and returns a RECEIPT, so the analyst sees what
     their one decision accomplished (the compounding, made visible).

  2. It feeds the LABELING SUBSTRATE (store decisions + labels tables), which is what turns
     the heuristic actor layer into a system that assembles the dataset to train its own
     replacement. Two entry points:

       record_decision(...)  logs a scored subject at decision time with its point-in-time
                             feature snapshot, including SHADOW (scored-but-not-enforced)
                             decisions so the training set is not censored by our own blocks.
                             Optionally writes the module's own read as a low-confidence
                             HEURISTIC label (the weak-supervision bootstrap).

       close_loop(...)       on a disposition, writes the OUTCOME label (was it fraud) and,
                             when the analyst supplies it, the structured INTENT labels
                             (motive, witting-ness, scam stage) as high-confidence, analyst-
                             sourced ground truth. Intent is the label space the historical
                             ledger never captured and the psychological modules need.

Pure Python over the store - unit-testable without the ML stack.
"""

from __future__ import annotations

from .store import eid


# Provenance normalisation: the human dispositions arrive tagged 'investigator'; the label
# store speaks in LABEL_SOURCES. Anything already a known source is passed through.
def _outcome_source(source: str) -> str:
    return "analyst" if source in ("investigator", "analyst", "") else source


def record_decision(store, subject_ref, entity_id: str = "", action: str = "",
                    module: str = "model", score=None, expected_liability=None,
                    features=None, rationale=None, shadow: bool = False,
                    institution_id: str = "", model_version: str = "",
                    policy_version: str = "", heuristic_labels=None,
                    decision_id=None) -> dict:
    """Log one scored subject into the training substrate at decision time.

    `features` is the exact snapshot used to decide (point-in-time correctness). Set
    `shadow=True` for a scored-but-not-enforced decision so the label distribution stays
    uncensored. `heuristic_labels` is an optional list of the module's own reads to record
    as weak (source='heuristic') labels, e.g.
    [{"space": "intent", "key": "motive", "value": "survival", "confidence": 0.3}]."""
    if store is None:
        return {}
    did = store.log_decision(
        subject_ref=str(subject_ref or ""), entity_id=entity_id, action=action,
        module=module, score=score, expected_liability=expected_liability,
        features=features, rationale=rationale, shadow=shadow,
        institution_id=institution_id, model_version=model_version,
        policy_version=policy_version, decision_id=decision_id,
    )
    written = 0
    for hl in (heuristic_labels or []):
        if not hl.get("key"):
            continue
        store.add_label(
            label_space=hl.get("space", "intent"), label_key=hl["key"],
            label_value=hl.get("value"), source="heuristic",
            confidence=float(hl.get("confidence", 0.3)),
            decision_id=did, subject_ref=str(subject_ref or ""), entity_id=entity_id,
            annotator=module,
        )
        written += 1
    return {"decision_id": did, "heuristic_labels_written": written}


def close_loop(store, transaction_id: str, recipient_id: str, label: str,
               is_fraud, rep_rate=None, source: str = "investigator",
               intent=None, effective_ts: str = "") -> dict:
    """Mirror one disposition onto the backbone and return the analyst receipt.
    `is_fraud` is True / False / None (unknown). `rep_rate`, when provided, is the
    authoritative empirical-Bayes fraud rate from the live reputation layer. `intent`, when
    provided, is a dict of structured adjudicated intent labels (e.g. {"motive": "survival",
    "witting_role": "unwitting", "scam_stage": "extraction"}) written as analyst ground truth
    into the label store. `effective_ts` is when the labeled fact actually became true (for
    label-latency measurement)."""
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

    # 4. feed the labeling substrate: the OUTCOME label, and any adjudicated INTENT labels.
    #    Keyed by the transaction id so add_label links them to the point-in-time decision
    #    (if one was logged at score time). Revisions supersede rather than overwrite.
    outcome_written = False
    if is_fraud is not None:
        store.add_label(
            label_space="outcome", label_key="is_fraud", label_value=int(is_fraud),
            source=_outcome_source(source), confidence=0.95,
            subject_ref=tid, entity_id=rec_id, effective_ts=effective_ts, annotator=source,
        )
        outcome_written = True

    intent_written = []
    for k, v in (intent or {}).items():
        if v in (None, ""):
            continue
        store.add_label(
            label_space="intent", label_key=str(k), label_value=v,
            source="analyst", confidence=0.9,
            subject_ref=tid, entity_id=rec_id, effective_ts=effective_ts, annotator=source,
        )
        intent_written.append(str(k))

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
        # the labeling substrate receipt: what ground truth this disposition just captured
        "labeling": {
            "outcome_label":  (int(is_fraud) if is_fraud is not None else None),
            "outcome_written": outcome_written,
            "intent_labels":  intent_written,
            "substrate":      store.labeling_stats(),
        },
    }
