"""
core/replay.py - fill the training substrate from REAL labeled transactions.

Phase 2, WS6. The substrate (point-in-time decisions, two label spaces, monitored holdout,
graduation gate, trainer) was fully built in Phase 1 and then sat empty: 322 decisions, 7
labels, 0.6% outcome coverage. The graduation gate needs 50 gold labels and 30 heuristic/gold
pairs before it will report anything but noise, so it was structurally incapable of firing.
This module runs the loop at volume against real labeled data so the gate has something to
decide on.

What makes this a simulation of a fraud system rather than a simulation of omniscience:

  1. The dataset knows every outcome. Production does not. A decision that we HOLD never
     reveals whether it was fraud, so a held decision here is recorded and left DELIBERATELY
     UNLABELLED. Only ALLOWed decisions (including holdout releases) receive an outcome label.
     Censoring the labels is the entire point; a replay that labels everything would produce
     a flattering number that means nothing.

  2. Features are the point-in-time snapshot, taken from the source row as it was, and stored
     on the decision. Nothing is recomputed later. Note that only 13 of the model's 23
     features exist in the source data; the other 10 are exactly the ones the skew audit found
     had no reproducible definition at serving time, so they are legitimately unavailable here.

  3. The baseline rule is fixed in advance (see BASELINE_RULE) and was chosen as the STRONGEST
     of several domain-motivated candidates measured on the full dataset, not the weakest. The
     model has to beat a rule given its best shot. Choosing the baseline after seeing which
     one the model beats would make the whole exercise circular.

  4. Outcome labels only. Intent (motive) is never labelled here: motive does not fall out of
     a ledger, it requires a human adjudicator, and grading the motive heuristic against the
     dataset's typology field would be grading it against its own answer key.

The gold source is "confirmed_loss", which is what a real outcome label is: the ledger telling
you after the fact that money was lost.
"""

from __future__ import annotations

from .holdout import holdout_decision, holdout_rationale
from .liability import expected_liability
from .loop import record_decision

# The 13 model features that genuinely exist in the source data, point-in-time.
REPLAY_FEATURES = (
    "amount_zscore", "amount_vs_max", "hour_risk", "rail_risk",
    "recipient_familiarity", "device_familiarity",
    "velocity_1h", "velocity_4h", "velocity_24h",
    "new_recipient_streak", "is_crypto", "is_instant_rail", "is_p2p",
)

# Fixed in advance. A two-signal rule an analyst would plausibly write for the scam /
# irrevocable-rail wedge: a payment large relative to the customer's own history, going to a
# recipient they have no history with. Measured on all 880,726 rows it fires 3,479 times at
# 0.340 precision and 0.208 recall (f1 0.258), the best f1 of the domain candidates tried.
BASELINE_RULE = {
    "name": "large_vs_history_unfamiliar_recipient",
    "description": "amount_vs_max > 0.8 AND recipient_familiarity < 0.2",
    "measured_on_full_dataset": {"fires": 3479, "precision": 0.340, "recall": 0.208, "f1": 0.258},
}


def baseline_rule(features: dict) -> bool:
    """The hand-written rule the trained model has to beat."""
    return (float(features.get("amount_vs_max", 0.0) or 0.0) > 0.8
            and float(features.get("recipient_familiarity", 1.0) or 0.0) < 0.2)


def _as_bool(v) -> bool:
    """Source data carries booleans as 'True'/'False' text in some columns and 0/1 in others."""
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    if s in ("true", "t", "yes", "y"):
        return True
    if s in ("false", "f", "no", "n", "", "none"):
        return False
    try:
        return float(s) != 0.0
    except (TypeError, ValueError):
        return False


def _snapshot(row: dict) -> dict:
    """Point-in-time feature snapshot: only what was actually observable on this row."""
    out = {}
    for f in REPLAY_FEATURES:
        v = row.get(f)
        if v is None or v == "":
            continue
        try:
            out[f] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def replay_row(store, row: dict, holdout_config: dict | None = None) -> dict:
    """Run ONE transaction through the decision path and record what a production system
    would legitimately know afterwards. Returns a small dict describing what happened."""
    tid = str(row.get("transaction_id") or "")
    features = _snapshot(row)
    if not tid or not features:
        return {"skipped": True}

    amount = float(row.get("amount", 0.0) or 0.0)
    rail = str(row.get("payment_rail", "") or "")
    fired = baseline_rule(features)
    proposed = "HOLD" if fired else "ALLOW"

    # Price the decision before the holdout sees it: the liability ceiling is what stops the
    # holdout releasing a case too expensive to be worth the counterfactual.
    liab = expected_liability(0.9 if fired else 0.1, amount, typology="", rail=rail)

    ho = holdout_decision(tid, proposed, liab, config=holdout_config)
    enforced = ho["enforced_action"]

    record_decision(
        store, subject_ref=tid,
        entity_id=f"user:{row.get('user_id', 'unknown')}",
        action=enforced, module="rule",
        score=(0.9 if fired else 0.1), expected_liability=liab,
        features=features,
        rationale={**holdout_rationale(ho), "rule": BASELINE_RULE["name"], "fired": fired},
        heuristic_labels=[{"space": "outcome", "key": "is_fraud",
                           "value": str(bool(fired)), "confidence": 0.3}],
        decision_id=f"replay:{tid}",
    )

    # THE CENSORING. Only an allowed transaction reveals its outcome. A held one does not, and
    # gets no label, which is what makes the resulting training set honestly incomplete.
    observed = enforced == "ALLOW"
    if observed:
        store.add_label(
            "outcome", "is_fraud", str(_as_bool(row.get("is_fraud"))),
            source="confirmed_loss", confidence=1.0,
            # keyed BOTH ways on purpose: decision_id is what training joins on, subject_ref
            # is how a late-arriving chargeback would actually reach us (by transaction id)
            decision_id=f"replay:{tid}", subject_ref=tid,
            notes="ledger outcome observed because the transaction was allowed",
        )

    return {"skipped": False, "fired": fired, "enforced": enforced,
            "observed": observed, "released": bool(ho["release"]),
            "is_fraud": _as_bool(row.get("is_fraud"))}


def replay(store, rows, holdout_config: dict | None = None, limit: int | None = None) -> dict:
    """Replay an iterable of transaction dicts. Returns a summary of what the substrate saw,
    including how much of it is censored, which is the number that makes the rest credible."""
    n = fired = held = allowed = released = observed = labeled_fraud = skipped = 0
    for row in rows:
        if limit is not None and n >= limit:
            break
        r = replay_row(store, row, holdout_config=holdout_config)
        if r.get("skipped"):
            skipped += 1
            continue
        n += 1
        fired += r["fired"]
        released += r["released"]
        if r["enforced"] == "ALLOW":
            allowed += 1
        else:
            held += 1
        if r["observed"]:
            observed += 1
            labeled_fraud += r["is_fraud"]

    return {
        "replayed": n,
        "skipped": skipped,
        "rule": BASELINE_RULE["name"],
        "rule_fired": fired,
        "enforced_hold": held,
        "allowed": allowed,
        "holdout_released": released,
        "observed_labeled": observed,
        "censored": held,
        "censored_fraction": round(held / n, 4) if n else 0.0,
        "labeled_fraud": labeled_fraud,
        "labeled_fraud_rate": round(labeled_fraud / observed, 5) if observed else 0.0,
        "note": "held decisions are recorded but deliberately unlabelled; their outcome is "
                "censored by our own enforcement, exactly as in production",
    }
