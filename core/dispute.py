"""
core/dispute.py - the card dispute rail, and the labels it is allowed to emit.

WHY THIS EXISTS. Card authorizations produced no labels at all: the card rail wrote nothing to
the substrate, so it had no outcome ledger entries, no holdout membership, and no way to measure
decay. This is the label pipeline for that rail, and card labels arrive through a structured
adversarial process that push labels do not have:

    chargeback -> representment -> pre-arbitration -> arbitration -> settled

Each stage has a network clock and a reason code. Most systems flatten all of that into a single
boolean the moment a chargeback lands. That is wrong in three separate ways, and each one is a
rule below.

RULE 1: A CHARGEBACK IS A CLAIM, NOT AN OUTCOME.
Labelling at initiation is the immature-window mistake wearing new clothes. The cardholder has
asserted something; nobody has adjudicated it. `derive_outcome` returns None until the dispute
reaches a terminal state, for the same reason `label_maturity` refuses to fit a curve on cohorts
that have not settled.

RULE 2: THE REASON-CODE FAMILY DECIDES WHICH LABEL SPACE IS EVEN IN PLAY.
"Merchandise not received" (13.1) is a service dispute. It is not fraud, the cardholder is not
claiming fraud, and writing it into `outcome.is_fraud` trains the model that late shipping is
fraud. Only the FRAUD family may ever produce a fraud label. Everything else is recorded and
deliberately withheld from the fraud label space.

RULE 3: A REPRESENTMENT THE ISSUER LOSES INVERTS THE CLAIM.
If the merchant contests with compelling evidence and wins, the network has ruled the transaction
was not fraud. That is a label CORRECTION, and it is exactly the disagreement case the outcome
ledger's `superseded_by` was built to carry. Systems that label at initiation never revisit it,
so their training data permanently contains every dispute the cardholder lost.

AND THE ONE THAT PAYS FOR ITSELF: 10.4 and 4863 are ambiguous BY CONSTRUCTION. Visa's own
definition of 10.4 covers true fraud, friendly fraud, and merchant error under one code. The
reason code alone cannot separate them; only the terminal state can. A fraud-family chargeback
that the issuer loses on compelling evidence is the canonical first-party abuse shape, which is
a different problem with a different action, and it is invisible to any binary label.

Pure stdlib, no store dependency, so this is testable without the ML stack like the rest of core/.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# ── the taxonomy ─────────────────────────────────────────────────────────────

FRAUD = "fraud"              # cardholder (or the network) asserts the transaction was not theirs
AUTHORIZATION = "authorization"   # the auth itself was invalid: declined, expired, never obtained
PROCESSING = "processing"    # a mechanical error: duplicate, wrong amount, wrong currency
CONSUMER = "consumer"        # a service dispute: not received, not as described, not cancelled

# network, family, and whether the code implies a card-present environment.
# `cardholder_asserted` is False where the dispute is raised by a monitoring PROGRAM rather than
# by a person, which matters because a program artifact is not evidence about this cardholder.
REASON_CODES = {
    # ── Visa, post-VCR ──
    "10.1": ("visa", FRAUD, True,  "EMV Liability Shift Counterfeit Fraud", True),
    "10.2": ("visa", FRAUD, True,  "EMV Liability Shift Non-Counterfeit Fraud", True),
    "10.3": ("visa", FRAUD, True,  "Other Fraud - Card Present Environment", True),
    "10.4": ("visa", FRAUD, False, "Other Fraud - Card Absent Environment", True),
    "10.5": ("visa", FRAUD, False, "Visa Fraud Monitoring Program", False),
    "11.1": ("visa", AUTHORIZATION, False, "Card Recovery Bulletin", True),
    "11.2": ("visa", AUTHORIZATION, False, "Declined Authorization", True),
    "11.3": ("visa", AUTHORIZATION, False, "No Authorization", True),
    "12.1": ("visa", PROCESSING, False, "Late Presentment", True),
    "12.5": ("visa", PROCESSING, False, "Incorrect Amount", True),
    "12.6": ("visa", PROCESSING, False, "Duplicate Processing", True),
    "13.1": ("visa", CONSUMER, False, "Merchandise/Services Not Received", True),
    "13.2": ("visa", CONSUMER, False, "Cancelled Recurring Transaction", True),
    "13.3": ("visa", CONSUMER, False, "Not as Described or Defective", True),
    "13.6": ("visa", CONSUMER, False, "Credit Not Processed", True),
    "13.7": ("visa", CONSUMER, False, "Cancelled Merchandise/Services", True),
    # ── Mastercard ──
    "4837": ("mastercard", FRAUD, False, "No Cardholder Authorization", True),
    "4849": ("mastercard", FRAUD, False, "Questionable Merchant Activity", False),
    "4863": ("mastercard", FRAUD, False, "Cardholder Does Not Recognize", True),
    "4870": ("mastercard", FRAUD, True,  "Chip Liability Shift", True),
    "4871": ("mastercard", FRAUD, True,  "Chip/PIN Liability Shift", True),
    "4808": ("mastercard", AUTHORIZATION, False, "Authorization-Related Chargeback", True),
    "4834": ("mastercard", PROCESSING, False, "Point-of-Interaction Error", True),
    "4831": ("mastercard", PROCESSING, False, "Transaction Amount Differs", True),
    "4853": ("mastercard", CONSUMER, False, "Cardholder Dispute", True),
}

# Codes whose definition explicitly spans true fraud, friendly fraud and merchant error. These
# CANNOT be resolved from the code alone, which is the whole reason terminal state is required.
AMBIGUOUS_FRAUD_CODES = ("10.4", "4863")

# ── the state machine ────────────────────────────────────────────────────────

# Stages, in order. A dispute moves forward only; a repeat of the current stage is ignored rather
# than treated as progress, because acquirers do re-send.
STAGES = ("chargeback", "representment", "pre_arbitration", "arbitration")

# Terminal outcomes. `issuer_won` means the cardholder's claim stood.
TERMINAL = {
    "issuer_won":  "the claim stood; the issuer kept the credit",
    "merchant_won": "the merchant's evidence prevailed; the charge was valid",
    "withdrawn":   "the cardholder withdrew the dispute",
    "expired":     "a deadline lapsed with no response",
}

# ASSUMPTION, not sourced. Typical network windows in days, used only to compute a maturity
# floor. They vary by code, region and network release, so this is a modelled clock and any
# real deployment replaces it with the operative rulebook.
STAGE_WINDOW_DAYS = {"chargeback": 120, "representment": 30, "pre_arbitration": 30,
                     "arbitration": 45}


def classify(code: str) -> dict:
    """What kind of dispute is this, and may it ever touch the fraud label space?"""
    entry = REASON_CODES.get(str(code or "").strip())
    if not entry:
        return {"known": False, "family": None, "network": None, "card_present": None,
                "text": "", "cardholder_asserted": None, "fraud_eligible": False,
                "ambiguous": False}
    network, family, cp, text, asserted = entry
    return {"known": True, "family": family, "network": network, "card_present": cp,
            "text": text, "cardholder_asserted": asserted,
            "fraud_eligible": family == FRAUD,
            "ambiguous": str(code).strip() in AMBIGUOUS_FRAUD_CODES}


def _parse(ts):
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def advance(events: list) -> dict:
    """Fold an append-only event list into the dispute's current state.

    Events are dicts with at least `stage` (one of STAGES) or `terminal` (a key of TERMINAL),
    plus `ts` and, on the opening chargeback, `reason_code` and `amount`.

    ONLY FORWARD. An acquirer re-sending a representment must not look like escalation to
    pre-arbitration, and a stage arriving out of order is recorded but does not move the pointer
    backwards, because a dispute that appears to regress would reopen an already-emitted label.
    """
    st = {"stage": None, "terminal": None, "reason_code": "", "amount": 0.0,
          "opened_ts": None, "closed_ts": None, "n_events": 0,
          "compelling_evidence": False, "stages_seen": []}

    for ev in events or []:
        st["n_events"] += 1
        ts = _parse(ev.get("ts"))
        stage = str(ev.get("stage") or "").strip()
        terminal = str(ev.get("terminal") or "").strip()

        if ev.get("reason_code") and not st["reason_code"]:
            st["reason_code"] = str(ev["reason_code"]).strip()
        if ev.get("amount") and not st["amount"]:
            st["amount"] = float(ev["amount"] or 0.0)
        if ev.get("compelling_evidence"):
            st["compelling_evidence"] = True

        if stage in STAGES:
            st["stages_seen"].append(stage)
            cur = STAGES.index(st["stage"]) if st["stage"] in STAGES else -1
            if STAGES.index(stage) > cur:
                st["stage"] = stage
                if stage == "chargeback" and ts and not st["opened_ts"]:
                    st["opened_ts"] = ts
        if terminal in TERMINAL:
            st["terminal"] = terminal
            st["closed_ts"] = ts

    st["settled"] = st["terminal"] is not None
    st["classification"] = classify(st["reason_code"])
    return st


def maturity_floor_days(reason_code: str = "", stage: str | None = None) -> int:
    """Days from the transaction before a dispute on it can be considered settled.

    THE REASON CARDS ARE WORTH THIS EFFORT. On the push rail the arrival lag is unknowable and
    `label_maturity` correctly refuses to fit a curve. Here the clock is defined by the rulebook,
    so a card cohort has a maturity floor that can be computed rather than estimated. Modelled
    from STAGE_WINDOW_DAYS, which is an ASSUMPTION and not a citation.
    """
    total = STAGE_WINDOW_DAYS["chargeback"]
    if stage in STAGES:
        for s in STAGES[1:STAGES.index(stage) + 1]:
            total += STAGE_WINDOW_DAYS[s]
    return total


def settled_by(transaction_ts, reason_code: str = "", stage: str | None = None):
    """The datetime after which a dispute on this transaction is past every open window."""
    t = _parse(transaction_ts)
    return t + timedelta(days=maturity_floor_days(reason_code, stage)) if t else None


# ── the label the rail is allowed to emit ────────────────────────────────────

def derive_outcome(state: dict) -> dict:
    """The outcome label for a dispute, or a refusal with the reason.

    Returns {"emit": bool, "label_value": "fraud"|"legit"|None, "confidence": float,
             "reason": str, "first_party_signal": bool}.

    `emit` is False far more often than systems in this space assume, and every False is one of
    the three rules in the module docstring doing its job.
    """
    cls = state.get("classification") or classify(state.get("reason_code", ""))
    out = {"emit": False, "label_value": None, "confidence": 0.0,
           "first_party_signal": False, "reason": ""}

    if not cls["known"]:
        out["reason"] = (f"reason code {state.get('reason_code','')!r} is not in the taxonomy; "
                         f"an unrecognised code is recorded but never guessed at")
        return out

    # RULE 1. A claim is not an outcome.
    if not state.get("settled"):
        out["reason"] = (f"dispute is at stage {state.get('stage')!r} and has not settled. A "
                         f"chargeback is a claim, not an adjudication; labelling here is the "
                         f"immature-window error")
        return out

    # RULE 2. Only the fraud family may reach the fraud label space.
    if not cls["fraud_eligible"]:
        out["reason"] = (f"{state['reason_code']} is a {cls['family']} dispute "
                         f"({cls['text']}), not a fraud claim. Recorded, but withheld from "
                         f"outcome.is_fraud: training on it teaches the model that a service "
                         f"failure is fraud")
        return out

    terminal = state.get("terminal")

    # A monitoring-program chargeback is not evidence about this cardholder.
    if not cls["cardholder_asserted"]:
        out["reason"] = (f"{state['reason_code']} ({cls['text']}) is raised by a monitoring "
                         f"program, not by the cardholder. It says something about the merchant, "
                         f"not about whether this transaction was fraud")
        return out

    # RULE 3. A lost representment inverts the claim.
    if terminal == "merchant_won":
        out.update({"emit": True, "label_value": "legit",
                    "confidence": 0.80 if state.get("compelling_evidence") else 0.65,
                    "first_party_signal": bool(state.get("compelling_evidence")),
                    "reason": ("a fraud claim the merchant defeated on evidence. The network "
                               "ruled the charge valid, so this is a label CORRECTION, and a "
                               "cardholder who disputed a charge they made is the canonical "
                               "first-party abuse shape")})
        return out

    if terminal == "withdrawn":
        out.update({"emit": True, "label_value": "legit", "confidence": 0.55,
                    "first_party_signal": True,
                    "reason": ("the cardholder withdrew a fraud claim. Weaker evidence than a "
                               "contested win, but it points the same way")})
        return out

    if terminal == "issuer_won":
        conf = 0.75 if cls["ambiguous"] else 0.90
        out.update({"emit": True, "label_value": "fraud", "confidence": conf,
                    "reason": (f"fraud claim upheld at settlement. Confidence is held at {conf} "
                               + ("because this code spans true fraud, friendly fraud and "
                                  "merchant error by definition, and an uncontested win does "
                                  "not separate them"
                                  if cls["ambiguous"] else
                                  "on an unambiguous fraud code"))})
        return out

    # expired: nobody adjudicated anything.
    out["reason"] = ("the dispute expired on a lapsed deadline. No party contested and no party "
                     "prevailed, so there is no adjudicated fact to record")
    return out


def to_ledger_record(state: dict, subject_ref: str, transaction_ts=None) -> dict | None:
    """Shape an emitted outcome for core.outcome_ledger.record_outcome, or None if withheld.

    `effective_ts` is the SETTLEMENT time, not the ingest time. The maturity curve measures the
    lag between the fact becoming true and us learning it, so recording the moment we happened
    to process the file would make the arrival lag look like zero.
    """
    d = derive_outcome(state)
    if not d["emit"]:
        return None
    closed = state.get("closed_ts")
    code = state.get("reason_code", "")
    return {
        "subject_ref": subject_ref,
        "outcome": d["label_value"],
        "source": "chargeback",
        # The gradation is the point. A withdrawn dispute and a contested win the merchant took
        # on evidence are both "legit" and are not equally strong; the ledger records the
        # difference so training can weight them apart.
        "confidence": d["confidence"],
        "effective_ts": closed.isoformat().replace("+00:00", "Z") if closed else "",
        # reason_code gets its own field rather than being buried in the reference, so the
        # ledger can be queried by dispute type without string-splitting.
        "reason_code": code,
        "reference": f"{code}/{state.get('terminal','')}",
        "amount": state.get("amount") or None,
        "notes": d["reason"],
    }
