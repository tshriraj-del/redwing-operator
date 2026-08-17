"""
core/authorization.py - the authorization decision path. The spine everything else plugs into.

WHAT WAS MISSING. Screening, two-sided pricing, the decision policy, the model registry and the
decline contract all existed and none of them were reachable from an authorization, because
there was no authorization. The platform could score a payment and could not answer the only
question a card network ever asks: approve or decline, in under two seconds, with a reason code.

That is the difference between a fraud scorer and a payment risk system, and it is why this
module is a spine rather than another component. It calls the pieces that already exist and adds
the three things an issuer owes the network:

    A LATENCY BUDGET. Networks give an issuer roughly two seconds. Miss it and the network
    answers on your behalf under stand-in, using rules configured in advance. A system with no
    budget has no answer for its own slowness, which means the network's default becomes your
    policy without anyone deciding that.

    A RESPONSE CODE. Not an internal action. `HOLD` means nothing to an acquirer; `05` does.
    The mapping is where an internal risk vocabulary becomes something a terminal, a merchant
    and a member can act on, and getting it wrong is how members end up staring at "Do Not
    Honor" for an insufficient balance.

    SOFT VERSUS HARD. A soft decline invites a retry; a hard one forbids it. Networks limit
    re-attempts on specific codes and fine violations, so this distinction is a contractual
    fact rather than a UX nicety, and any recovery flow built later has to respect it.

WHAT THIS IS NOT. It is not connected to a network. There is no ISO 8583 wire format, no
acquirer, no stand-in agreement with anybody. It MODELS the contract an issuer operates under so
the decisioning can be built and measured against it, and every place that boundary matters is
marked. Approximating a contractual rule and calling it compliance would be worse than not
claiming it.

Pure stdlib.
"""

from __future__ import annotations

import time

from .decision_policy import decide as decide_action
from .liability import price_decision
from .screening import screen

# ── the network contract, modelled ───────────────────────────────────────────

# Networks expect an authorization response inside roughly two seconds. Well under it in
# practice, because the acquirer, the switch and the network each consume part of the window;
# an issuer that uses the whole budget is late by the time the message gets back.
AUTH_BUDGET_MS = 1_500
STIP_THRESHOLD_MS = 1_200      # past this, stop and let stand-in rules answer

RESPONSE_CODES = {
    "00": "Approved",
    "05": "Do Not Honor",
    "51": "Insufficient Funds",
    "54": "Expired Card",
    "57": "Transaction Not Permitted to Cardholder",
    "59": "Suspected Fraud",
    "61": "Exceeds Withdrawal Amount Limit",
    "62": "Restricted Card",
    "65": "Exceeds Withdrawal Count Limit / SCA Required",
    "82": "Negative CAM, dCVV, iCVV or CVV Results",
    "91": "Issuer or Switch Inoperative",
}

# SOFT declines invite a retry, HARD ones forbid it. This is contractual: networks limit
# re-attempts on specific codes and fine violations, so a recovery flow that retries a hard
# decline is not merely rude, it is a scheme breach.
SOFT_DECLINES = ("51", "61", "65", "91")
HARD_DECLINES = ("05", "54", "57", "59", "62", "82")

# Internal action -> what the acquirer is told. `HOLD` means nothing outside this building.
#
# STEP_UP maps to 65 rather than 05 deliberately: under SCA the 65/1A family is the code that
# tells the merchant to re-attempt WITH authentication, so a step-up expressed as "Do Not Honor"
# would throw away the retry the step-up exists to enable.
_ACTION_TO_CODE = {
    "ALLOW": "00",
    "MONITOR": "00",       # approved, and the account is watched. The member sees nothing.
    "STEP_UP": "65",       # soft: come back authenticated
    "HOLD": "05",
    "BLOCK": "05",
    "DECLINE": "05",
}


def is_soft(code: str) -> bool:
    return str(code) in SOFT_DECLINES


def retry_allowed(code: str) -> bool:
    """May the acquirer re-attempt this? Hard declines say no, and networks enforce it."""
    return str(code) == "00" or is_soft(code)


def _funds_and_limits(msg: dict) -> str | None:
    """The declines that are the MEMBER'S situation rather than our opinion of them.

    Checked before the risk engine because that is the order a real auth path evaluates, and
    because the answer changes what the member is told: an insufficient balance is theirs to
    fix and deserves to be named, while a risk decline deliberately is not.
    """
    try:
        amount = float(msg.get("amount") or 0)
        available = msg.get("available_balance")
        if available is not None and amount > float(available):
            return "51"
    except (TypeError, ValueError):
        pass
    try:
        if msg.get("daily_count") is not None and int(msg["daily_count"]) >= int(
                msg.get("daily_count_limit", 10**9)):
            return "65"
        if msg.get("daily_amount") is not None and (
                float(msg["daily_amount"]) + float(msg.get("amount") or 0)
                > float(msg.get("daily_amount_limit", float("inf")))):
            return "61"
    except (TypeError, ValueError):
        pass
    return None


def stand_in(msg: dict) -> dict:
    """What the network would do on our behalf if we do not answer in time.

    Modelled, not agreed with anybody. The point of having it is that an issuer must DECIDE its
    stand-in posture in advance: without one, the network's default silently becomes the
    institution's policy and nobody chose it. Conservative here, approving small
    card-present amounts and declining the rest, because stand-in runs blind to everything this
    system knows.
    """
    try:
        amount = float(msg.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0.0
    cp = str(msg.get("entry_mode", "")) in ("chip", "contactless")
    if cp and amount <= 100:
        return {"code": "00", "why": "stand-in floor limit: small card-present amount"}
    return {"code": "05", "why": "stand-in declined: no issuer response inside the window"}


def authorize(msg: dict, *, score_fn=None, budget_ms: int = AUTH_BUDGET_MS,
              posture: dict | None = None, now_ms=None) -> dict:
    """One authorization, start to finish.

    `score_fn(msg) -> (p_fraud, detail)` is injected so this module stays free of the ML stack
    and the tests can drive it deterministically. Everything else is the real path: the same
    screening gate, the same two-sided pricing, the same policy table.
    """
    t0 = (now_ms() if now_ms else time.monotonic() * 1000.0)
    trail = []

    def elapsed():
        return (now_ms() if now_ms else time.monotonic() * 1000.0) - t0

    def respond(code, *, action, reason, extra=None):
        ms = round(elapsed(), 2)
        out = {
            "approved": code == "00",
            "response_code": code,
            "response_text": RESPONSE_CODES.get(code, "Unknown"),
            "action": action,
            "soft_decline": code != "00" and is_soft(code),
            "retry_allowed": retry_allowed(code),
            "reason": reason,
            "latency_ms": ms,
            "within_budget": ms <= budget_ms,
            "trail": trail,
        }
        out.update(extra or {})
        return out

    # 1. SCREENING. Before anything else and before any score exists, because a payment to a
    #    designated party cannot be approved at any score. Fails CLOSED.
    scr = screen(counterparty=str(msg.get("merchant_name") or msg.get("merchant_id") or ""),
                 member=str(msg.get("cardholder_name") or ""))
    trail.append({"step": "screening", "result": scr.get("result")})
    if scr.get("blocked"):
        # 57 rather than 05: "not permitted to cardholder" is the honest code for a payment we
        # are prohibited from processing, and it is distinguishable from a risk decline by
        # anyone reading the response.
        return respond("57", action="BLOCK", reason=scr["reason"],
                       extra={"screening": scr, "terminal": True})

    # 2. FUNDS AND LIMITS. The member's own situation, evaluated before our opinion of them.
    fl = _funds_and_limits(msg)
    trail.append({"step": "funds_limits", "result": fl or "ok"})
    if fl:
        return respond(fl, action="DECLINE",
                       reason=RESPONSE_CODES[fl], extra={"member_situation": True})

    # 3. RISK. The score, and then the money, and then the policy.
    p_fraud, detail = (score_fn(msg) if score_fn else (0.0, {"scored": False}))
    trail.append({"step": "score", "p_fraud": round(float(p_fraud), 4)})

    if elapsed() > STIP_THRESHOLD_MS:
        # Past the point where a response can get back in time. Answer with the stand-in rule
        # rather than continuing to think, because a correct answer after the window is a
        # timeout, and a timeout is the network deciding for us.
        si = stand_in(msg)
        trail.append({"step": "stand_in", "why": si["why"]})
        return respond(si["code"], action="STAND_IN", reason=si["why"],
                       extra={"stand_in": True, "score_detail": detail})

    priced = price_decision(
        p_fraud, msg.get("amount", 0), typology=str(detail.get("typology", "")),
        rail="card", action="DECLINE", ltv_band=str(msg.get("ltv_band", "")),
        account_age_days=msg.get("account_age_days", 365), config=posture)
    trail.append({"step": "priced", "recommended": priced.get("recommended_action")})

    pol = decide_action(priced.get("recommended_action") or "ALLOW", p_fraud,
                        rail="card", direction="outbound",
                        tier=("new_account" if float(msg.get("account_age_days", 365)) < 30
                              else ""))
    trail.append({"step": "policy", "action": pol["action"], "bounded_by": pol["bounded_by"]})

    code = _ACTION_TO_CODE.get(pol["action"], "05")
    # A risk decline stays deliberately vague, which is the industry's answer to the fact that
    # an explanation is also a description of the control. core/decline_contract.py is what
    # makes it possible to vary that per member without handing attackers a roadmap.
    reason = ("approved" if code == "00" else
              "step-up required" if code == "65" else "risk decision")
    # The gate views are lifted to the TOP LEVEL, matching build_event() and /score, which both
    # expose `device_gate` there. A control that is observable under a different key on each path
    # is a control a conformance probe cannot check uniformly, and uniform checkability is the
    # entire point of the harness. `detail` keeps them too, so nothing is moved, only surfaced.
    return respond(code, action=pol["action"], reason=reason,
                   extra={"screening": scr, "priced": priced, "policy": pol,
                          "score_detail": detail,
                          "device_gate": (detail or {}).get("device_gate") or {},
                          "sequence_gate": (detail or {}).get("sequence_gate") or {}})


# ── the durable record ───────────────────────────────────────────────────────
#
# ADR-001 action item 2, and the most consequential gap the conformance test tracked: this path
# wrote nothing, so the card rail produced no labels, no outcome-ledger entries and no holdout
# membership. It could never be measured for decay and could never graduate.

def card_subject_ref(msg: dict) -> str:
    """The key a chargeback will arrive under, months later.

    THIS IS THE WHOLE JOIN, and the two identifiers involved live at different moments. At
    authorization the message carries the RRN (DE 37) and STAN; the ARN is assembled at
    CLEARING and is what the dispute file references. An issuer keeps the mapping between them,
    so either is a usable key here and the ARN is preferred when present because it is the one
    the outcome will arrive under.

    Filing the decision under a purely internal id would make the outcome unjoinable to the
    decision that caused it, which is exactly the failure this record exists to prevent, so an
    internal id is accepted only as a last resort.
    """
    for key in ("arn", "acquirer_reference_number", "rrn", "retrieval_reference_number",
                "auth_id", "transaction_id"):
        v = str(msg.get(key, "") or "").strip()
        if v:
            return v
    return ""


def durable_record(msg: dict, decision: dict, *, holdout_fn=None) -> dict:
    """Everything needed to persist one authorization, computed WITHOUT touching a store.

    WHY THE SPLIT. A card authorization answers against a network deadline, so the write cannot
    sit in the response path. That forces the record to be built here and persisted after the
    response, which is only safe because two properties hold:

      the subject_ref is the ARN, so a dispute months later joins to this exact decision, and
      the holdout call is a pure hash of that ref, so membership is decided IN the decision and
      not by whoever gets around to writing it. Deciding holdout at write time would let a
      delayed or dropped write silently change which cases were sampled, and the holdout is only
      unbiased if nothing downstream can influence membership.

    `holdout_fn` is injected so this stays free of import cycles and the tests can drive it.
    Returns {} when there is no usable subject_ref, because a decision nobody can join an
    outcome to is not worth a row and would inflate coverage denominators with dead weight.
    """
    ref = card_subject_ref(msg)
    if not ref:
        return {}

    # THE CARD, as opposed to the transaction. `subject_ref` is the ARN and identifies one
    # authorization; `entity_id` identifies the card across all of them, which is what makes a
    # per-card trailing window queryable at all. Filed under the already-indexed entity column,
    # and it is a salted hash: the PAN is never stored. See core/card_identity.py.
    from .card_identity import card_key
    from .store import eid
    ckey = card_key(msg)
    entity = eid("card", ckey) if ckey else ""

    action = str(decision.get("action") or "")
    priced = decision.get("priced") or {}
    # `price_decision` returns `cost_of_allowing`, which IS the expected liability; there is no
    # `expected_liability` key and reading one wrote 0.0 on every card decision. That silently
    # disabled the holdout's liability ceiling too, since nothing could exceed a limit when
    # every case reported zero. Found by watching the sequence gate never fire.
    liability = float(priced.get("cost_of_allowing") or 0.0)
    amount = float(msg.get("amount", 0.0) or 0.0)

    ho = {"release": False, "enforced_action": action, "holdout": False, "reason": ""}
    if holdout_fn:
        ho = holdout_fn(ref, action, liability)

    return {
        "subject_ref": ref,
        "entity_id": entity,
        "card_key": ckey,
        "action": ho.get("enforced_action") or action,
        "score": float((decision.get("score_detail") or {}).get("p_fraud",
                       priced.get("p_fraud", 0.0)) or 0.0),
        "expected_liability": liability,
        # AMOUNT IS PERSISTED SEPARATELY FROM LIABILITY and the two are not interchangeable.
        # Liability is p x amount x reimbursement rate (dollars at risk); amount is dollars
        # transacted. The sequence gate compares this authorization's amount to the card's own
        # recent AMOUNTS, exactly as the training pass does, and comparing it to liability
        # instead is a units error that silently produces a meaningless ratio.
        "features": {**((decision.get("score_detail") or {}).get("features") or {}),
                     "amount": amount},
        "holdout": bool(ho.get("holdout")),
        "released": bool(ho.get("release")),
        "rationale": {
            "path": "authorize",
            "rail": "card",
            "response_code": decision.get("response_code"),
            "proposed_action": action,
            "enforced": True,
            "holdout": bool(ho.get("holdout")),
            "released": bool(ho.get("release")),
            "holdout_reason": ho.get("reason", ""),
            "stand_in": bool(decision.get("stand_in")),
            "within_budget": decision.get("within_budget"),
            # The dispute rail reads these back to compute a maturity floor per authorization.
            "entry_mode": str(msg.get("entry_mode", "") or ""),
            "mcc_code": msg.get("mcc_code"),
            "bin": str(msg.get("bin", "") or ""),
        },
    }
