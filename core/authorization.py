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
    return respond(code, action=pol["action"], reason=reason,
                   extra={"screening": scr, "priced": priced, "policy": pol,
                          "score_detail": detail})
