"""
core/narrative.py - scam-narrative reasoning (Phase 1, WS4).

The case should explain the CON, not just the transaction. Supervised models see a
feature vector; analysts want "this looks like pig-butchering, stage 3, crypto
off-ramp". This turns a typology plus the case's own signals into a plain-language
narrative, deterministically, so it works with no LLM key (the FraudSense copilot can
enrich it, but the base read is free and always available). Pure Python, testable.
"""

from __future__ import annotations

_RAIL_LABEL = {"zelle": "Zelle", "fednow": "FedNow", "rtp": "RTP",
               "wire": "wire", "crypto": "crypto", "card": "card", "ach": "ACH"}
_IRREVOCABLE = {"zelle", "fednow", "rtp", "wire", "crypto"}


def _money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except (TypeError, ValueError):
        return "$0"


def scam_narrative(typology: str = "", signals: dict | None = None) -> dict:
    """Return {headline, typology, stage, narrative, cues} for a case. `signals` may
    carry amount, rail, is_new_recipient, expected_liability."""
    s = signals or {}
    amount   = s.get("amount", 0.0)
    rail     = str(s.get("rail", "") or "").lower()
    rail_txt = _RAIL_LABEL.get(rail, rail or "an unknown rail")
    new_payee   = bool(s.get("is_new_recipient"))
    irrevocable = rail in _IRREVOCABLE
    typ = str(typology or "").strip().lower()

    cues = []
    if irrevocable:
        cues.append(f"irrevocable rail ({rail_txt}): funds are unrecoverable once sent")
    if new_payee:
        cues.append("first-ever payment to this payee")
    if amount:
        cues.append(f"{_money(amount)} payment")
    if s.get("expected_liability"):
        cues.append(f"{_money(s['expected_liability'])} of reimbursement liability at stake")

    if typ in ("pig_butchering", "app_scam", "deepfake_social_engineering"):
        headline = "Authorized-push-payment scam"
        stage = "cash-out to crypto off-ramp" if rail == "crypto" else "victim-authorised transfer"
        narrative = (
            f"Pattern consistent with {typ.replace('_', ' ')}. The customer is authorising "
            f"{_money(amount)} over {rail_txt} to {'a new payee' if new_payee else 'this payee'}. "
            f"Because the victim authorises the payment, the rail never flags it and there is no "
            f"chargeback: the loss falls on the institution under reimbursement rules. The tell is "
            f"not the mechanics but the story - grooming, urgency, and a destination the customer "
            f"has no history with.")
    elif typ == "account_takeover_ai":
        headline, stage = "Account takeover", "post-compromise cash-out"
        narrative = (
            f"Consistent with an AI-driven account takeover: {_money(amount)} moving over {rail_txt}"
            f"{' to a new payee' if new_payee else ''} after a session anomaly. The genuine "
            f"customer did not initiate this.")
    elif typ == "synthetic_id_ai":
        headline, stage = "Synthetic identity", "bust-out"
        narrative = (
            f"Consistent with a synthetic identity: a thin-file account moving {_money(amount)} - "
            f"the bust-out after cultivating just enough history to look real.")
    elif typ == "card_testing_bot":
        headline, stage = "Card testing", "automated validation"
        narrative = (
            "Micro-amount, high-velocity automated card validation: the bot is checking stolen "
            "card numbers before a larger cash-out elsewhere.")
    else:
        headline, stage = "Elevated risk", "review"
        narrative = (
            f"{_money(amount)} over {rail_txt}{' to a new payee' if new_payee else ''}. No single "
            f"dominant typology; weigh the cues before deciding.")

    return {"headline": headline, "typology": typ or "unspecified", "stage": stage,
            "narrative": narrative, "cues": cues}
