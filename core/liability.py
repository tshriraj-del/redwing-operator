"""
core/liability.py - liability-priced decisions (Phase 1, WS4).

Post-UK-PSR (mandatory APP-scam reimbursement, Oct 2024) and under mounting US
pressure on instant-rail scams, the buyer's real question is not "fraud or not" but
"how many reimbursement dollars am I exposed to if I let this through". Irrevocable /
push rails carry scam-reimbursement liability the institution eats; card fraud is
largely chargeback- and network-protected. So we price every decision in dollars of
expected liability, not just probability. This aligns the product with the buyer's
actual P&L and is a framing no incumbent leads with.

    expected_liability = p_fraud * amount * reimbursement_rate(typology, rail)

Pure Python - unit-testable without the ML stack.
"""

from __future__ import annotations

# Share of a realised fraud loss the INSTITUTION bears (unrecoverable) by rail.
# Irrevocable / push rails: money is gone and, post-regulation, the bank reimburses
# the victim -> high. Card: chargeback + network liability shift -> low.
_RAIL_LIABILITY = {
    "crypto":  0.98,
    "zelle":   0.95,
    "fednow":  0.95,
    "rtp":     0.95,
    "wire":    0.90,
    "open_banking": 0.75,
    "ach":     0.60,
    "bnpl":    0.45,
    "card":    0.15,
    "":        0.50,
}

# Typology multiplier: authorized-push-payment scams (victim authorised the payment)
# carry the most reimbursement exposure; automated card abuse the least.
_TYPOLOGY_MULT = {
    "pig_butchering":              1.00,
    "app_scam":                    1.00,
    "deepfake_social_engineering": 1.00,
    "account_takeover_ai":         0.85,
    "synthetic_id_ai":             0.70,
    "card_testing_bot":            0.35,
    "none":                        1.00,
    "":                            1.00,
}


def reimbursement_rate(typology: str = "", rail: str = "") -> float:
    """Fraction of a loss on this rail/typology the institution is exposed to."""
    base = _RAIL_LIABILITY.get(str(rail).strip().lower(), 0.50)
    mult = _TYPOLOGY_MULT.get(str(typology).strip().lower(), 1.00)
    return round(min(1.0, base * mult), 4)


def expected_liability(p_fraud, amount, typology: str = "", rail: str = "") -> float:
    """Dollars of expected reimbursement liability from letting this payment through.
    Never raises - returns 0.0 on bad input so it can sit in the hot score path."""
    try:
        p = min(1.0, max(0.0, float(p_fraud)))
        amt = max(0.0, float(amount))
    except (TypeError, ValueError):
        return 0.0
    return round(p * amt * reimbursement_rate(typology, rail), 2)
