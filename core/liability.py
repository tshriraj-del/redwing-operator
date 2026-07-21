"""
core/liability.py - two-sided decision pricing (Phase 1 WS4, Phase 2 WS10).

Post-UK-PSR (mandatory APP-scam reimbursement, Oct 2024) and under mounting US
pressure on instant-rail scams, the buyer's real question is not "fraud or not" but
"how many reimbursement dollars am I exposed to if I let this through". Irrevocable /
push rails carry scam-reimbursement liability the institution eats; card fraud is
largely chargeback- and network-protected. So we price every decision in dollars of
expected liability, not just probability. This aligns the product with the buyer's
actual P&L and is a framing no incumbent leads with.

    expected_liability = p_fraud * amount * reimbursement_rate(typology, rail)

WS10 adds the other side. Pricing only the fraud loss makes the objective one-sided, and a
one-sided objective always over-blocks: the fraud team is measured on losses prevented while
the cost of a wrongly declined customer lands on someone else's P&L. Measured on this platform
at its own operating point, that bias costs 2.6 false positives per real fraud caught.

    cost_of_allowing = p_fraud       * amount * reimbursement_rate
    cost_of_blocking = (1 - p_fraud) * false_positive_cost(customer, amount)

Block when the first exceeds the second. Rearranged, that gives a break-even probability per
transaction, which replaces a hand-tuned global threshold with one derived from the money:

    p* = fp_cost / (fp_cost + amount * reimbursement_rate)

A $40 payment from a ten-year customer and a $9,000 wire from a three-week-old account get
very different bars, which is the correct behaviour and something a single global threshold
cannot express.

HONESTY: the fraud-side rates below are grounded in rail mechanics and post-UK-PSR
reimbursement rules. The false-positive-side parameters (churn probabilities, LTV bands,
contact cost) are ASSUMPTIONS, not measurements. They are defaults an institution is expected
to replace with its own retention data, they are all overridable, and every priced decision
returns the assumptions it used so a number can never be quoted without its inputs.

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


# -- The false-positive side (WS10) --------------------------------------------
#
# ALL of the constants below are assumptions to be replaced with an institution's own
# retention data. They are deliberately conservative: if anything they understate the cost of
# a wrongful block, so the model does not get to justify under-blocking with invented numbers.

# Typologies where the scored transaction is RECONNAISSANCE, not the loss event. A card-testing
# probe is deliberately tiny; its value to the attacker is confirming a stolen card works before
# the real hit. Pricing such a decision on its own amount understates it enormously, because
# what blocking buys is the prevention of a later, larger loss this transaction is scouting for.
# Two-sided pricing is right for ordinary payments and structurally wrong here, so these are
# flagged rather than silently mispriced.
_RECON_TYPOLOGIES = ("card_testing_bot", "enumeration", "bin_attack")

# Remaining lifetime value by band, in dollars. What is lost if the customer leaves.
_LTV_BAND = {"low": 400.0, "medium": 2_500.0, "high": 12_000.0, "": 2_500.0}

# Probability a wrongly-actioned customer churns, by how hard the action was. A silent hold
# an analyst clears in minutes is not the same event as a declined payment at the till.
_CHURN_BY_ACTION = {
    "BLOCK":   0.10,
    "DECLINE": 0.10,
    "HOLD":    0.04,
    "STEP_UP": 0.01,
    "ALLOW":   0.0,
    "":        0.06,
}

# New customers have not formed a habit and leave easily; long-tenured ones are more forgiving
# per incident, but each is worth more, which the LTV band already carries.
_TENURE_CHURN_MULT = ((30, 2.0), (90, 1.5), (365, 1.0), (10**9, 0.7))

_CONTACT_COST = 12.0     # one support contact to unwind a false decline
_MARGIN_RATE = 0.02      # revenue margin forgone on the declined payment itself


def _tenure_mult(account_age_days) -> float:
    try:
        age = max(0.0, float(account_age_days))
    except (TypeError, ValueError):
        age = 365.0
    for cutoff, mult in _TENURE_CHURN_MULT:
        if age <= cutoff:
            return mult
    return 1.0


def false_positive_cost(amount, action: str = "BLOCK", ltv_band: str = "",
                        account_age_days=365, config: dict | None = None) -> dict:
    """Dollars of expected damage from wrongly actioning a legitimate customer.

    Three components, because they behave differently: the margin forgone on this payment
    scales with amount, the support contact is roughly fixed, and the attrition term dominates
    and scales with what the customer is worth. Returns the breakdown, not just a total, so a
    decision can always be argued with rather than merely obeyed."""
    cfg = config or {}
    ltv = cfg.get("ltv", _LTV_BAND.get(str(ltv_band).strip().lower(), _LTV_BAND[""]))
    churn = cfg.get("churn", _CHURN_BY_ACTION.get(str(action).strip().upper(), _CHURN_BY_ACTION[""]))
    contact = cfg.get("contact_cost", _CONTACT_COST)
    margin = cfg.get("margin_rate", _MARGIN_RATE)

    try:
        amt = max(0.0, float(amount))
    except (TypeError, ValueError):
        amt = 0.0

    p_churn = min(1.0, churn * _tenure_mult(account_age_days))
    attrition = p_churn * float(ltv)
    lost_margin = amt * float(margin)
    total = attrition + lost_margin + (float(contact) if churn > 0 else 0.0)

    return {
        "total": round(total, 2),
        "attrition": round(attrition, 2),
        "lost_margin": round(lost_margin, 2),
        "contact_cost": round(float(contact) if churn > 0 else 0.0, 2),
        "assumptions": {"ltv": float(ltv), "p_churn": round(p_churn, 4),
                        "action": str(action).upper(), "tenure_mult": _tenure_mult(account_age_days)},
    }


def breakeven_p(amount, typology: str = "", rail: str = "", action: str = "BLOCK",
                ltv_band: str = "", account_age_days=365, config: dict | None = None) -> float:
    """The fraud probability at which blocking and allowing cost the same.

    Above it, block; below it, allow. This is the threshold, derived rather than tuned. It
    falls as the amount rises (more to lose by allowing) and rises with customer value (more
    to lose by blocking)."""
    fp = false_positive_cost(amount, action, ltv_band, account_age_days, config)["total"]
    try:
        exposure = max(0.0, float(amount)) * reimbursement_rate(typology, rail)
    except (TypeError, ValueError):
        exposure = 0.0
    if exposure + fp <= 0:
        return 1.0
    return round(fp / (fp + exposure), 4)


def price_decision(p_fraud, amount, typology: str = "", rail: str = "", action: str = "BLOCK",
                   ltv_band: str = "", account_age_days=365, config: dict | None = None) -> dict:
    """Price both sides of one decision and return the call the money supports.

    This is the WS10 objective: the agent optimises the institution's P&L rather than the fraud
    department's metric. A decision that prevents $80 of expected loss by risking $310 of
    expected customer damage is a bad trade even though it 'stopped fraud'."""
    try:
        p = min(1.0, max(0.0, float(p_fraud)))
    except (TypeError, ValueError):
        p = 0.0

    cost_allow = expected_liability(p, amount, typology, rail)
    fp = false_positive_cost(amount, action, ltv_band, account_age_days, config)
    cost_block = round((1.0 - p) * fp["total"], 2)
    thr = breakeven_p(amount, typology, rail, action, ltv_band, account_age_days, config)

    recon = str(typology).strip().lower() in _RECON_TYPOLOGIES
    recommended = action.upper() if cost_allow > cost_block else "ALLOW"

    out = {
        "p_fraud": round(p, 4),
        "cost_of_allowing": cost_allow,
        "cost_of_blocking": cost_block,
        "net_benefit_of_blocking": round(cost_allow - cost_block, 2),
        "breakeven_p": thr,
        "recommended_action": recommended,
        "false_positive_cost": fp,
        "prices_this_transaction_only": True,
        "rationale": (
            f"expected fraud loss ${cost_allow:,.2f} exceeds expected customer damage "
            f"${cost_block:,.2f}; block pays for itself" if cost_allow > cost_block else
            f"expected customer damage ${cost_block:,.2f} exceeds expected fraud loss "
            f"${cost_allow:,.2f}; blocking destroys more value than it saves"
        ),
    }

    if recon:
        # Do not let a confident recommendation ride on a price we know is wrong. The economics
        # are still returned, because they are true about THIS transaction; what is withdrawn
        # is the recommendation, since the loss being prevented is not on this row.
        out["recommended_action"] = "DEFER_TO_DETECTION"
        out["caveat"] = (
            f"'{typology}' is reconnaissance: this payment is a probe, so its own amount "
            f"(${float(amount or 0):,.2f}) understates what blocking prevents. Two-sided "
            f"pricing does not apply; defer to the detection score, which is not amount-bound."
        )
        out["rationale"] = out["caveat"]
    return out
