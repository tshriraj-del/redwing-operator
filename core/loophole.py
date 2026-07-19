"""
core/loophole.py - loophole and policy-exploitation intelligence (novel module).

motive.py can label an actor's motive as "loophole". This is the operational counterpart:
detect the exploit, decide whether the failure is the ACTOR or the SYSTEM, and synthesize
the control that closes the gap. The governing idea from the concept doc: loophole fraud is
scalable and technical, not always a criminal mindset, and punishing the actor without
fixing the gap just invites the next one. So this is run like vulnerability management, not
like enforcement: find the leak, quantify it, patch it.

It recognises the exploit families that recur across fintechs: promotion / bonus abuse
(multi-accounting, referral farming), threshold arbitrage (structuring just under a control
limit, the sub-$2 card-testing gap generalised), settlement-gap timing (spending the
authorisation-to-settlement float), policy-gap abuse (goodwill credits, returns, free-trial
cycling), and rate / rounding edges.

The centrepiece is synthesize_control: a concrete, closing rule per family, handed to the
Rule Factory. And a systemic test: when many actors walk the same edge, the response flips
from "punish the actor" to "fix the policy", because at population scale the control gap,
not the customer, is the fraud.

Pure Python, deterministic, unit-testable.
"""

from __future__ import annotations

EXPLOIT_FAMILIES = {
    "promo_abuse":         "Promotion / bonus abuse (multi-accounting, referral farming)",
    "threshold_arbitrage": "Threshold arbitrage (activity walked just under a control limit)",
    "settlement_gap":      "Timing / settlement-gap exploitation (spending the float)",
    "policy_gap":          "Policy-gap abuse (goodwill, returns, free-trial cycling)",
    "rate_rounding":       "Rate / rounding-edge exploitation",
    "rule_evasion":        "Rules-engine blind-spot / known-rule evasion",
}

# tell -> {family: weight}. `signals` is a flat dict of tell -> strength (0-1).
EXPLOIT_TELLS = {
    "repeated_just_below_threshold": {"threshold_arbitrage": 0.8, "rule_evasion": 0.3},
    "systematic_edge_testing":       {"threshold_arbitrage": 0.5, "rule_evasion": 0.5},
    "velocity_just_under_limit":     {"threshold_arbitrage": 0.6},
    "multi_account_same_beneficiary":{"promo_abuse": 0.8},
    "referral_self_dealing":         {"promo_abuse": 0.75},
    "welcome_bonus_multi_signup":    {"promo_abuse": 0.7},
    "free_trial_cycling":            {"policy_gap": 0.6, "promo_abuse": 0.3},
    "goodwill_credit_repeat":        {"policy_gap": 0.7},
    "returns_policy_abuse":          {"policy_gap": 0.6},
    "settlement_window_spend":       {"settlement_gap": 0.8},
    "pending_balance_spend":         {"settlement_gap": 0.6},
    "auth_settle_mismatch":          {"settlement_gap": 0.6},
    "fx_rounding_exploit":           {"rate_rounding": 0.75},
    "fee_edge_exploit":              {"rate_rounding": 0.6},
    "known_rule_evasion":            {"rule_evasion": 0.8},
}

# A concrete closing control per family: the Rule Factory handoff.
_CONTROLS = {
    "threshold_arbitrage": {
        "control": "Aggregate-and-apply: evaluate the control on cumulative value per entity / device / instrument over a rolling window, not on the single transaction.",
        "mechanism": "Sum sub-threshold activity across the window and re-apply the limit to the cumulative amount; add a just-below-clustering detector."},
    "promo_abuse": {
        "control": "Cluster-dedupe incentives: grant per verified entity cluster, not per account, and cap benefit per cluster.",
        "mechanism": "Resolve device / payment-instrument / address / identity into one cluster before granting a bonus or referral; cap and cool-down per cluster."},
    "settlement_gap": {
        "control": "Reserve-against-pending: reconcile spendable balance to settled funds and hold a reserve for open authorisations.",
        "mechanism": "Decrement available balance at authorisation, not settlement; block spend of unsettled or pending credits."},
    "policy_gap": {
        "control": "Rate-limit discretion: cap goodwill credits / returns / trials per entity over a rolling window and escalate past a cluster threshold.",
        "mechanism": "Track discretionary grants per entity-cluster; require manual escalation beyond N per window."},
    "rate_rounding": {
        "control": "Neutralise the edge: round to neutral or the institution's favour and cap cumulative rounding benefit per entity per period.",
        "mechanism": "Replace favourable-to-customer rounding with banker's rounding; monitor cumulative rounding gain per entity."},
    "rule_evasion": {
        "control": "Shape-to-the-rule detector: flag behaviour clustered just inside a known rule edge and randomise / lower the visible threshold.",
        "mechanism": "Detect just-below clustering against each live rule; introduce jitter so the exact edge is not learnable."},
}


def _clamp(x) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def detect_exploit(signals: dict) -> dict:
    """Identify the most likely exploit family and the tells that point to it."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    scores = {f: 0.0 for f in EXPLOIT_FAMILIES}
    drivers = {f: [] for f in EXPLOIT_FAMILIES}
    for tell, strength in sig.items():
        if strength <= 0:
            continue
        for fam, w in EXPLOIT_TELLS.get(tell, {}).items():
            scores[fam] += strength * w
            drivers[fam].append(tell)

    if not any(scores.values()):
        return {"family": None, "family_label": None, "confidence": 0.0,
                "drivers": [], "scores": {}}
    top = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    return {
        "family": top,
        "family_label": EXPLOIT_FAMILIES[top],
        "confidence": round(scores[top] / total, 3),
        "drivers": drivers[top],
        "scores": {f: round(v, 3) for f, v in scores.items() if v > 0},
    }


def is_systemic(signals: dict) -> dict:
    """Decide whether this is one actor or a population walking the same edge. At population
    scale the policy gap, not the customer, is the fraud, and the response must change."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    prevalence = max(sig.get("population_prevalence", 0.0),
                     sig.get("multi_account_same_beneficiary", 0.0) * 0.6)
    systemic = prevalence >= 0.5
    return {
        "systemic": systemic,
        "prevalence": round(prevalence, 3),
        "verdict": ("A population is exploiting this gap; the policy is the failure, fix it first"
                    if systemic else "Isolated actor; block and monitor for spread"),
    }


def synthesize_control(family: str) -> dict:
    """Propose the concrete control that closes the gap (the Rule Factory candidate)."""
    if family not in _CONTROLS:
        return {"control": None, "mechanism": None,
                "rationale": "No exploit family identified; nothing to close."}
    c = _CONTROLS[family]
    return {
        "control": c["control"],
        "mechanism": c["mechanism"],
        "rationale": "Closing the gap removes the exploit for everyone, not just this actor. Route to the Rule Factory as a candidate control and stress-test before deploy.",
    }


def recommend_loophole_action(exploit: dict, systemic: dict) -> dict:
    """Block the exploit, close the gap, and watch for migration to the next edge. When the
    exploit is systemic, prioritise the patch over mass enforcement."""
    fam = exploit.get("family")
    if not fam:
        return {"posture": "MONITOR", "primary_action": "No exploit detected; monitor only",
                "rationale": "No loophole signal.", "steps": ["Proceed normally"],
                "control": None, "reportable": False}

    control = synthesize_control(fam)
    if systemic.get("systemic"):
        return {
            "posture": "CLOSE-GAP (systemic) + MONITOR-MIGRATION",
            "primary_action": "Patch the policy first; do not mass-punish the population",
            "rationale": "At population scale the control gap is the fraud. Enforcing against thousands of customers who found an open edge is neither fair nor scalable; close it and reserve enforcement for the organised operators exploiting it at volume.",
            "steps": ["Deploy the synthesized control to close the gap for everyone",
                      "Quantify the leak (exposure = prevalence x per-actor benefit x population)",
                      "Reserve enforcement for high-volume / organised abusers only",
                      "Monitor for actors migrating to the next adjacent edge"],
            "control": control,
            "reportable": False,
        }
    return {
        "posture": "BLOCK-EXPLOIT + CLOSE-GAP + MONITOR-MIGRATION",
        "primary_action": "Block this actor's exploit path and close the gap",
        "rationale": "An isolated actor exploiting a control gap. Block the path, but fix the gap so the next actor cannot repeat it.",
        "steps": ["Block or reverse the specific exploit for this actor",
                  "Route the synthesized control to the Rule Factory and stress-test it",
                  "Monitor for other actors walking the same edge (rising prevalence flips this to systemic)"],
        "control": control,
        "reportable": exploit.get("confidence", 0.0) >= 0.7 and fam in ("threshold_arbitrage",),
    }


def assess_loophole(signals: dict) -> dict:
    """One call: exploit family + systemic test + synthesized closing control + response."""
    exploit = detect_exploit(signals)
    systemic = is_systemic(signals)
    action = recommend_loophole_action(exploit, systemic)
    return {"exploit": exploit, "systemic": systemic, "action": action}
