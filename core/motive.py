"""
core/motive.py - Actor and Motive Intelligence (the novel layer).

Fraud platforms score WHAT (typology) and HOW MUCH (a 0-1 risk). They rarely infer WHY
(motive) or WHO (offender lifecycle), yet those change the correct RESPONSE more than
the score does. This module adds:

  1. a registry of novel behavioural tells that platforms do not model explicitly,
  2. motive inference over those tells,
  3. offender lifecycle + severity (a population with trajectories, not a binary flag),
  4. an intervention matrix that maps (motive x severity x victim-status) to a response
     on a full spectrum - EDUCATE / SUPPORT ... BLOCK / SAR / LAW-ENFORCEMENT, plus
     VICTIM-PROTECT - i.e. a rehabilitative + protective + enforcement posture, not a
     binary gate.

Pure Python, deterministic, unit-testable.
"""

from __future__ import annotations

MOTIVES = {
    "coerced_victim":      "Coerced or coached victim",
    "survival":            "Survival-driven",
    "need":                "Need-based",
    "loophole":            "Loophole exploitation",
    "opportunistic":       "Opportunistic",
    "income_source":       "Professional / income-source",
    "organized_malicious": "Organized malicious",
}

# Novel behavioural tells: signals almost no platform models as first-class, each
# pointing at one or more motives. `signals` passed to infer_motive is a dict of
# tell -> strength (0-1).
BEHAVIORAL_TELLS = {
    "duress":                  "Signs the actor is acting under coercion",
    "coaching_copresence":     "A call/second app is active while the actor transacts (someone else is driving)",
    "script_reading":          "Pause-type-pause cadence and pasted payee details: reading a scammer's script",
    "reverse_familiarity":     "Unnatural fluency or unfamiliarity with the actor's own claimed identity",
    "hesitation_entropy":      "Micro-timing of PII entry inconsistent with a lived-in identity",
    "survival_spend":          "Small, essential-category spend timed to benefit cycles",
    "benefit_timing":          "Activity aligned to a benefits calendar",
    "essential_category":      "Spend concentrated in food / utilities / rent",
    "desperation_velocity":    "Escalating attempts across products in a short window",
    "application_inflation":   "Overstated income/assets on an application",
    "moral_licensing":         "Rationalisation language in disputes ('I deserve this')",
    "windfall_kept":           "A one-off mistake (misdirected funds) exploited",
    "boundary_probing":        "Systematic testing of a policy/limit edge",
    "threshold_walking":       "Amounts that walk just under a control threshold",
    "automation_scalable":     "Scripted, scalable execution",
    "professional_execution":  "Calm, methodical, high-competence execution",
    "sophisticated_tooling":   "Anti-detection tooling (emulators, fingerprint spoofing)",
    "shared_device_ring":      "Device/identity shared across a cluster of accounts",
    "cross_border_coordination":"Coordinated activity across jurisdictions",
    "multi_typology":          "One actor spanning several fraud typologies",
    "escalation_pattern":      "Rising amounts / frequency over time",
}

# tell -> {motive: weight}
_CONTRIB = {
    "duress":                   {"coerced_victim": 1.0},
    "coaching_copresence":      {"coerced_victim": 0.85},
    "script_reading":           {"coerced_victim": 0.7},
    "reverse_familiarity":      {"income_source": 0.3, "organized_malicious": 0.3},
    "hesitation_entropy":       {"income_source": 0.2},
    "survival_spend":           {"survival": 0.85, "need": 0.3},
    "benefit_timing":           {"survival": 0.6},
    "essential_category":       {"survival": 0.5, "need": 0.25},
    "desperation_velocity":     {"need": 0.75, "survival": 0.4},
    "application_inflation":    {"need": 0.65},
    "moral_licensing":          {"opportunistic": 0.55, "need": 0.2},
    "windfall_kept":            {"opportunistic": 0.75},
    "boundary_probing":         {"loophole": 0.9},
    "threshold_walking":        {"loophole": 0.7},
    "automation_scalable":      {"loophole": 0.4, "income_source": 0.4},
    "professional_execution":   {"income_source": 0.8},
    "sophisticated_tooling":    {"income_source": 0.6, "organized_malicious": 0.45},
    "shared_device_ring":       {"income_source": 0.6, "organized_malicious": 0.6},
    "cross_border_coordination":{"organized_malicious": 0.8},
    "multi_typology":           {"organized_malicious": 0.6},
    "escalation_pattern":       {"organized_malicious": 0.5, "income_source": 0.3},
}


def infer_motive(signals: dict) -> dict:
    """signals: {tell: strength 0-1}. Returns the most likely motive with confidence
    and the tells that drove it. Coercion dominates when present - a victim is never
    scored as an offender just because the transaction looks bad."""
    scores = {m: 0.0 for m in MOTIVES}
    drivers_by_motive = {m: [] for m in MOTIVES}
    for tell, strength in (signals or {}).items():
        s = max(0.0, min(1.0, float(strength or 0.0)))
        if s <= 0:
            continue
        for motive, w in _CONTRIB.get(tell, {}).items():
            scores[motive] += s * w
            drivers_by_motive[motive].append(tell)

    # Victim protection overrides: strong coercion signal wins outright.
    if scores["coerced_victim"] >= 0.7:
        top = "coerced_victim"
    else:
        top = max(scores, key=scores.get)

    total = sum(scores.values()) or 1.0
    confidence = round(scores[top] / total, 3)
    return {
        "motive": top,
        "motive_label": MOTIVES[top],
        "confidence": confidence,
        "drivers": drivers_by_motive[top],
        "scores": {m: round(v, 3) for m, v in scores.items() if v > 0},
        "is_victim": top == "coerced_victim",
    }


def offender_profile(signals: dict, motive: str) -> dict:
    """Where the actor sits on the offender lifecycle, and how severe. A coerced victim
    is not an offender. Otherwise: first_time -> escalating -> professional ->
    ring_operator, each with recidivism and escalation risk."""
    s = signals or {}
    def g(k):
        return max(0.0, min(1.0, float(s.get(k, 0.0) or 0.0)))

    if motive == "coerced_victim":
        return {"lifecycle": "coerced_victim", "severity": "victim", "severity_score": 0,
                "recidivism_risk": None, "escalation_risk": None}

    ring   = max(g("shared_device_ring"), g("cross_border_coordination"))
    prof   = max(g("professional_execution"), g("sophisticated_tooling"), g("automation_scalable"))
    escal  = g("escalation_pattern")

    if ring >= 0.6 and prof >= 0.5:
        lifecycle = "ring_operator"
    elif prof >= 0.6:
        lifecycle = "professional"
    elif escal >= 0.5:
        lifecycle = "escalating"
    else:
        lifecycle = "first_time"

    base = {"first_time": 20, "escalating": 50, "professional": 75, "ring_operator": 92}[lifecycle]
    # need/survival motives soften severity; organized hardens it
    adj = {"need": -10, "survival": -15, "loophole": -5, "opportunistic": -8,
           "income_source": +5, "organized_malicious": +8}.get(motive, 0)
    severity_score = max(0, min(100, base + adj + int(escal * 10)))
    band = ("Low" if severity_score < 35 else "Medium" if severity_score < 65
            else "High" if severity_score < 85 else "Critical")
    return {
        "lifecycle": lifecycle,
        "severity": band,
        "severity_score": severity_score,
        "recidivism_risk": round(min(1.0, base / 100 + escal * 0.2), 3),
        "escalation_risk": round(escal, 3),
    }


# ── Intervention matrix ───────────────────────────────────────────────────────
# The centrepiece: (motive x severity x victim-status) -> a proportionate response,
# including the rehabilitative and victim-protective pathways platforms omit.

def recommend_intervention(motive: str, profile: dict) -> dict:
    """Map the actor read to the right response. Returns a posture on the spectrum and
    concrete, humane-where-appropriate steps."""
    sev = profile.get("severity_score", 50)

    if motive == "coerced_victim":
        return {
            "posture": "VICTIM-PROTECT",
            "primary_action": "Pause the payment and protect the person",
            "rationale": "The actor is being coached or coerced; the authorised payment is the harm, not the person.",
            "steps": ["Introduce a cooling-off hold and a scam-warning interstitial",
                      "Attempt a live out-of-band check on a trusted channel",
                      "Refer to safeguarding / vulnerable-customer support",
                      "Do NOT penalise or close the account"],
            "do_not": "Treat the victim as an offender or auto-close the account.",
            "reportable": False,
        }
    if motive == "survival":
        return {
            "posture": "SUPPORT",
            "primary_action": "Soft-decline with a support pathway",
            "rationale": "Behaviour is driven by hardship, not criminal enterprise; enforcement alone is disproportionate and ineffective.",
            "steps": ["Decline the specific request without a punitive flag",
                      "Offer a hardship / financial-support referral",
                      "Apply light monitoring rather than a block",
                      "Escalate only if the pattern turns professional"],
            "reportable": False,
        }
    if motive == "need":
        return {
            "posture": "FRICTION + EDUCATE",
            "primary_action": "Step up verification and correct the record",
            "rationale": "Need-based inflation has low recidivism; friction plus education resolves most cases without a block.",
            "steps": ["Step-up: request documentary proof of the inflated fields",
                      "Explain the correct process and re-invite a clean application",
                      "Approve with monitoring if verification clears"],
            "reportable": False,
        }
    if motive == "loophole":
        return {
            "posture": "BLOCK + CLOSE-GAP",
            "primary_action": "Block the exploit and close the policy gap",
            "rationale": "The system, not just the actor, is the failure; punishing the actor without fixing the gap invites the next one.",
            "steps": ["Block the specific exploit path",
                      "Route the gap to Rule Factory to synthesize a control",
                      "Monitor for other actors walking the same edge"],
            "reportable": sev >= 65,
        }
    if motive == "opportunistic":
        return {
            "posture": "STEP-UP",
            "primary_action": "Recover and step up",
            "rationale": "A one-off exploit of a mistake; recovery and friction are usually proportionate.",
            "steps": ["Attempt recovery / reversal of the windfall",
                      "Step-up on the next action from this account"],
            "reportable": False,
        }
    if motive == "income_source":
        return {
            "posture": "BLOCK + SAR",
            "primary_action": "Block, file a SAR, and open a ring investigation",
            "rationale": "Fraud is the actor's livelihood; high recidivism and likely ring connections warrant durable action.",
            "steps": ["Block the account and quarantine linked devices/identities",
                      "File a SAR on confirmation and threshold",
                      "Pivot to the fraud graph to map the ring"],
            "reportable": True,
        }
    # organized_malicious
    return {
        "posture": "BLOCK + SAR + LAW-ENFORCEMENT",
        "primary_action": "Block, file, and refer to law enforcement",
        "rationale": "Coordinated, escalating, cross-jurisdiction activity exceeds internal remedies.",
        "steps": ["Block and freeze linked accounts across the cluster",
                  "File a SAR and a fraud-ring referral",
                  "Coordinate with law-enforcement / cross-border channels",
                  "Feed the ring's signature back to the consortium"],
        "reportable": True,
    }


def assess_actor(signals: dict) -> dict:
    """One call: motive + offender profile + recommended intervention."""
    m = infer_motive(signals)
    p = offender_profile(signals, m["motive"])
    i = recommend_intervention(m["motive"], p)
    return {"motive": m, "offender": p, "intervention": i}
