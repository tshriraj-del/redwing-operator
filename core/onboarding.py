"""
core/onboarding.py - the Behavioral Onboarding Gauntlet (novel concept module).

Onboarding is where synthetic identities, stolen-identity accounts, and mule farms get
established, and where the industry default (a static KYC pass/fail checklist) is weakest:
it applies the same friction to everyone, so it is either too soft for fraudsters or too
harsh for real customers. This module is an ADAPTIVE gauntlet instead: it reads
behavioural micro-tells during the application and escalates friction in real time,
targeting the specific dimension that looks weak rather than throwing up a blanket wall.

Two things make it novel:
  1. TARGETED friction. A fresh/disposable email triggers email+domain verification; a
     headless device triggers device attestation; fumbling one's own DOB (reverse-
     familiarity) triggers knowledge-based auth on the claimed identity. Challenge the
     weak dimension, not the whole applicant.
  2. A COERCION OFF-RAMP. If the behaviour looks like someone being coached or coerced
     into opening an account (a trafficked mule), the gauntlet routes to protect-and-
     safeguard, never to a punitive decline. Onboarding is a place victims are created.

Consumes the device/identity attribute fabric (core/attributes.py). Pure, testable.
"""

from __future__ import annotations

from .attributes import evaluate as _evaluate_attributes

# ── The gauntlet: escalating tiers of friction ────────────────────────────────
GAUNTLET = [
    {"tier": 0, "name": "Frictionless",       "gate": "Form validation only"},
    {"tier": 1, "name": "Verify contactables", "gate": "Email + phone one-time-passcode"},
    {"tier": 2, "name": "Prove the identity",  "gate": "Document capture + liveness + PII-to-bureau match"},
    {"tier": 3, "name": "Human in the loop",   "gate": "Knowledge-based auth / video / analyst review"},
    {"tier": 4, "name": "Decline or refer",    "gate": "Decline; SAR if warranted; safeguard if coerced"},
]

# ── Onboarding behavioural tells (distinct from transaction tells) ─────────────
# tell -> {dimension it implicates, weight, whether it is a hard escalation trigger}
ONBOARDING_TELLS = {
    "pii_pasted":            {"dim": "knowledge",    "w": 0.35, "hard": False},  # SSN/DOB pasted, not typed
    "pii_hesitation":        {"dim": "knowledge",    "w": 0.55, "hard": False},  # unnatural rhythm on own PII (reverse-familiarity)
    "too_fast_entry":        {"dim": "device",       "w": 0.5,  "hard": False},  # scripted / bot
    "too_slow_lookup":       {"dim": "knowledge",    "w": 0.45, "hard": False},  # looking up stolen data
    "field_churn":           {"dim": "knowledge",    "w": 0.3,  "hard": False},  # many corrections
    "abandon_return_changed":{"dim": "knowledge",    "w": 0.5,  "hard": False},  # shopping for what passes
    "app_velocity":          {"dim": "coordination", "w": 0.5,  "hard": False},  # many applications from device
    "shared_cohort":         {"dim": "coordination", "w": 0.8,  "hard": True},   # farm signup: shared attributes across a cohort
    "scripted_timing":       {"dim": "device",       "w": 0.6,  "hard": False},  # automation cadence
    "coaching_pauses":       {"dim": "coercion",     "w": 0.7,  "hard": False},  # reading instructions
    "app_switching":         {"dim": "coercion",     "w": 0.6,  "hard": False},  # being walked through it
    "geo_mismatch":          {"dim": "document",     "w": 0.5,  "hard": False},  # IP vs claimed address
}

_DIMENSIONS = ("contactable", "document", "device", "knowledge", "coordination", "coercion")

_CHALLENGE = {
    "contactable":  ("Email + domain verification and a live phone OTP with a line-type check",
                     "the contactables look fresh or non-genuine"),
    "document":     ("Document capture, liveness, and a PII-to-bureau match",
                     "the claimed identity does not resolve cleanly to real records"),
    "device":       ("Device attestation and a bot / automation challenge",
                     "the device looks scripted, headless, or tampered"),
    "knowledge":    ("Knowledge-based auth on the CLAIMED identity",
                     "the applicant is unnaturally unfamiliar with, or too fluent on, their own data"),
    "coordination": ("Cohort review and a velocity hold across the shared attribute",
                     "the application is part of a coordinated signup cohort (a farm)"),
}


def assess_onboarding(applicant_id: str, typology: str = "", behavior=None,
                      attributes=None) -> dict:
    """Run the adaptive gauntlet for one application. `behavior` is a dict of onboarding
    tell -> strength (0-1). Pulls the device/identity attribute fabric for coherence.
    Returns the required tier, targeted challenges, and a decision with a coercion off-ramp."""
    behavior = {k: max(0.0, min(1.0, float(v or 0.0))) for k, v in (behavior or {}).items()}
    attr = attributes or _evaluate_attributes(applicant_id, typology)

    # Per-dimension weakness: seed from the attribute fabric, add the behavioural tells.
    dim = {d: 0.0 for d in _DIMENSIONS}
    # attribute-fabric contributions
    idv = attr["identity"]
    dv = attr["device"]
    if idv.get("Email intelligence", {}).get("disposable"):        dim["contactable"] = max(dim["contactable"], 0.7)
    if idv.get("Phone intelligence", {}).get("line_type") == "VOIP": dim["contactable"] = max(dim["contactable"], 0.6)
    dim["document"]  = max(dim["document"],  idv.get("Identity linkage", {}).get("synthetic_id_score", 0.0),
                           1 - idv.get("Doc & biometric", {}).get("id_doc_authentic", 1.0))
    dim["device"]    = max(dim["device"], dv.get("Integrity & tamper", {}).get("risk", 0.0),
                           dv.get("Network & connection", {}).get("risk", 0.0))

    hard = False
    drivers = []
    for tell, strength in behavior.items():
        spec = ONBOARDING_TELLS.get(tell)
        if not spec or strength <= 0:
            continue
        dim[spec["dim"]] = max(dim[spec["dim"]], strength * (0.6 + spec["w"] * 0.7))
        drivers.append(tell)
        if spec["hard"] and strength >= 0.6:
            hard = True

    coercion = dim["coercion"]
    dubious = round(max(v for k, v in dim.items() if k != "coercion"), 3)

    # ── Coercion off-ramp: a person being made to open a mule account is a victim ──
    if coercion >= 0.6:
        return {
            "applicant_id": applicant_id, "dubious_score": dubious, "coercion_score": round(coercion, 3),
            "tier": 4, "tier_name": "Decline or refer", "decision": "PROTECT",
            "posture": "VICTIM-PROTECT",
            "rationale": "The application looks coached or coerced. Onboarding is where mule accounts are created from victims; protect the person rather than punish them.",
            "challenges": [], "weak_dimensions": {k: round(v, 3) for k, v in dim.items() if v >= 0.4},
            "steps": ["Pause the application and open a safe out-of-band channel",
                      "Route to safeguarding / vulnerable-customer support",
                      "Do NOT decline-and-flag the person as a fraudster"],
            "drivers": drivers, "is_coerced": True,
            "attribute_surface": attr["surface"]["total"],
        }

    # ── Targeted challenges for the weak dimensions ──
    challenges = []
    for d in ("contactable", "document", "device", "knowledge", "coordination"):
        if dim[d] >= 0.5 and d in _CHALLENGE:
            challenge, why = _CHALLENGE[d]
            challenges.append({"dimension": d, "challenge": challenge, "why": why, "weakness": round(dim[d], 3)})

    # ── Adaptive tier: dubious score bands, with a hard-trigger override ──
    if hard or dubious >= 0.8:
        tier, decision = (3, "MANUAL REVIEW") if not (dubious >= 0.9) else (4, "DECLINE")
    elif dubious >= 0.6:
        tier, decision = 2, "STEP-UP"
    elif dubious >= 0.35 or challenges:
        tier, decision = 1, "STEP-UP"
    else:
        tier, decision = 0, "ONBOARD"

    return {
        "applicant_id": applicant_id,
        "dubious_score": dubious, "coercion_score": round(coercion, 3),
        "tier": tier, "tier_name": GAUNTLET[tier]["name"], "gate": GAUNTLET[tier]["gate"],
        "decision": decision,
        "posture": "GATE" if decision != "ONBOARD" else "PASS",
        "weak_dimensions": {k: round(v, 3) for k, v in dim.items() if v >= 0.4},
        "challenges": challenges,
        "rationale": (f"{len(challenges)} weak dimension(s) drove targeted friction to tier {tier}."
                      if decision != "ONBOARD"
                      else "No dubious behaviour; onboard with minimal friction to protect conversion."),
        "drivers": drivers,
        "is_coerced": False,
        "attribute_surface": attr["surface"]["total"],
    }
