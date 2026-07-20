"""
core/scam_arc.py - the Scam Kill-Chain and Coercion-in-Flight detector (novel module).

motive.py reads the OFFENDER: who is attacking and why. This is its mirror - the VICTIM
side that almost no platform models. In an authorised push payment (APP) scam the customer
sends the money themselves, willingly, under manipulation. Every offender-centric control
waves it through: the device is trusted, the identity is real, the payment is authenticated.
The harm is the authorised payment, and the person is a victim, not a fraudster.

This module models three things platforms omit:

  1. THE GROOMING ARC. A scam is not an event, it is a relationship with stages
     (contact -> grooming -> hook -> escalation -> extraction -> re-victimization). We
     locate where the person is on that arc from behavioural + transactional tells, per
     scam playbook (romance / pig-butchering, investment, impersonation-"safe account",
     tech-support, employment-mule, recovery). Where they are changes the right response
     more than the amount does.

  2. COERCION IN FLIGHT. At the moment of payment, is someone being coached right now? A
     live call open, script-reading cadence, remote-access active, duress. This is the
     break-glass, real-time read that turns an authorised payment into a protected one.

  3. THE SECOND-LOSS GUARD. Prior victims are farmed for a recovery scam ("we can get your
     money back, just pay this fee"). Detect the re-victimization and refuse to let a
     victim be drained twice.

Interventions are STAGE-APPROPRIATE and PROTECTIVE, never punitive: educate early, warn
and add friction in the middle, hard-stop with safeguarding and a break-the-spell script
at extraction. The governing principle: protect the person from their own authorised
payment. Pure Python, deterministic, unit-testable.
"""

from __future__ import annotations

# -- The universal grooming arc (all manipulation scams walk it) ----------------
SCAM_ARC = [
    {"stage": 0, "key": "contact",         "label": "Contact",                    "money_moving": False},
    {"stage": 1, "key": "grooming",        "label": "Grooming / trust-building",  "money_moving": False},
    {"stage": 2, "key": "hook",            "label": "The hook (first commitment)", "money_moving": True},
    {"stage": 3, "key": "escalation",      "label": "Escalation",                 "money_moving": True},
    {"stage": 4, "key": "extraction",      "label": "Extraction (the drain)",     "money_moving": True},
    {"stage": 5, "key": "revictimization", "label": "Re-victimization (recovery scam)", "money_moving": True},
]

# -- Scam playbooks (the recognisable manipulation scripts) ---------------------
SCAM_PLAYBOOKS = {
    "romance_pig_butchering": "Romance / pig-butchering (grooming into a fake investment)",
    "investment_crypto":      "Investment / crypto 'too good to be true' scam",
    "impersonation_safe_account": "Bank / police / government impersonation ('move to a safe account')",
    "tech_support":           "Tech-support / remote-access scam",
    "employment_mule":        "Employment / task scam (recruitment into muling)",
    "recovery_scam":          "Recovery / refund scam (re-victimizing a prior victim)",
}

# -- Victim-side tells: distinct from offender tells and onboarding tells -------
# tell -> {stage it evidences, {playbook: weight}, w: strength weight, live: fires at the
# moment of payment}. `signals` passed in is a dict of tell -> strength (0-1).
VICTIM_TELLS = {
    # contact / grooming (no money yet)
    "unsolicited_contact":       {"stage": 0, "w": 0.4, "live": False,
                                  "playbooks": {"romance_pig_butchering": 0.4, "employment_mule": 0.4, "investment_crypto": 0.3}},
    "online_only_relationship":  {"stage": 1, "w": 0.6, "live": False,
                                  "playbooks": {"romance_pig_butchering": 0.8}},
    "never_met_in_person":       {"stage": 1, "w": 0.5, "live": False,
                                  "playbooks": {"romance_pig_butchering": 0.7}},
    "authority_urgency":         {"stage": 1, "w": 0.6, "live": False,
                                  "playbooks": {"impersonation_safe_account": 0.8, "tech_support": 0.4}},
    "too_good_returns":          {"stage": 1, "w": 0.6, "live": False,
                                  "playbooks": {"investment_crypto": 0.8, "romance_pig_butchering": 0.4}},
    "isolation_secrecy":         {"stage": 1, "w": 0.7, "live": False,   # hides it from family/staff
                                  "playbooks": {"romance_pig_butchering": 0.5, "impersonation_safe_account": 0.5, "investment_crypto": 0.4}},
    # hook (first small commitment)
    "first_payee_new_crypto":    {"stage": 2, "w": 0.7, "live": False,
                                  "playbooks": {"investment_crypto": 0.7, "romance_pig_butchering": 0.6}},
    "small_test_payment":        {"stage": 2, "w": 0.5, "live": False,
                                  "playbooks": {"investment_crypto": 0.5, "romance_pig_butchering": 0.5, "employment_mule": 0.4}},
    "shown_fake_gains":          {"stage": 2, "w": 0.6, "live": False,   # a fake dashboard shows profit
                                  "playbooks": {"investment_crypto": 0.7, "romance_pig_butchering": 0.6}},
    # escalation
    "escalating_transfers":      {"stage": 3, "w": 0.7, "live": False,
                                  "playbooks": {"investment_crypto": 0.6, "romance_pig_butchering": 0.6, "impersonation_safe_account": 0.5}},
    "source_of_funds_new":       {"stage": 3, "w": 0.7, "live": False,   # loan / liquidation / savings drained to fund it
                                  "playbooks": {"investment_crypto": 0.6, "romance_pig_butchering": 0.5}},
    "account_drain_sequence":    {"stage": 3, "w": 0.6, "live": False,   # savings -> current -> out
                                  "playbooks": {"impersonation_safe_account": 0.6, "investment_crypto": 0.4}},
    "purpose_evasive":           {"stage": 3, "w": 0.6, "live": False,   # will not / cannot explain the payment
                                  "playbooks": {"impersonation_safe_account": 0.5, "romance_pig_butchering": 0.4, "investment_crypto": 0.4}},
    # extraction (the drain, often with a live handler)
    "fee_to_release":            {"stage": 4, "w": 0.8, "live": False,   # pay a tax/fee to unlock a bigger sum
                                  "playbooks": {"investment_crypto": 0.8, "romance_pig_butchering": 0.6, "recovery_scam": 0.6}},
    "safe_account_move":         {"stage": 4, "w": 0.8, "live": False,   # told to move money to a "safe account"
                                  "playbooks": {"impersonation_safe_account": 0.9}},
    "remote_access_active":      {"stage": 4, "w": 0.8, "live": True,    # AnyDesk/TeamViewer open during payment
                                  "playbooks": {"tech_support": 0.9, "impersonation_safe_account": 0.4}},
    "coaching_copresence":       {"stage": 4, "w": 0.8, "live": True,    # a call is live while they pay
                                  "playbooks": {"impersonation_safe_account": 0.6, "tech_support": 0.5, "investment_crypto": 0.4, "romance_pig_butchering": 0.4}},
    "script_reading":            {"stage": 4, "w": 0.7, "live": True,    # pause-type-pause, pasted payee block
                                  "playbooks": {"impersonation_safe_account": 0.5, "investment_crypto": 0.4}},
    "duress":                    {"stage": 4, "w": 0.9, "live": True,    # distress / fear cues
                                  "playbooks": {"impersonation_safe_account": 0.5, "tech_support": 0.4}},
    "coached_answers":           {"stage": 4, "w": 0.7, "live": True,    # answers to purpose questions are being fed
                                  "playbooks": {"impersonation_safe_account": 0.6, "tech_support": 0.5}},
    # re-victimization
    "prior_victim":              {"stage": 5, "w": 0.7, "live": False,
                                  "playbooks": {"recovery_scam": 0.9}},
    "recovery_promise":          {"stage": 5, "w": 0.8, "live": False,   # "we can get your lost money back"
                                  "playbooks": {"recovery_scam": 0.9}},
}

_LIVE_TELLS = tuple(t for t, s in VICTIM_TELLS.items() if s["live"])


def _clamp(x) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def locate_on_arc(signals: dict) -> dict:
    """Locate the person on the grooming arc and name the most likely playbook.
    The stage is the furthest-progressed stage with real evidence, because a scam only
    goes forward. Returns stage, playbook, confidence, and the tells that placed them."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}

    playbook_scores = {p: 0.0 for p in SCAM_PLAYBOOKS}
    stage_evidence = {s["stage"]: 0.0 for s in SCAM_ARC}
    drivers_by_stage = {s["stage"]: [] for s in SCAM_ARC}

    for tell, strength in sig.items():
        spec = VICTIM_TELLS.get(tell)
        if not spec or strength <= 0:
            continue
        ev = strength * spec["w"]
        stage_evidence[spec["stage"]] = max(stage_evidence[spec["stage"]], ev)
        drivers_by_stage[spec["stage"]].append(tell)
        for pb, w in spec["playbooks"].items():
            playbook_scores[pb] += strength * w

    # Furthest stage with meaningful evidence (a scam does not regress).
    reached = [s for s, ev in stage_evidence.items() if ev >= 0.3]
    stage_idx = max(reached) if reached else 0
    stage = SCAM_ARC[stage_idx]

    total_pb = sum(playbook_scores.values()) or 1.0
    top_pb = max(playbook_scores, key=playbook_scores.get)
    pb_conf = round(playbook_scores[top_pb] / total_pb, 3)

    # Confidence in the placement: strongest evidence at or before this stage.
    placed_conf = round(max((stage_evidence[s] for s in range(stage_idx + 1)), default=0.0), 3)

    # Gather the drivers from this stage and the ones before it (the arc so far).
    arc_drivers = []
    for s in range(stage_idx + 1):
        arc_drivers.extend(drivers_by_stage[s])

    return {
        "stage": stage_idx,
        "stage_key": stage["key"],
        "stage_label": stage["label"],
        "money_moving": stage["money_moving"],
        "placement_confidence": placed_conf,
        "playbook": top_pb if playbook_scores[top_pb] > 0 else None,
        "playbook_label": SCAM_PLAYBOOKS[top_pb] if playbook_scores[top_pb] > 0 else None,
        "playbook_confidence": pb_conf,
        "playbook_scores": {p: round(v, 3) for p, v in playbook_scores.items() if v > 0},
        "drivers": arc_drivers,
        "on_arc": placed_conf >= 0.3,
    }


def coercion_in_flight(signals: dict) -> dict:
    """The moment-of-payment read: is someone being coached or coerced right now? This is
    the break-glass signal that turns an authorised payment into a protected one."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    live = {t: sig.get(t, 0.0) for t in _LIVE_TELLS if sig.get(t, 0.0) > 0}
    if not live:
        return {"coached": False, "duress": False, "live_score": 0.0, "tells": {}}

    # Weighted live pressure; duress and remote-access are the strongest tells.
    score = 0.0
    for tell, strength in live.items():
        score = max(score, strength * VICTIM_TELLS[tell]["w"])
    duress = sig.get("duress", 0.0) >= 0.5
    remote = sig.get("remote_access_active", 0.0) >= 0.5
    return {
        "coached": score >= 0.5,
        "duress": duress,
        "remote_access": remote,
        "live_score": round(score, 3),
        "tells": {t: round(s, 3) for t, s in live.items()},
    }


# -- Stage-appropriate protective intervention ---------------------------------
def protect(arc: dict, live: dict) -> dict:
    """Map (stage x playbook x live-coercion) to a proportionate, PROTECTIVE response.
    Early: educate. Middle: warn + friction. Extraction / live coercion: hard-stop the
    payment, cool off, break the spell, safeguard. Never punish the victim."""
    stage = arc.get("stage", 0)
    playbook = arc.get("playbook")

    # -- Break-glass: live coercion at the moment of payment overrides the stage --
    if live.get("coached") or live.get("duress"):
        steps = [
            "Hard-stop this payment now; do not release it despite valid authentication",
            "Open a cooling-off hold and a scam-specific warning interstitial",
            "Move to a private, out-of-band channel the handler cannot hear",
        ]
        if live.get("remote_access"):
            steps.insert(1, "Tell the customer to disconnect any remote-access tool immediately; no genuine bank or agency needs it")
        if live.get("duress"):
            steps.append("Treat as a vulnerable-customer / duress case; involve safeguarding and, if there is a safety risk, the police")
        steps.append("Ask break-the-spell questions: who told you to send this, are they on the line now, did anyone tell you to keep it secret")
        return {
            "posture": "VICTIM-PROTECT (break-glass)",
            "primary_action": "Stop the payment and protect the person in real time",
            "rationale": "Someone is coaching or coercing this payment as it happens. The authorised transfer is the harm; the customer is a victim, not a fraudster.",
            "steps": steps,
            "punitive": False,
            "reportable_as_scam": True,
        }

    # -- Re-victimization: the second-loss guard --
    if stage == 5 or playbook == "recovery_scam":
        return {
            "posture": "VICTIM-PROTECT (second-loss guard)",
            "primary_action": "Block the recovery payment and warn about the refund scam",
            "rationale": "Prior victims are farmed for a recovery scam. No legitimate recovery service asks a victim to pay a fee up front; this is a second loss in progress.",
            "steps": [
                "Block or hold the fee payment",
                "Explain that recovery / refund fees are themselves a scam and no fee unlocks stolen funds",
                "Route to victim support and log the recovery attempt on the case",
                "Flag the account so the next recovery approach is caught early",
            ],
            "punitive": False,
            "reportable_as_scam": True,
        }

    # -- Extraction: the drain, hard friction + named-scam warning --
    if stage == 4:
        return {
            "posture": "HARD-FRICTION + WARN",
            "primary_action": "Cool-off hold with a named-scam warning and a purpose check",
            "rationale": "The pattern matches the extraction phase of a scam. A specific, named warning at the moment of payment is far more effective than a generic one.",
            "steps": [
                "Introduce a cooling-off delay before the payment can complete",
                f"Show a warning naming the likely scam: {arc.get('playbook_label') or 'a known scam pattern'}",
                "Ask the customer to state the purpose of the payment in their own words",
                "If a fee-to-release or safe-account narrative appears, block and escalate to a scam specialist",
            ],
            "punitive": False,
            "reportable_as_scam": True,
        }

    # -- Escalation: friction + effective warning, invite a trusted second opinion --
    if stage == 3:
        return {
            "posture": "FRICTION + WARN",
            "primary_action": "Step-up with a scam-pattern warning and a trusted-contact nudge",
            "rationale": "Payments are escalating along a recognisable scam script. Interrupting the pattern early, before extraction, prevents the largest losses.",
            "steps": [
                "Step-up verification on the payment and slow the cadence",
                f"Warn that this pattern is consistent with {arc.get('playbook_label') or 'a scam'}",
                "Encourage the customer to check with a trusted person or the real organisation via a number they look up themselves",
                "Watch for source-of-funds changes (a new loan or liquidation to fund it)",
            ],
            "punitive": False,
            "reportable_as_scam": True,
        }

    # -- Hook: gentle friction + education at the first commitment --
    if stage == 2:
        return {
            "posture": "EDUCATE + LIGHT-FRICTION",
            "primary_action": "Educate at the first payment and add a light checkpoint",
            "rationale": "This is the first small commitment. A well-timed nudge here, before trust deepens and amounts grow, is the highest-leverage moment to break the arc.",
            "steps": [
                "Show a short, relevant caution about the specific scam type at the first payment",
                "Ask one purpose question and confirm the payee is who they think it is",
                "Set a low-friction watch for escalation on this payee",
            ],
            "punitive": False,
            "reportable_as_scam": False,
        }

    # -- Contact / grooming: no money is moving yet, so educate, do not gate --
    if arc.get("on_arc"):
        return {
            "posture": "EDUCATE",
            "primary_action": "Gentle, non-punitive education; no payment friction",
            "rationale": "No money is moving yet. The right move is awareness, not friction, so the customer recognises the manipulation themselves before the hook.",
            "steps": [
                "Surface a relevant, non-alarming awareness message about this scam pattern",
                "Offer an easy way to report a suspicious contact",
                "Note the early signal on the profile without penalising the customer",
            ],
            "punitive": False,
            "reportable_as_scam": False,
        }

    return {
        "posture": "MONITOR",
        "primary_action": "No scam arc detected; monitor only",
        "rationale": "No meaningful victim-side signal. Do not add friction to a normal customer.",
        "steps": ["Proceed normally", "Keep the victim-side detectors running in the background"],
        "punitive": False,
        "reportable_as_scam": False,
    }


def assess_scam(signals: dict) -> dict:
    """One call: locate the victim on the grooming arc, read live coercion, and return the
    stage-appropriate protective intervention."""
    arc = locate_on_arc(signals)
    live = coercion_in_flight(signals)
    intervention = protect(arc, live)
    return {"arc": arc, "live": live, "intervention": intervention}
