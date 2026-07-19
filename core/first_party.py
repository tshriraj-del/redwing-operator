"""
core/first_party.py - first-party / friendly-fraud intent (novel module).

Third-party fraud is someone using another person's identity. FIRST-PARTY fraud is the
genuine account holder defrauding the institution with their own real identity: friendly
chargebacks ("item not received" on a delivered order), returns/refund abuse, deposit and
bust-out fraud (build credit, max out, vanish), deliberate application inflation. It is one
of the largest and least-modelled loss categories, and it is HARD because the offender
looks exactly like a good customer, and because many disputes are genuine.

The ethical mirror of scam_arc. There, the job is to protect the genuine victim. Here, the
job is to stop a fraudster HIDING behind victimhood, without ever denying a real victim
their consumer-protection rights. So this module runs on a presumption of good faith and
only moves off "genuine" when there is real behavioural evidence of abuse.

It reads the serious-offender-mindset tells the user flagged: moral licensing ("the bank
can afford it"), selective high-value disputes, serial disputing, the bust-out shape, and
coached dispute templates. It grades intent on a spectrum, genuine -> opportunistic ->
serial -> bust-out, plus a coached branch that hands back to scam_arc when the "customer"
is really a victim being told to dispute.

Pure Python, deterministic, unit-testable. Flat signal dicts like the rest of the layer.
"""

from __future__ import annotations

FPF_INTENTS = {
    "genuine":       "Genuine dispute / good-faith account holder",
    "opportunistic": "Opportunistic friendly fraud (one-off, moral-licensing)",
    "serial":        "Serial / professional first-party abuser",
    "bust_out":      "Bust-out (build credit, max out, default)",
    "coached":       "Coached to dispute (may be a scam victim, verify)",
}

# tell -> {intent: weight}. `signals` is a flat dict of tell -> strength (0-1).
FPF_TELLS = {
    # opportunistic / friendly-fraud
    "dispute_after_delivery_confirmed":     {"opportunistic": 0.55, "serial": 0.25},
    "dispute_after_full_usage":             {"opportunistic": 0.55, "serial": 0.25},
    "moral_licensing_language":             {"opportunistic": 0.6},
    "returns_abuse_wardrobing":             {"opportunistic": 0.45, "serial": 0.35},
    # serial / professional
    "serial_disputer":                      {"serial": 0.8},
    "selective_high_value_disputes":        {"serial": 0.5, "opportunistic": 0.2},
    "prior_disputes_lost":                  {"serial": 0.6},
    "refund_rebuy_loop":                    {"serial": 0.5},
    "disputes_across_many_merchants":       {"serial": 0.55},
    # bust-out
    "credit_build_then_maxout":             {"bust_out": 0.8},
    "balance_maxed_pre_signal":             {"bust_out": 0.6},
    "contact_change_pre_default":           {"bust_out": 0.55},
    "application_overstatement_deliberate": {"bust_out": 0.4, "serial": 0.3},
    "never_intended_to_repay":              {"bust_out": 0.7},
    # coached (possible victim)
    "coached_dispute_template":             {"coached": 0.6, "serial": 0.2},
    "coached_by_third_party":               {"coached": 0.75},
}

# Contra-signals: evidence the account holder is acting in good faith. These defend the
# presumption of innocence so a real victim is never treated as a fraudster.
GENUINE_TELLS = {
    "merchant_error_evidence":     0.8,   # the merchant actually erred (double charge, wrong item)
    "first_dispute_ever":          0.4,
    "cooperative_provides_evidence": 0.5,
    "long_good_standing":          0.4,
    "unauthorised_use_corroborated": 0.7, # genuinely was not them (real third-party fraud)
}


def _clamp(x) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def classify_intent(signals: dict) -> dict:
    """Grade first-party intent on the presumption of good faith. Only moves off 'genuine'
    with real behavioural evidence of abuse. Returns intent, confidence, and drivers."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}

    scores = {i: 0.0 for i in FPF_INTENTS if i != "genuine"}
    drivers = {i: [] for i in scores}
    for tell, strength in sig.items():
        if strength <= 0:
            continue
        for intent, w in FPF_TELLS.get(tell, {}).items():
            scores[intent] += strength * w
            drivers[intent].append(tell)

    genuine_support = sum(sig.get(t, 0.0) * w for t, w in GENUINE_TELLS.items())
    duress = sig.get("duress", 0.0) >= 0.5

    # Precedence: most severe / most protective first, each gated by real evidence.
    if scores["bust_out"] >= 0.6:
        intent = "bust_out"
    elif scores["serial"] >= 0.6:
        intent = "serial"
    elif (scores["coached"] >= 0.5) or (duress and scores["coached"] > 0):
        intent = "coached"
    elif scores["opportunistic"] >= 0.5 and scores["opportunistic"] > genuine_support:
        intent = "opportunistic"
    else:
        intent = "genuine"

    if intent == "genuine":
        conf = round(min(1.0, 0.5 + genuine_support / 2), 3)
        used_drivers = [t for t in GENUINE_TELLS if sig.get(t, 0.0) > 0]
    else:
        total = sum(scores.values()) or 1.0
        conf = round(scores[intent] / total, 3)
        used_drivers = drivers[intent]

    return {
        "intent": intent,
        "intent_label": FPF_INTENTS[intent],
        "confidence": conf,
        "genuine_support": round(genuine_support, 3),
        "drivers": used_drivers,
        "scores": {i: round(v, 3) for i, v in scores.items() if v > 0},
        "presumed_genuine": intent == "genuine",
    }


def recommend_fpf_action(intent_read: dict) -> dict:
    """Map the intent read to a proportionate response. Honour genuine disputes; recover and
    educate the opportunist; represent and restrict the serial abuser; contain the bust-out;
    verify the coached before doing anything to the person."""
    intent = intent_read["intent"]

    if intent == "genuine":
        return {
            "posture": "HONOUR",
            "primary_action": "Uphold the dispute / refund; no penalty",
            "rationale": "No evidence of abuse. Consumer-protection rights and the presumption of good faith govern; a genuine claimant is never treated as a fraudster.",
            "steps": ["Process the chargeback / refund in the customer's favour",
                      "Log the evidence for pattern learning without penalising the customer"],
            "reportable": False, "punitive": False,
        }
    if intent == "opportunistic":
        return {
            "posture": "RECOVER + FRICTION + EDUCATE",
            "primary_action": "Represent with evidence, recover the loss, and warn",
            "rationale": "A one-off friendly-fraud attempt with moral-licensing tells. Low recidivism; a proportionate challenge plus education usually resolves it without escalation.",
            "steps": ["Represent the chargeback with delivery / usage evidence",
                      "Send a clear, non-accusatory notice that the claim did not match the record",
                      "Add light monitoring on the next dispute from this account"],
            "reportable": False, "punitive": False,
        }
    if intent == "serial":
        return {
            "posture": "REPRESENT + RESTRICT",
            "primary_action": "Fight the chargebacks and restrict dispute privileges",
            "rationale": "A repeat, selective abuser exploiting dispute rights as a discount. Durable friction on the abuse channel is warranted while preserving genuine recourse.",
            "steps": ["Represent every disputed transaction with full evidence",
                      "Restrict or manually-review this account's future disputes",
                      "Consider account closure with notice if the pattern persists",
                      "File a SAR if the volume / value crosses the reporting threshold"],
            "reportable": True, "punitive": True,
        }
    if intent == "bust_out":
        return {
            "posture": "CONTAIN + LOSS-MITIGATE + SAR",
            "primary_action": "Freeze exposure and contain the bust-out",
            "rationale": "A classic bust-out shape: credit built then drawn to the limit with no intent to repay. The priority is to stop further exposure before default.",
            "steps": ["Freeze / reduce the credit line and block further draw-down",
                      "Accelerate collections and secure any recoverable balance",
                      "File a SAR on the first-party fraud pattern",
                      "Quarantine linked accounts / devices for the network graph"],
            "reportable": True, "punitive": True,
        }
    # coached
    return {
        "posture": "VERIFY-COERCION",
        "primary_action": "Establish whether this is abuse or a coached victim before acting",
        "rationale": "The dispute uses coached / templated wording. This is either first-party-fraud-as-a-service or a scam victim being told to dispute; the two demand opposite responses.",
        "steps": ["Pause any punitive step and open an out-of-band conversation",
                  "If duress or scam grooming appears, route to scam_arc victim safeguarding",
                  "If it is a sold dispute template with no victim signal, treat as serial abuse"],
        "reportable": False, "punitive": False,
    }


def assess_first_party(signals: dict) -> dict:
    """One call: first-party intent + the proportionate response."""
    intent_read = classify_intent(signals)
    action = recommend_fpf_action(intent_read)
    return {"intent": intent_read, "action": action}
