"""
core/mule_network.py - the money-mule recruitment and network lifecycle (novel module).

A mule account is the cash-out valve of almost every scam: the victim's authorised
payment lands in a mule account, is layered, and is cashed out before it can be clawed
back. Platforms tend to treat "mule" as a single bad flag. It is not. A mule is a
LIFECYCLE and a SPECTRUM:

  - RECRUITMENT. Ordinary people are recruited into muling through recognisable funnels:
    a fake "payment-processing agent" job, a gamified task scam, a romance groomed into
    receiving money, social-media "easy money" DMs. Catching the funnel catches the mule
    before the first inbound.

  - WITTING-NESS SPECTRUM. unwitting (tricked, believes it is a real job) ->
    naive/willfully-blind (suspects, keeps a cut, looks away) -> witting (knowingly
    laundering for pay) -> herder (recruits and runs other mules). This changes the
    response from PROTECT to ENFORCE, the same rehabilitative-to-enforcement spectrum as
    the motive layer, applied to the mule.

  - ACCOUNT LIFECYCLE. dormant/pre-positioned -> activation -> receiving -> layering
    (rapid pass-through) -> cash-out -> burnout/rotation. Where an account is in that
    cycle changes what to do right now.

  - HERDING NETWORK. One controller runs many mule accounts: shared devices, a controlling
    login IP, synchronized timing, a fan-in / fan-out topology. This is where the mule
    layer meets the fraud graph.

The governing nuance platforms miss: a mule account holds a real victim's stolen money.
So the FUND action (freeze the pass-through to preserve the upstream victim's funds for
recovery) is decided separately from the PERSON action (how culpable is the account
holder). Freeze the money to protect the victim; treat the holder by their witting-ness.

Pure Python, deterministic, unit-testable. Consumes flat signal dicts like motive.py /
scam_arc.py so the whole actor layer speaks one language.
"""

from __future__ import annotations

# -- The witting-ness spectrum --------------------------------------------------
MULE_ROLES = {
    "unwitting":       "Unwitting mule (tricked; believes it is a legitimate job)",
    "naive_complicit": "Naive / willfully-blind mule (suspects, looks away, keeps a cut)",
    "witting":         "Witting mule (knowingly launders funds for payment)",
    "herder":          "Mule herder / controller (recruits and runs other mules)",
}

# tell -> {role: weight}. `signals` is a flat dict of tell -> strength (0-1).
ROLE_TELLS = {
    "believes_legitimate_job":    {"unwitting": 0.8},
    "forwards_full_amount":       {"unwitting": 0.45, "naive_complicit": 0.2},   # keeps nothing = pass-through
    "stops_on_warning":           {"unwitting": 0.7},                            # a warning breaks the spell
    "distressed_on_challenge":    {"unwitting": 0.5},
    "kept_small_cut":             {"naive_complicit": 0.6},
    "ignored_red_flags":          {"naive_complicit": 0.6},
    "continues_after_warning":    {"witting": 0.7, "naive_complicit": 0.2},      # warned and carries on
    "keeps_consistent_cut":       {"witting": 0.7},                              # a paid, repeat mule
    "many_victim_sources":        {"witting": 0.5, "herder": 0.3},              # receives from many unrelated parties
    "launder_language":           {"witting": 0.5, "herder": 0.35},            # coaches, uses laundering terms
    "recruits_others":            {"herder": 0.9},
    "controls_multiple_accounts": {"herder": 0.8},
}

# -- Recruitment funnels ---------------------------------------------------------
RECRUITMENT_CHANNELS = {
    "job_scam":       {"label": "Money-mule 'job' (fake payment-processing agent)",
                       "tells": {"job_ad_referral": 0.8, "believes_legitimate_job": 0.4}},
    "task_scam":      {"label": "Task / micro-job scam (gamified, escalating)",
                       "tells": {"task_scam_onboarding": 0.85}},
    "romance":        {"label": "Romance-to-mule pipeline (groomed into receiving)",
                       "tells": {"romance_to_mule": 0.9}},
    "social_recruit": {"label": "Social-media 'easy money' recruitment",
                       "tells": {"social_media_recruit": 0.8}},
    "crypto_gaming":  {"label": "Crypto / gaming-community recruitment",
                       "tells": {"gaming_crypto_community": 0.7}},
    "for_hire":       {"label": "Self-recruited / for-hire (witting)",
                       "tells": {"keeps_consistent_cut": 0.4, "launder_language": 0.4, "recruits_others": 0.3}},
}

# -- The account lifecycle (a mule account walks it) -----------------------------
MULE_LIFECYCLE = [
    {"phase": 0, "key": "dormant",    "label": "Dormant / pre-positioned"},
    {"phase": 1, "key": "activation", "label": "Activation"},
    {"phase": 2, "key": "receiving",  "label": "Receiving (funds in)"},
    {"phase": 3, "key": "layering",   "label": "Layering (rapid pass-through)"},
    {"phase": 4, "key": "cashout",    "label": "Cash-out"},
    {"phase": 5, "key": "burnout",    "label": "Burnout / rotation"},
]

LIFECYCLE_TELLS = {
    "dormant_reactivation":        1,
    "contact_change_pre_activity": 1,
    "first_inbound_unrelated":     2,
    "large_unexpected_inbound":    2,
    "rapid_passthrough":           3,
    "high_passthrough_ratio":      3,
    "structuring_below_threshold": 3,
    "multi_hop_layering":          3,
    "cashout_crypto":              4,
    "cashout_atm":                 4,
    "cashout_giftcard":            4,
    "collector_fanin":             4,
    "account_flagged_abandoned":   5,
}

# -- Herding / control signals (mule layer meets the fraud graph) ----------------
CONTROL_TELLS = {
    "shared_device_across_accounts": 0.8,
    "herder_ip_control":             0.7,   # a controlling login IP/ASN across the herd
    "synchronized_timing":           0.7,   # accounts act in lockstep
    "scripted_movement":             0.6,
    "fanin_fanout_topology":         0.7,   # classic mule-network shape
}

_ACTIVE_PHASES = (2, 3, 4)   # receiving / layering / cash-out = money is at risk right now


def _clamp(x) -> float:
    return max(0.0, min(1.0, float(x or 0.0)))


def classify_mule(signals: dict) -> dict:
    """Place the account holder on the witting-ness spectrum. Returns 'undetermined' when
    there is no behavioural evidence of intent, rather than assuming guilt or innocence."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    scores = {r: 0.0 for r in MULE_ROLES}
    drivers = {r: [] for r in MULE_ROLES}
    for tell, strength in sig.items():
        if strength <= 0:
            continue
        for role, w in ROLE_TELLS.get(tell, {}).items():
            scores[role] += strength * w
            drivers[role].append(tell)

    total = sum(scores.values())
    if total <= 0:
        return {"role": "undetermined", "role_label": "Intent not yet established",
                "confidence": 0.0, "drivers": [], "scores": {},
                "is_victim_adjacent": False}

    top = max(scores, key=scores.get)
    return {
        "role": top,
        "role_label": MULE_ROLES[top],
        "confidence": round(scores[top] / total, 3),
        "drivers": drivers[top],
        "scores": {r: round(v, 3) for r, v in scores.items() if v > 0},
        # an unwitting mule recruited via romance is really a scam victim: hand to scam_arc
        "is_victim_adjacent": top == "unwitting",
    }


def recruitment_channel(signals: dict) -> dict:
    """How the mule was most likely recruited."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    scores = {}
    for ch, spec in RECRUITMENT_CHANNELS.items():
        s = sum(sig.get(t, 0.0) * w for t, w in spec["tells"].items())
        if s > 0:
            scores[ch] = s
    if not scores:
        return {"channel": None, "channel_label": None, "confidence": 0.0, "scores": {}}
    top = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    return {"channel": top, "channel_label": RECRUITMENT_CHANNELS[top]["label"],
            "confidence": round(scores[top] / total, 3),
            "scores": {c: round(v, 3) for c, v in scores.items()}}


def mule_lifecycle(signals: dict) -> dict:
    """Where the account is in the mule cash-out cycle, plus a pass-through read."""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    phase_ev = {p["phase"]: 0.0 for p in MULE_LIFECYCLE}
    drivers = {p["phase"]: [] for p in MULE_LIFECYCLE}
    for tell, strength in sig.items():
        phase = LIFECYCLE_TELLS.get(tell)
        if phase is None or strength <= 0:
            continue
        phase_ev[phase] = max(phase_ev[phase], strength)
        drivers[phase].append(tell)

    reached = [p for p, ev in phase_ev.items() if ev >= 0.3]
    phase_idx = max(reached) if reached else 0
    phase = MULE_LIFECYCLE[phase_idx]

    # pass-through: money that does not rest. Derived from the layering tells.
    pass_through = round(max(sig.get("high_passthrough_ratio", 0.0),
                             sig.get("rapid_passthrough", 0.0) * 0.9), 3)

    arc_drivers = []
    for p in range(phase_idx + 1):
        arc_drivers.extend(drivers[p])

    return {
        "phase": phase_idx,
        "phase_key": phase["key"],
        "phase_label": phase["label"],
        "money_at_risk": phase_idx in _ACTIVE_PHASES,
        "pass_through_ratio": pass_through,
        "drivers": arc_drivers,
        "active": phase_idx >= 2 and max(phase_ev.values(), default=0.0) >= 0.3,
    }


def herding_signal(signals: dict) -> dict:
    """Is this account being controlled as part of a herd, and/or is it the controller?"""
    sig = {k: _clamp(v) for k, v in (signals or {}).items()}
    control = 0.0
    indicators = []
    for tell, w in CONTROL_TELLS.items():
        s = sig.get(tell, 0.0)
        if s > 0:
            control = max(control, s * w)
            indicators.append(tell)

    is_controller = (sig.get("recruits_others", 0.0) >= 0.5 or
                     sig.get("controls_multiple_accounts", 0.0) >= 0.5)
    herd_role = ("controller" if is_controller
                 else "controlled_node" if control >= 0.5
                 else "standalone")
    return {
        "controlled": control >= 0.5,
        "control_score": round(control, 3),
        "herd_role": herd_role,
        "indicators": indicators,
    }


# -- The two-track response: protect the money, treat the person by culpability --
def recommend_mule_action(role: dict, lifecycle: dict, herd: dict,
                          recruitment: dict) -> dict:
    """Two decisions, deliberately separated:
    FUND action  - freeze the pass-through so the upstream victim's money can be recovered;
    PERSON action - how to treat the account holder, graded by witting-ness."""
    r = role["role"]
    money_at_risk = lifecycle.get("money_at_risk", False)

    # FUND action: if money is moving through, hold it for the upstream victim, regardless
    # of how culpable the account holder turns out to be.
    if money_at_risk:
        fund_action = ("Hold the inbound and pending pass-through so the upstream victim's "
                       "funds can be recovered; attempt recall on the originating payment")
    else:
        fund_action = "No funds currently in flight; watch for the first inbound"

    # PERSON action: the witting-ness spectrum drives posture, SAR, and enforcement.
    if r == "herder":
        person = {
            "posture": "BLOCK + SAR + LAW-ENFORCEMENT + MAP-THE-HERD",
            "rationale": "The holder recruits and controls other mule accounts. This is an operator, not a single account; durable, network-level action is warranted.",
            "steps": ["Freeze and quarantine this account and every linked controlled account",
                      "File a SAR and a mule-network referral",
                      "Pivot to the fraud graph and map the herd from the shared control signals",
                      "Feed the herd's control signature back to the consortium"],
            "reportable": True, "punitive": True,
        }
    elif r == "witting":
        person = {
            "posture": "BLOCK + SAR",
            "rationale": "The holder is knowingly laundering funds for payment (a kept cut, continued after warning). Recidivism is high; close the account.",
            "steps": ["Block and close the account; quarantine the linked device/identity",
                      "File a SAR on the mule activity",
                      "Retain the pattern for the network graph in case a herder sits above them"],
            "reportable": True, "punitive": True,
        }
    elif r == "naive_complicit":
        person = {
            "posture": "HARD-FRICTION + FINAL-WARNING + EDUCATE",
            "rationale": "The holder suspects something is wrong and keeps a cut, but may still be redirectable. A firm, informed intervention can stop a naive mule becoming a witting one.",
            "steps": ["Freeze the pass-through and require an in-person / verified explanation of the funds",
                      "State plainly that receiving and forwarding money for others is money laundering and carries criminal liability",
                      "Offer a clean exit and a report-what-happened path",
                      "Escalate to BLOCK + SAR if the behaviour continues"],
            "reportable": lifecycle.get("phase", 0) >= 3, "punitive": False,
        }
    elif r == "unwitting":
        person = {
            "posture": "PROTECT + EDUCATE",
            "rationale": "The holder was tricked and believes this is a legitimate job; they are closer to a victim than an offender. Stop them being used without criminalising them.",
            "steps": ["Freeze the pass-through to stop the account being used as a valve",
                      "Explain that this is a money-mule scam and they are being used; no genuine job pays you to receive and forward money",
                      "If recruited via a romance or grooming funnel, route to victim safeguarding (see scam_arc)",
                      "Do NOT file a punitive report on a first-time, cooperative, tricked holder"],
            "reportable": False, "punitive": False,
        }
    else:  # undetermined
        person = {
            "posture": "HOLD + ESTABLISH-INTENT",
            "rationale": "The account shows mule-pattern movement but intent is not established. Hold and verify before deciding culpability; do not assume guilt or innocence.",
            "steps": ["Hold the suspicious inbound and pass-through",
                      "Verify the source of funds and the relationship to the sender out-of-band",
                      "Classify witting-ness from the response, then apply the matched posture"],
            "reportable": False, "punitive": False,
        }

    # A romance-recruited unwitting mule is a scam victim first.
    victim_route = (role.get("is_victim_adjacent") and
                    recruitment.get("channel") == "romance")

    return {
        "fund_action": fund_action,
        "person_action": person,
        "herd_action": ("Map and act on the whole herd" if herd.get("herd_role") == "controller"
                        else "Preserve the control signals for network mapping" if herd.get("controlled")
                        else "No herd controlling this account"),
        "route_to_victim_protection": bool(victim_route),
    }


def assess_mule(signals: dict) -> dict:
    """One call: witting-ness role + recruitment channel + account lifecycle + herding
    read + the two-track (fund vs person) response."""
    role = classify_mule(signals)
    recruitment = recruitment_channel(signals)
    lifecycle = mule_lifecycle(signals)
    herd = herding_signal(signals)
    action = recommend_mule_action(role, lifecycle, herd, recruitment)
    return {"role": role, "recruitment": recruitment, "lifecycle": lifecycle,
            "herd": herd, "action": action}
