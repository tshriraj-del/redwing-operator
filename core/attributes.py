"""
core/attributes.py - bureau-scale device + identity attribute fabric (concept).

Two of the model's feature families (device, identity) were scaffolds. This turns them
into structured families with a derivation grammar, the way a bureau reaches thousands
of attributes: ~150 base attributes x derivations (value/match/age/reputation/first-seen)
x time windows x entity pivots. Values are DERIVED deterministically and kept COHERENT
with the case's typology (a synthetic-ID case yields a thin file + fresh email + a
shared-SSN cluster; an ATO yields device-integrity anomalies + impossible travel + a
recent SIM swap; a scam VICTIM looks clean here, because the tell is behavioural, not
in the device or the identity). Same discipline as case_file.py: real to reason over,
without inventing precision.

evaluate(entity_id, typology, flags) -> {device, identity, surface, top_signals, ...}
"""

from __future__ import annotations

import hashlib
import random

# ── Schema: families -> base attributes, and how many derived leaves each expands to ──
# surface = sum(len(attrs) * derivations) x entity pivots. This is how ~150 base
# attributes become thousands of evaluable leaves.

DEVICE_FAMILIES = {
    "Hardware & OS":        {"derivations": 6, "attrs": ["device_model", "os_version", "cpu_cores", "screen_res", "timezone", "locale", "battery_api"]},
    "Browser/App":          {"derivations": 6, "attrs": ["canvas_hash", "webgl_hash", "audio_hash", "font_set", "ja3_tls", "ua_consistency", "app_build_integrity"]},
    "Integrity & tamper":   {"derivations": 5, "attrs": ["emulator", "rooted_jailbroken", "headless_browser", "automation_framework", "hooking_frida", "debugger_attached", "repackaged_app"]},
    "Network & connection": {"derivations": 8, "attrs": ["ip_type", "asn", "proxy_vpn_tor", "ip_reputation", "ip_geo", "ip_to_billing_km", "hosting_provider", "webrtc_leak"]},
    "Behavioral biometrics":{"derivations": 7, "attrs": ["keystroke_cadence", "mouse_entropy", "touch_pressure", "swipe_velocity", "scroll_rhythm", "motion_sensor", "paste_vs_typed"]},
    "Session dynamics":     {"derivations": 6, "attrs": ["session_duration", "action_cadence", "bot_regularity", "api_vs_ui", "nav_path_entropy", "time_of_day"]},
    "Device reputation":    {"derivations": 8, "attrs": ["device_age_days", "device_fraud_history", "accounts_per_device", "devices_per_account", "new_account_velocity"]},
    "Device graph":         {"derivations": 6, "attrs": ["shared_device_cluster", "consortium_sightings", "linked_device_count"]},
}

IDENTITY_FAMILIES = {
    "Core PII validity":    {"derivations": 6, "attrs": ["name_valid", "dob_valid", "ssn_valid", "ssn_issue_age", "address_valid", "phone_valid", "email_valid"]},
    "Email intelligence":   {"derivations": 7, "attrs": ["email_age_days", "domain_age_days", "disposable", "breach_appearances", "deliverable", "name_email_match", "linked_accounts"]},
    "Phone intelligence":   {"derivations": 7, "attrs": ["phone_age_days", "line_type", "carrier", "porting_recency_days", "sim_swap_risk", "name_phone_match", "geo_consistency"]},
    "Address intelligence": {"derivations": 6, "attrs": ["address_type", "address_age_days", "occupancy", "deliverable", "address_velocity"]},
    "Identity linkage":     {"derivations": 8, "attrs": ["ids_sharing_ssn", "ids_sharing_phone", "ids_sharing_email", "cluster_size", "thin_file", "credit_file_exists", "record_tenure_days", "synthetic_id_score"]},
    "Doc & biometric":      {"derivations": 5, "attrs": ["id_doc_authentic", "liveness", "face_match", "selfie_quality"]},
    "Bureau / credit":      {"derivations": 6, "attrs": ["tradeline_age_days", "inquiry_velocity", "address_changes_90d", "utilization"]},
    "Negative / consortium":{"derivations": 6, "attrs": ["prior_fraud_reports", "ic3_hits", "watchlist_hit", "known_mule_flag", "victim_history"]},
    "Onboarding behavior":  {"derivations": 7, "attrs": ["application_velocity", "cross_app_reuse", "kyc_entry_hesitation", "kyc_paste_rate", "form_abandon_return"]},
}

# Entity pivots each attribute can be computed against (user / device / household / network).
_PIVOTS = 3


def _surface(families: dict) -> int:
    return sum(len(f["attrs"]) * f["derivations"] for f in families.values()) * _PIVOTS


DEVICE_SURFACE   = _surface(DEVICE_FAMILIES)
IDENTITY_SURFACE = _surface(IDENTITY_FAMILIES)
TOTAL_SURFACE    = DEVICE_SURFACE + IDENTITY_SURFACE


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


# ── Coherence: how a typology tilts the two surfaces ──────────────────────────
# Each entry raises the risk of specific attribute groups so the derived fabric agrees
# with ground truth. A scam VICTIM (pig_butchering / app_scam) is deliberately NOT here:
# their device and identity look clean, and the tell lives in the motive layer.

_TILT = {
    "card_testing_bot":     {"integrity": 0.95, "network_dc": 0.9,  "bot": 0.95, "device_rep": 0.6},
    "ai_powered_ato":       {"device_new": 0.9, "impossible_travel": 0.9, "sim_swap": 0.8, "biometric_dev": 0.85},
    "account_takeover_ai":  {"device_new": 0.9, "impossible_travel": 0.9, "sim_swap": 0.8, "biometric_dev": 0.85},
    "synthetic_id_ai":      {"thin_file": 0.95, "email_fresh": 0.9, "voip": 0.7, "shared_ssn": 0.85, "doc_weak": 0.8},
    "synthetic_identity":   {"thin_file": 0.95, "email_fresh": 0.9, "voip": 0.7, "shared_ssn": 0.85, "doc_weak": 0.8},
    "mule_cashout":         {"shared_device": 0.9, "network_dc": 0.7, "thin_file": 0.5, "consortium": 0.8},
}


def _score(rng, base_low, base_high, boost=0.0):
    v = rng.uniform(base_low, base_high) + boost
    return round(min(1.0, max(0.0, v)), 3)


def evaluate(entity_id: str, typology: str = "", flags=None) -> dict:
    """Derive the device + identity attribute fabric for one entity, coherent with the
    typology. Returns per-family risk, the total attribute surface evaluated, and the
    top contributing signals in plain language."""
    rng = _rng("attr", entity_id, typology)
    typ = str(typology or "").lower()
    flags = set(flags or [])
    t = _TILT.get(typ, {})

    device, identity, top = {}, {}, []

    def add_top(name, risk, note):
        if risk >= 0.6:
            top.append({"attribute": name, "risk": risk, "note": note})

    # ── Device families (representative decision-relevant leaves per family) ──
    integ = _score(rng, 0.02, 0.12, t.get("integrity", 0.0))
    device["Integrity & tamper"] = {
        "headless_browser": integ >= 0.5,
        "automation_framework": ("puppeteer" if t.get("bot") and rng.random() > 0.4 else None),
        "emulator": _score(rng, 0.0, 0.1, t.get("integrity", 0.0)) >= 0.5,
        "risk": integ,
    }
    add_top("device.headless_browser", integ, "Device runs a headless/automation environment")

    netdc = _score(rng, 0.05, 0.2, t.get("network_dc", 0.0))
    device["Network & connection"] = {
        "ip_type": ("datacenter" if netdc >= 0.5 else rng.choice(["residential", "mobile"])),
        "proxy_vpn_tor": netdc >= 0.55,
        "ip_to_billing_km": (rng.randint(3000, 9000) if t.get("impossible_travel") else rng.randint(0, 60)),
        "risk": netdc,
    }
    add_top("device.ip_type=datacenter", netdc, "Connection originates from a datacenter / proxy")
    if t.get("impossible_travel"):
        add_top("device.impossible_travel", 0.85, "IP-to-billing distance implies impossible travel")

    devnew = _score(rng, 0.03, 0.15, t.get("device_new", 0.0))
    device["Device reputation"] = {
        "device_age_days": (rng.randint(0, 2) if devnew >= 0.5 else rng.randint(30, 1200)),
        "accounts_per_device": (rng.randint(6, 20) if t.get("shared_device") else rng.randint(1, 2)),
        "device_fraud_history": round(t.get("consortium", 0.0) * rng.uniform(0.3, 0.6), 3),
        "risk": max(devnew, t.get("shared_device", 0.0) * 0.9),
    }
    if t.get("shared_device"):
        add_top("device.accounts_per_device", 0.85, "One device fingerprint is shared across many accounts")
    bio = _score(rng, 0.05, 0.2, t.get("biometric_dev", 0.0))
    device["Behavioral biometrics"] = {"deviation_from_baseline": bio,
                                        "paste_vs_typed": (bio >= 0.5), "risk": bio}
    if bio >= 0.6:
        add_top("device.biometric_deviation", bio, "Interaction rhythm deviates from the account's baseline")

    # ── Identity families ──
    thin = _score(rng, 0.05, 0.2, t.get("thin_file", 0.0))
    synth = _score(rng, 0.02, 0.12, t.get("shared_ssn", 0.0) + t.get("thin_file", 0.0) * 0.3)
    identity["Identity linkage"] = {
        "thin_file": thin >= 0.5,
        "credit_file_exists": not (thin >= 0.5),
        "ids_sharing_ssn": (rng.randint(3, 9) if t.get("shared_ssn") else rng.randint(0, 1)),
        "record_tenure_days": (rng.randint(5, 90) if thin >= 0.5 else rng.randint(400, 4000)),
        "synthetic_id_score": synth,
        "risk": max(thin, synth),
    }
    add_top("identity.synthetic_id_score", synth, "Identity shows synthetic-ID construction markers")
    if t.get("shared_ssn"):
        add_top("identity.ids_sharing_ssn", 0.8, "SSN is shared across multiple distinct identities")

    emailf = _score(rng, 0.05, 0.2, t.get("email_fresh", 0.0))
    identity["Email intelligence"] = {
        "email_age_days": (rng.randint(0, 6) if emailf >= 0.5 else rng.randint(120, 3000)),
        "disposable": emailf >= 0.6, "breach_appearances": rng.randint(0, 3),
        "name_email_match": round(1 - emailf, 3), "risk": emailf,
    }
    if emailf >= 0.6:
        add_top("identity.email_age", emailf, "Email address created within days of the application")

    voip = _score(rng, 0.05, 0.2, t.get("voip", 0.0))
    identity["Phone intelligence"] = {
        "line_type": ("VOIP" if voip >= 0.5 else rng.choice(["mobile", "mobile", "landline"])),
        "porting_recency_days": (rng.randint(0, 5) if t.get("sim_swap") else rng.randint(120, 2000)),
        "sim_swap_risk": _score(rng, 0.02, 0.1, t.get("sim_swap", 0.0)),
        "risk": max(voip, t.get("sim_swap", 0.0) * 0.9),
    }
    if voip >= 0.5:
        add_top("identity.line_type=VOIP", voip, "Phone is a VOIP line, common in synthetic identities")
    if t.get("sim_swap"):
        add_top("identity.sim_swap", 0.8, "Recent SIM port precedes the takeover")

    docw = _score(rng, 0.02, 0.12, t.get("doc_weak", 0.0))
    identity["Doc & biometric"] = {"id_doc_authentic": round(1 - docw, 3),
                                   "liveness": round(1 - docw, 3), "face_match": round(1 - docw, 3), "risk": docw}
    if docw >= 0.6:
        add_top("identity.doc_liveness", docw, "Document authenticity / liveness checks are weak")

    # context flags nudge onboarding risk
    onb = 0.75 if "prior fraud flags on account" in {f.lower() for f in flags} else _score(rng, 0.05, 0.25)
    identity["Onboarding behavior"] = {"application_velocity": round(onb, 3),
                                       "kyc_entry_hesitation": _score(rng, 0.05, 0.3), "risk": onb}

    top.sort(key=lambda x: x["risk"], reverse=True)
    device_risk   = round(max([v.get("risk", 0.0) for v in device.values()] or [0.0]), 3)
    identity_risk = round(max([v.get("risk", 0.0) for v in identity.values()] or [0.0]), 3)

    return {
        "device": device,
        "identity": identity,
        "surface": {"device": DEVICE_SURFACE, "identity": IDENTITY_SURFACE, "total": TOTAL_SURFACE},
        "device_families": list(DEVICE_FAMILIES),
        "identity_families": list(IDENTITY_FAMILIES),
        "device_risk": device_risk,
        "identity_risk": identity_risk,
        "top_signals": top[:8],
    }
