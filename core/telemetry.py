"""
core/telemetry.py - real behavioural telemetry, and the honest bridge to the actor layer.

The actor modules (motive, scam_arc, onboarding) run on behavioural TELLS: was someone coached
mid-payment, is the session scripted, did they fumble their own PII. A raw transaction row does
not carry those. The tempting shortcut is to derive tells from the case typology, but the
typology is the ANSWER, so that is leakage: the model would be reading its own label. The
honest fix is real telemetry: signals a client SDK actually reports about the device, the
session, the biometrics, and the interaction, captured independently of the outcome.

This module does two things:
  1. Documents the telemetry schema a real client emits (TELEMETRY_FIELDS).
  2. derive_signals(raw) maps the REPORTED values to the actor modules' tells, and nothing
     else. If a field was not reported, no tell fires. With no telemetry at all, the actor
     layer stays silent, which is the correct, honest behaviour: you cannot infer motive from
     an amount and a rail.

The values in any demo are still example telemetry, but the derivation never touches the
typology, so when a real SDK feeds real behaviour this works unchanged and without leakage.
"""

from __future__ import annotations

# What a real client SDK reports. Grouped for documentation; all optional.
TELEMETRY_FIELDS = {
    "device_integrity": ["emulator", "rooted_jailbroken", "headless", "automation_framework",
                          "app_integrity_ok", "remote_access_tool_active"],
    "network":          ["ip_type", "proxy_or_vpn"],
    "session_dynamics": ["ttfa_seconds", "action_cadence_regularity", "dwell_median_ms",
                         "nav_path_directness", "session_duration_s"],
    "biometrics":       ["keystroke_cadence_cv", "paste_rate", "mouse_entropy", "typing_hesitation"],
    "interaction":      ["call_active", "app_switching_count", "long_reads_before_field",
                         "dictation_shaped_input", "duress_detected"],
    "stated_context":   ["new_payee", "payee_type", "relationship_online_only",
                         "never_met_in_person", "safe_account_narrative", "fee_to_release"],
}


def _f(raw: dict, key: str, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(raw.get(key, default))))
    except (TypeError, ValueError):
        return default


def _b(raw: dict, key: str) -> float:
    return 1.0 if raw.get(key) in (True, 1, "true", "True", "yes") else 0.0


def derive_signals(raw: dict) -> dict:
    """Map reported telemetry to actor-module tells. Only reported values produce tells; the
    output feeds motive.assess_actor and scam_arc.assess_scam (each ignores tells it does not
    model). Deriving strictly from reported behaviour, never from the typology, keeps it honest."""
    raw = raw or {}
    sig: dict = {}

    def put(tell: str, strength: float):
        if strength and strength > 0:
            sig[tell] = round(max(sig.get(tell, 0.0), min(1.0, strength)), 3)

    # -- coercion / live-coaching (the strongest, most humane signals) --
    call = _b(raw, "call_active")
    put("coaching_copresence", call * 0.85)
    put("coaching_pauses", call * 0.7)
    reads = max(_f(raw, "long_reads_before_field"), _b(raw, "dictation_shaped_input") * 0.8)
    put("script_reading", reads)
    put("coached_answers", _b(raw, "dictation_shaped_input") * 0.7)
    put("app_switching", min(1.0, float(raw.get("app_switching_count", 0) or 0) / 3.0))
    put("remote_access_active", _b(raw, "remote_access_tool_active") * 0.9)
    put("duress", _f(raw, "duress_detected") or _b(raw, "duress_detected") * 0.9)

    # -- automation / professional execution (device + session) --
    auto = max(_b(raw, "automation_framework"), _b(raw, "headless"))
    put("automation_scalable", auto * 0.8)
    put("scripted_timing", _f(raw, "action_cadence_regularity"))
    tooling = max(_b(raw, "emulator"), _b(raw, "rooted_jailbroken"),
                  _b(raw, "proxy_or_vpn"), 1.0 if raw.get("ip_type") in ("tor", "vpn", "hosting") else 0.0)
    put("sophisticated_tooling", tooling * 0.7)
    # a speedrun: instant first action, beeline navigation, metronomic cadence
    ttfa = raw.get("ttfa_seconds")
    fast = 1.0 if (isinstance(ttfa, (int, float)) and ttfa <= 2.0) else 0.0
    speedrun = min(1.0, (fast + _f(raw, "nav_path_directness") + _f(raw, "action_cadence_regularity")) / 2.0)
    put("professional_execution", speedrun if speedrun >= 0.5 else 0.0)
    put("too_fast_entry", fast * 0.7)

    # -- biometrics: pasting and hesitation on one's own identity --
    put("pii_pasted", _f(raw, "paste_rate"))
    hes = max(_f(raw, "typing_hesitation"), _f(raw, "keystroke_cadence_cv"))
    put("pii_hesitation", hes)
    put("hesitation_entropy", hes * 0.8)
    put("reverse_familiarity", _f(raw, "typing_hesitation") * 0.6)

    # -- stated purpose / relationship (from a purpose-of-payment step) --
    if _b(raw, "new_payee") and str(raw.get("payee_type", "")).lower() == "crypto":
        put("first_payee_new_crypto", 0.8)
    put("online_only_relationship", _b(raw, "relationship_online_only") * 0.8)
    put("never_met_in_person", _b(raw, "never_met_in_person") * 0.7)
    safe = _b(raw, "safe_account_narrative")
    put("safe_account_move", safe * 0.9)
    put("authority_urgency", safe * 0.7)
    put("fee_to_release", _b(raw, "fee_to_release") * 0.8)

    return sig


def assess_from_telemetry(raw: dict) -> dict:
    """Derive signals from telemetry and run the actor read (offender view + victim view)."""
    from .motive import assess_actor
    from .scam_arc import assess_scam
    signals = derive_signals(raw or {})
    return {
        "telemetry_present": bool(raw),
        "signals": signals,
        "actor": assess_actor(signals) if signals else None,
        "victim": assess_scam(signals) if signals else None,
    }


def assess_subject(store, subject_ref: str) -> dict:
    """Pull a subject's stored telemetry and assess it. Empty telemetry -> a silent read, which
    is the honest answer when there is no behaviour to reason over."""
    raw = store.get_telemetry(subject_ref) if store is not None else {}
    return assess_from_telemetry(raw)
