"""
core/fingerprint.py - derive a stable device identity from client-reported components.

WHAT WAS MISSING. `core/device_graph` (in the ML repo) counts how many accounts a device_id
appears on and flags the thin-and-shared ones. That is the half of device intelligence worth
having, and it was reasoning over an id that ARRIVED IN THE TRANSACTION, assigned by nobody. This
module is the half that produces the id.

WHERE THIS CAN AND CANNOT RUN, because it is an architectural fact rather than a choice. A card
authorization reaches an issuer as an ISO 8583 message from the acquirer through the network:
there is no browser, no client, no JavaScript context, and nothing to fingerprint. That is
exactly why issuers lean on network-supplied intelligence and 3DS device data on the card rail.
Fingerprinting belongs on the surfaces where the institution owns the client - its own app and
web session - which is the push/enrolment path, and that is where core/telemetry.py already
defines the reporting contract this fills.

THE HARD PART IS NOT COLLECTION, IT IS STABILITY. Hashing every attribute together gives a
different id every time a browser updates, a font is installed, or the user travels. So the
components are split by how fast they drift:

    ANCHOR    hardware and platform: GPU renderer, CPU cores, screen geometry, audio DSP,
              platform. Changes when the physical device changes, which is the point.
    DRIFT     software surface: browser version, fonts, timezone, language. Changes often and
              legitimately, so it must not be in the primary hash.

The primary id is the ANCHOR hash. DRIFT is kept as a similarity vector so a device whose anchor
shifts (a GPU driver update genuinely does move the renderer string) can be re-linked to its
previous identity instead of appearing as a brand-new device.

ENTROPY IS REPORTED, AND THIS IS THE SAFETY PROPERTY. A privacy-hardened browser deliberately
returns generic values: canvas blocked, one timezone, a common resolution, no font list. Those
users collide onto the SAME fingerprint by design. Treating that id as an identity would put
thousands of unrelated accounts on one "device", which is precisely the many-accounts-thin-usage
shape the device gate fires on: privacy-conscious users would be classified as a fraud farm.

So a fingerprint carries a confidence derived from its own entropy, and a low-entropy fingerprint
is explicitly NOT an identity. The gate must refuse to fire on one. That refusal is the single
most important line in this file, because the failure it prevents is invisible: the graph looks
richer, the signal looks stronger, and the people being flagged are the ones protecting
themselves.

Pure stdlib, no ML stack, so it stays testable alongside the rest of core/.
"""

from __future__ import annotations

import hashlib
import math

# ── the component contract ───────────────────────────────────────────────────
#
# What a collector reports. ANCHOR components decide identity; DRIFT components only help
# re-link. The split is by observed rate of change, not by how identifying a value feels.

ANCHOR_COMPONENTS = (
    "gpu_renderer",        # WebGL UNMASKED_RENDERER: the GPU, stable per physical device
    "gpu_vendor",
    "cpu_cores",           # navigator.hardwareConcurrency
    "device_memory_gb",
    "screen_w", "screen_h", "color_depth",
    "platform",            # navigator.platform / userAgentData.platform
    "audio_dsp_hash",      # OfflineAudioContext output hash: a DSP characteristic, not software
    "touch_points",        # navigator.maxTouchPoints
)

DRIFT_COMPONENTS = (
    "browser_family", "browser_major",   # updates constantly, must not be in the anchor
    "timezone",                          # travel
    "language",
    "font_hash",                         # a font install changes this
    "canvas_hash",                       # driver and browser updates move it
    "webgl_params_hash",
)

# ── entropy ──────────────────────────────────────────────────────────────────
#
# Approximate bits of identifying information each component contributes, from published
# fingerprinting surveys (Eckersley's Panopticlick and its successors) rounded conservatively
# DOWNWARD. Being wrong low here means calling a fingerprint weak when it is strong, which
# produces a missed signal. Being wrong high means calling a weak fingerprint an identity, which
# produces the privacy-browser-as-fraud-farm failure. The asymmetry decides the rounding.
_ENTROPY_BITS = {
    "gpu_renderer": 6.0, "gpu_vendor": 2.0, "cpu_cores": 2.0, "device_memory_gb": 1.5,
    "screen_w": 2.5, "screen_h": 2.5, "color_depth": 0.8, "platform": 2.0,
    "audio_dsp_hash": 5.0, "touch_points": 1.0,
    "browser_family": 1.5, "browser_major": 2.0, "timezone": 3.0, "language": 1.5,
    "font_hash": 6.0, "canvas_hash": 8.0, "webgl_params_hash": 5.0,
}

# Values a hardened or instrumented browser returns instead of the truth. These contribute NO
# entropy: everyone blocking canvas returns the same blocked marker, so counting it as 8 bits
# would be counting a value shared by millions as if it were distinguishing.
_NULL_VALUES = {"", "blocked", "denied", "unavailable", "unsupported", "0", "null", "none",
                "undefined", "generic", "masked"}

# The set above is EXACT-MATCH, and exact-match is not enough. Measured: `blocked` scored 0 bits
# and `blocked.` scored the full 19, so one punctuation mark turned a refusal into an identity.
# Any extension or hardened browser returning "unavailable (policy)" instead of the bare word
# turned its whole user base into one high-confidence device. Matched as a PREFIX after
# normalisation, and zero-width characters are stripped first because they are invisible in a
# diff and were the cheapest way to defeat the check.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

# What a well-formed value looks like, per component. Entropy was credited by component NAME, so
# seventeen single-character values scored 52.3 bits and were certified as a high-confidence
# identity. A component only pays its bits if the value could plausibly have come from the
# collector that is supposed to produce it.
_WELL_FORMED = {
    "canvas_hash": lambda v: len(v) >= 6,
    "font_hash": lambda v: len(v) >= 6,
    "audio_dsp_hash": lambda v: len(v) >= 6,
    "webgl_params_hash": lambda v: len(v) >= 6,
    "gpu_renderer": lambda v: len(v) >= 6,
    "gpu_vendor": lambda v: len(v) >= 2,
    "platform": lambda v: len(v) >= 3,
    "timezone": lambda v: "/" in v or v in ("utc", "gmt"),
    "language": lambda v: len(v) >= 2,
    "browser_family": lambda v: len(v) >= 3,
    "browser_major": lambda v: v.isdigit(),
    "cpu_cores": lambda v: v.replace(".", "").isdigit() and 0 < float(v) <= 256,
    "device_memory_gb": lambda v: v.replace(".", "").isdigit() and 0 < float(v) <= 1024,
    "screen_w": lambda v: v.replace(".", "").isdigit() and 240 <= float(v) <= 16384,
    "screen_h": lambda v: v.replace(".", "").isdigit() and 240 <= float(v) <= 16384,
    "color_depth": lambda v: v.replace(".", "").isdigit() and 1 <= float(v) <= 64,
    "touch_points": lambda v: v.replace(".", "").isdigit() and 0 <= float(v) <= 32,
}

# An identity needs something that VARIES. Cores, colour depth and touch points are
# near-universal; agreeing on them is agreeing on nothing. At least one of these must be present
# and well-formed before a fingerprint is an identity, and before a re-link is believed.
HIGH_ENTROPY_ANCHORS = ("gpu_renderer", "audio_dsp_hash")

# Below this, a fingerprint identifies a CROWD rather than a device. Chosen so a browser
# returning only coarse, widely-shared values cannot become an identity: roughly a population of
# 2^18, which is a large crowd and a useless identity.
MIN_ENTROPY_BITS_FOR_IDENTITY = 18.0

# How much of the anchor must survive for a re-link to be believable.
RELINK_SIMILARITY = 0.72

# ── automation and integrity ─────────────────────────────────────────────────
#
# These are not identity, they are the other reason to collect at all: a fingerprint that says
# "headless Chrome under automation on a datacentre IP" is worth more than any id.
_AUTOMATION_FLAGS = ("webdriver", "headless", "cdp_detected", "phantom_markers",
                     "selenium_markers", "playwright_markers", "notification_permission_denied_headless")
_INTEGRITY_FLAGS = ("emulator", "rooted_jailbroken", "hooking_frida", "debugger_attached",
                    "repackaged_app", "remote_access_tool_active")


# INPUT BOUNDS. Every value here is attacker-controlled: the payload arrives from a client the
# institution does not control, on a path meant to fire once per session. Unbounded, a 50,000-key
# payload with 1KB values was accepted and derived without complaint, and the endpoint then
# persisted it. These caps are generous against the real contract (17 declared components) and
# hostile to anything else.
MAX_COMPONENTS = 64
MAX_VALUE_CHARS = 512


def bounded(components: dict) -> dict:
    """Clamp an untrusted component payload before anything reads it.

    Truncates rather than rejects: a client sending one over-long GPU string is far more likely
    to be a real browser on unusual hardware than an attack, and refusing the whole fingerprint
    would lose a legitimate device. A payload with more keys than the contract declares is a
    different matter, and the extras are dropped entirely - they cannot be components, because
    the component list is closed.
    """
    if not isinstance(components, dict):
        return {}
    known = set(ANCHOR_COMPONENTS) | set(DRIFT_COMPONENTS) | set(_AUTOMATION_FLAGS) | \
        set(_INTEGRITY_FLAGS)
    out = {}
    for k, v in components.items():
        if len(out) >= MAX_COMPONENTS:
            break
        if k not in known:
            continue          # closed contract: an undeclared key is not a component
        out[k] = v[:MAX_VALUE_CHARS] if isinstance(v, str) else v
    return out


def _norm(v) -> str:
    # str() on a dict or list produces a long repr that would otherwise become a component
    # value. Only scalars can be components, so anything else normalises to absent.
    if isinstance(v, (dict, list, tuple, set, bytes)):
        return ""
    if v is None:
        return ""
    s = str(v).translate(_ZERO_WIDTH).strip().lower()[:MAX_VALUE_CHARS]
    # PREFIX, not equality. `blocked.` and `unavailable (policy)` are refusals wearing a
    # different coat, and exact-match credited them full entropy.
    for marker in _NULL_VALUES:
        if marker and s.startswith(marker):
            return ""
    if s in _NULL_VALUES:
        return ""
    # A value carrying the hash delimiters is either a broken collector or an attempt to
    # impersonate other components. See _hash: neither is a value worth keeping.
    if "|" in s or "=" in s:
        return ""
    return s


def _hash(parts) -> str:
    """LENGTH-PREFIXED, not delimiter-joined.

    The first version joined `name=value` strings on `|` with no escaping, so a value containing
    `|name=value` was indistinguishable from a separate component. Measured: an attacker
    reporting ONLY audio_dsp_hash reproduced a victim's device_id byte-identically while
    reporting neither cpu_cores nor gpu_renderer. Prefixing each field with its own length makes
    the encoding injective, so no arrangement of one value can spell another set.
    """
    return hashlib.sha256(
        "".join(f"{len(p)}:{p}" for p in parts).encode()).hexdigest()


def entropy_bits(components: dict) -> float:
    """Approximate identifying bits this fingerprint actually carries.

    Only components that returned a REAL value count. A blocked canvas contributes zero, not
    eight, because every blocked canvas looks identical.
    """
    components = components or {}
    total = 0.0
    for name, bits in _ENTROPY_BITS.items():
        v = _norm(components.get(name))
        if not v:
            continue
        # A present key is not evidence. Seventeen single-character values once scored 52.3 bits
        # and were certified high-confidence, because entropy was credited by component NAME.
        check = _WELL_FORMED.get(name)
        if check is not None:
            try:
                if not check(v):
                    continue
            except Exception:                                     # noqa: BLE001
                continue
        total += bits
    return round(total, 2)


def anchor_entropy_bits(components: dict) -> float:
    """Bits carried by the components that ACTUALLY FORM THE ID.

    The floor used to be checked against anchor plus drift while `device_id` hashed anchor only,
    and drift alone is 27.0 bits against a floor of 18.0. So canvas, fonts, timezone and language
    bought an identity for an id they never enter, and three near-universal anchors were enough.
    Measured: two different people with different drift derived the SAME id and both came back
    is_identity=True, which is the shape the device gate escalates on.
    """
    components = components or {}
    return round(sum(b for c, b in _ENTROPY_BITS.items()
                     if c in ANCHOR_COMPONENTS
                     and _norm(components.get(c))
                     and (_WELL_FORMED.get(c) is None
                          or _safe_check(c, _norm(components.get(c))))), 2)


def _safe_check(name, value) -> bool:
    try:
        return bool(_WELL_FORMED[name](value))
    except Exception:                                             # noqa: BLE001
        return False


def automation_score(components: dict) -> dict:
    """Is this a browser being driven by software? Separate from identity on purpose.

    An automation signal is actionable even when the fingerprint is too weak to identify anyone,
    which is the common case for a scripted client: bots frequently run hardened or minimal
    browsers that carry almost no entropy.
    """
    components = components or {}
    fired = [f for f in _AUTOMATION_FLAGS if components.get(f) in (True, 1, "true", "1", "yes")]
    integrity = [f for f in _INTEGRITY_FLAGS if components.get(f) in (True, 1, "true", "1", "yes")]
    return {
        "automation_detected": bool(fired),
        "automation_flags": fired,
        "integrity_flags": integrity,
        # Not a probability. A count of independent tells, which is what an analyst can check.
        "tells": len(fired) + len(integrity),
    }


def derive(components: dict) -> dict:
    """Client-reported components in, a device identity out.

    Returns the id, the confidence in it, the entropy behind that confidence, the drift vector
    for later re-linking, and the automation view. Never raises: a malformed or hostile payload
    yields a low-confidence result rather than an exception on the scoring path.

    The payload is BOUNDED first. Every value in it is attacker-controlled and this runs on a
    request path, so an unbounded dict is a resource-exhaustion surface before it is anything
    else. See bounded().
    """
    components = bounded(components)

    anchor_present = [(c, _norm(components.get(c))) for c in ANCHOR_COMPONENTS]
    anchor_present = [(c, v) for c, v in anchor_present if v]
    drift_present = [(c, _norm(components.get(c))) for c in DRIFT_COMPONENTS]
    drift_present = [(c, v) for c, v in drift_present if v]

    bits = entropy_bits(components)
    # THE FLOOR IS CHECKED AGAINST ANCHOR ENTROPY, not total. `device_id` hashes anchors only,
    # so crediting drift toward the floor let canvas, fonts, timezone and language buy an
    # identity for an id they never enter. And at least one HIGH-entropy anchor must be present:
    # cores, colour depth and touch points are near-universal, so three of them agreeing is
    # three people agreeing they own a computer.
    a_bits = anchor_entropy_bits(components)
    has_strong = any(_norm(components.get(c)) and _safe_check(c, _norm(components.get(c)))
                     for c in HIGH_ENTROPY_ANCHORS)
    identifiable = (a_bits >= MIN_ENTROPY_BITS_FOR_IDENTITY
                    and len(anchor_present) >= 3 and has_strong)

    device_id = ""
    if anchor_present:
        device_id = "fp_" + _hash([f"{c}={v}" for c, v in sorted(anchor_present)])[:16]

    return {
        "device_id": device_id,
        # THE SAFETY PROPERTY. A fingerprint below the entropy floor names a crowd, not a device,
        # and anything that links accounts by device must refuse to use it. Privacy-hardened
        # browsers all collide here, and treating that collision as a shared device would make
        # the device graph read them as a fraud farm.
        "is_identity": bool(identifiable and device_id),
        "confidence": "high" if (identifiable and a_bits >= 20) else
                      "medium" if identifiable else "low",
        "entropy_bits": bits,
        # Reported separately so the split that caused the defect is visible to anyone reading
        # a response, rather than being a fact you have to know about the implementation.
        "anchor_entropy_bits": a_bits,
        "has_high_entropy_anchor": has_strong,
        "anchor_components_present": len(anchor_present),
        "drift_vector": {c: v for c, v in drift_present},
        "why_not_identity": (
            "" if identifiable else
            (f"only {a_bits} bits of ANCHOR entropy across {len(anchor_present)} anchor "
             f"components ({bits} total including drift, which does not enter the id); this "
             "fingerprint describes a crowd, not a device")
            if not has_strong or a_bits < MIN_ENTROPY_BITS_FOR_IDENTITY else
            "no high-entropy anchor (GPU or audio DSP) was present and well-formed"),
        **automation_score(components),
    }


def relink(candidate: dict, known: dict) -> dict:
    """Is this the same physical device as one already seen, after some drift?

    A GPU driver update moves the renderer string and a new monitor moves the geometry, so a
    device whose anchor shifts would otherwise appear brand new and its history would be lost.
    Compares the anchors component by component rather than by hash, because a hash gives no
    partial credit and partial credit is the entire question here.

    Deliberately NOT symmetric with identity: re-linking requires a high bar, since a wrong
    re-link merges two people's histories into one device, which is worse than losing one
    device's history.
    """
    ca = {c: _norm((candidate or {}).get(c)) for c in ANCHOR_COMPONENTS}
    ka = {c: _norm((known or {}).get(c)) for c in ANCHOR_COMPONENTS}

    # THE DENOMINATOR IS THE KNOWN DEVICE, not the intersection. Scoring over the intersection
    # let the candidate drive the denominator to 3 by nulling every anchor it could not guess,
    # so the 0.72 threshold never bound: platform, colour depth and screen width, three values
    # anyone can guess, scored 1.0 and re-linked. Absence now counts as disagreement, which is
    # what makes withholding expensive instead of free.
    denominator = [c for c in ANCHOR_COMPONENTS if ka.get(c)]
    if len(denominator) < 3:
        return {"same_device": False, "similarity": 0.0,
                "why": "the known device has too few anchors to judge against"}

    agree = [c for c in denominator if ca.get(c) and ca[c] == ka[c]]
    differed = [c for c in denominator if ca.get(c) and ca[c] != ka[c]]
    withheld = [c for c in denominator if not ca.get(c)]
    sim = len(agree) / len(denominator)

    # A wrong re-link merges two people's histories onto one device, which is worse than losing
    # one device's history. So something that VARIES has to match: agreeing on core count and
    # colour depth is agreeing on nothing.
    strong_agreed = [c for c in HIGH_ENTROPY_ANCHORS if c in agree]
    ok = sim >= RELINK_SIMILARITY and bool(strong_agreed)

    return {
        "same_device": ok,
        "similarity": round(sim, 3),
        "agreed": agree,
        "differed": differed,
        "withheld": withheld,
        "high_entropy_agreed": strong_agreed,
        "why": (f"{len(agree)}/{len(denominator)} of the known device's anchors match"
                + (f", {len(withheld)} withheld by the candidate" if withheld else "")
                + ("" if strong_agreed else
                   "; no high-entropy anchor (GPU or audio DSP) agreed, so this is not a merge "
                   "worth making")),
    }


def to_telemetry(fp: dict) -> dict:
    """Map a derived fingerprint onto the fields core/telemetry.py already consumes.

    The telemetry contract was written before anything produced these values; this is the
    producer. Only reported values are emitted, so an absent signal stays absent rather than
    becoming a confident False - the same discipline derive_signals() applies downstream.
    """
    fp = fp or {}
    out = {}
    if fp.get("automation_detected"):
        out["automation_framework"] = True
        if "headless" in (fp.get("automation_flags") or []):
            out["headless"] = True
    for flag in (fp.get("integrity_flags") or []):
        out[flag] = True
    return out
