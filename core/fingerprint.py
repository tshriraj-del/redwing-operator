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


def _norm(v) -> str:
    s = str(v).strip().lower() if v is not None else ""
    return "" if s in _NULL_VALUES else s


def _hash(parts) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def entropy_bits(components: dict) -> float:
    """Approximate identifying bits this fingerprint actually carries.

    Only components that returned a REAL value count. A blocked canvas contributes zero, not
    eight, because every blocked canvas looks identical.
    """
    components = components or {}
    total = 0.0
    for name, bits in _ENTROPY_BITS.items():
        if _norm(components.get(name)):
            total += bits
    return round(total, 2)


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
    """
    components = components or {}

    anchor_present = [(c, _norm(components.get(c))) for c in ANCHOR_COMPONENTS]
    anchor_present = [(c, v) for c, v in anchor_present if v]
    drift_present = [(c, _norm(components.get(c))) for c in DRIFT_COMPONENTS]
    drift_present = [(c, v) for c, v in drift_present if v]

    bits = entropy_bits(components)
    identifiable = bits >= MIN_ENTROPY_BITS_FOR_IDENTITY and len(anchor_present) >= 3

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
        "confidence": "high" if bits >= 28 else "medium" if identifiable else "low",
        "entropy_bits": bits,
        "anchor_components_present": len(anchor_present),
        "drift_vector": {c: v for c, v in drift_present},
        "why_not_identity": ("" if identifiable else
                             f"only {bits} bits of entropy across "
                             f"{len(anchor_present)} anchor components; this fingerprint "
                             "describes a crowd, not a device"),
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
    comparable = [c for c in ANCHOR_COMPONENTS if ca.get(c) and ka.get(c)]
    if len(comparable) < 3:
        return {"same_device": False, "similarity": 0.0,
                "why": "too few comparable anchor components to judge"}
    agree = [c for c in comparable if ca[c] == ka[c]]
    sim = len(agree) / len(comparable)
    return {
        "same_device": sim >= RELINK_SIMILARITY,
        "similarity": round(sim, 3),
        "agreed": agree,
        "differed": [c for c in comparable if ca[c] != ka[c]],
        "why": (f"{len(agree)}/{len(comparable)} anchor components match"),
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
