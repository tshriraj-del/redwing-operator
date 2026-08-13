"""
Tests for device fingerprint derivation.

Why this file exists. The device graph counts how many accounts a device_id appears on and flags
the thin-and-shared ones. It was reasoning over an id that arrived in the transaction, assigned by
nobody. This is the producer, and it has two properties worth holding down hard:

  STABILITY. Identity comes from ANCHOR components (GPU, cores, screen, audio DSP, platform) and
  never from DRIFT ones (browser version, fonts, timezone, canvas). Hashing them together is the
  standard mistake and it yields a new device every time a browser updates, which quietly
  destroys the graph the id feeds.

  THE ENTROPY FLOOR, which is the safety property. A privacy-hardened browser returns generic or
  blocked values, and every such browser returns the SAME ones. Treating that as an identity puts
  thousands of unrelated accounts onto one "device", which is exactly the many-accounts-used-
  thinly shape the device gate fires on. Without the floor, the people hardest to fingerprint get
  classified as a fraud farm. That failure is invisible from the metrics: the graph looks richer
  and the signal looks stronger.

Runs under pytest or standalone (python3 tests/test_fingerprint.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import fingerprint as FP  # noqa: E402


def _rich(**over):
    """A normal, non-hardened browser: enough entropy to be an identity."""
    c = {
        "gpu_renderer": "ANGLE (Apple, Apple M2 Pro, OpenGL 4.1)",
        "gpu_vendor": "Apple", "cpu_cores": 12, "device_memory_gb": 16,
        "screen_w": 3456, "screen_h": 2234, "color_depth": 30,
        "platform": "MacIntel", "audio_dsp_hash": "a1b2c3d4", "touch_points": 0,
        "browser_family": "Chrome", "browser_major": "141",
        "timezone": "Europe/London", "language": "en-GB",
        "font_hash": "ff00aa11", "canvas_hash": "9c8d7e6f",
        "webgl_params_hash": "1234abcd",
    }
    c.update(over)
    return c


# --------------------------------------------------------------------- stability

def test_a_browser_update_does_not_change_the_device():
    """THE stability property. Browser version, canvas and fonts are DRIFT: they move constantly
    and legitimately. If they reached the identity hash, every Chrome update would present the
    whole book as new devices and the device graph would reset itself every few weeks."""
    before = FP.derive(_rich())
    after = FP.derive(_rich(browser_major="142", canvas_hash="totally_different",
                            font_hash="also_different"))
    assert before["device_id"] == after["device_id"], (
        "a browser update produced a new device id; DRIFT components have leaked into the anchor")


def test_travel_does_not_change_the_device():
    """Timezone and language are DRIFT for the obvious reason."""
    assert (FP.derive(_rich())["device_id"]
            == FP.derive(_rich(timezone="America/New_York", language="en-US"))["device_id"])


def test_different_hardware_is_a_different_device():
    """The other half: the anchor must actually discriminate."""
    a = FP.derive(_rich())
    b = FP.derive(_rich(gpu_renderer="ANGLE (NVIDIA GeForce RTX 4070)", cpu_cores=24,
                        screen_w=2560, screen_h=1440, audio_dsp_hash="ffffffff"))
    assert a["device_id"] != b["device_id"]


def test_component_order_does_not_change_the_id():
    """A dict is unordered in principle and a hash over insertion order is a bug that only shows
    up on some clients."""
    base = _rich()
    shuffled = {k: base[k] for k in reversed(list(base))}
    assert FP.derive(base)["device_id"] == FP.derive(shuffled)["device_id"]


# --------------------------------------------------------------------- the entropy floor

def test_a_hardened_browser_is_not_treated_as_an_identity():
    """THE safety property, and the one worth the most. Every privacy-hardened browser returns
    the same blocked and generic values, so they all derive the SAME id. Accepting it as an
    identity would stack thousands of unrelated accounts onto one device, and the device gate
    fires on exactly that shape: many accounts, each used thinly. Privacy-conscious users would
    be classified as fraud infrastructure."""
    hardened = {
        "gpu_renderer": "blocked", "gpu_vendor": "blocked",
        "cpu_cores": 4, "device_memory_gb": "blocked",
        "screen_w": 1920, "screen_h": 1080, "color_depth": 24,
        "platform": "Win32", "audio_dsp_hash": "blocked", "touch_points": 0,
        "canvas_hash": "blocked", "font_hash": "blocked",
        "webgl_params_hash": "blocked", "timezone": "UTC", "language": "en-US",
    }
    fp = FP.derive(hardened)
    assert fp["is_identity"] is False, (
        f"a hardened browser with {fp['entropy_bits']} bits was accepted as an identity; it "
        "names a crowd, and the device graph would read that crowd as a shared fraud device")
    assert fp["confidence"] == "low"
    assert fp["why_not_identity"], "refusing an identity must say why"


def test_blocked_values_contribute_no_entropy():
    """MUTATION GUARD. If a blocked canvas counted its full 8 bits, every hardened browser would
    clear the floor on values that are identical across millions of users."""
    blocked = FP.entropy_bits({"canvas_hash": "blocked", "font_hash": "", "gpu_renderer": "none"})
    assert blocked == 0.0, f"null markers scored {blocked} bits"
    assert FP.entropy_bits({"canvas_hash": "9c8d7e6f"}) > 0


def test_a_rich_fingerprint_clears_the_floor():
    fp = FP.derive(_rich())
    assert fp["is_identity"] is True
    assert fp["entropy_bits"] >= FP.MIN_ENTROPY_BITS_FOR_IDENTITY
    assert fp["confidence"] in ("high", "medium")


def test_an_empty_payload_yields_no_identity_rather_than_an_error():
    """This sits on a scoring path. A hostile or malformed payload must degrade, never raise."""
    for bad in ({}, None, {"junk": "x"}, {"gpu_renderer": None}):
        fp = FP.derive(bad)
        assert fp["is_identity"] is False
        assert fp["device_id"] == "" or fp["confidence"] == "low"


# --------------------------------------------------------------------- automation

def test_automation_is_reported_even_when_identity_is_refused():
    """The point of keeping them separate. A scripted client usually runs a minimal, hardened
    browser carrying almost no entropy, so the fingerprint is worthless as an identity and the
    automation tells are the whole value."""
    fp = FP.derive({"cpu_cores": 2, "platform": "Linux x86_64",
                    "webdriver": True, "headless": True})
    assert fp["is_identity"] is False
    assert fp["automation_detected"] is True
    assert set(fp["automation_flags"]) >= {"webdriver", "headless"}


def test_a_clean_browser_reports_no_automation():
    fp = FP.derive(_rich())
    assert fp["automation_detected"] is False
    assert fp["automation_flags"] == []


def test_only_reported_tells_reach_telemetry():
    """Same discipline core/telemetry.py applies downstream: an absent signal stays absent rather
    than becoming a confident False, which would be an assertion nobody made."""
    assert FP.to_telemetry(FP.derive(_rich())) == {}
    t = FP.to_telemetry(FP.derive({"cpu_cores": 8, "emulator": True, "webdriver": True}))
    assert t.get("emulator") is True and t.get("automation_framework") is True
    assert "rooted_jailbroken" not in t


# --------------------------------------------------------------------- re-linking

def test_a_driver_update_relinks_to_the_same_device():
    """A GPU driver update genuinely moves the renderer string. Without re-linking that device
    loses its whole history and reappears as new, which is the slow way a device graph decays."""
    known = _rich()
    moved = _rich(gpu_renderer="ANGLE (Apple, Apple M2 Pro, OpenGL 4.6)")
    r = FP.relink(moved, known)
    assert r["same_device"] is True
    assert "gpu_renderer" in r["differed"]


def test_a_genuinely_different_machine_does_not_relink():
    """The bar is deliberately high: a wrong re-link merges two people's histories onto one
    device, which is worse than losing one device's history."""
    r = FP.relink(_rich(gpu_renderer="ANGLE (NVIDIA RTX 4070)", cpu_cores=24, screen_w=2560,
                        screen_h=1440, audio_dsp_hash="ffffffff", platform="Win32"), _rich())
    assert r["same_device"] is False


def test_relink_refuses_to_judge_on_too_little_evidence():
    r = FP.relink({"cpu_cores": 8}, {"cpu_cores": 8})
    assert r["same_device"] is False
    assert "too few" in r["why"]


# --------------------------------------------------------------------- input bounds

def test_an_oversized_payload_is_bounded_rather_than_derived_whole():
    """REGRESSION. Every value here is attacker-controlled and this runs on a request path.
    Unbounded, a 50,000-key payload with 1KB values was accepted and derived without complaint,
    and the endpoint then persisted it."""
    huge = {f"junk_{i}": "x" * 5000 for i in range(50_000)}
    huge.update(_rich())
    b = FP.bounded(huge)
    assert len(b) <= FP.MAX_COMPONENTS, f"{len(b)} components survived the bound"
    assert all(not k.startswith("junk_") for k in b), (
        "undeclared keys survived; the component contract is supposed to be closed")
    # the real components still made it through, so bounding did not cost the fingerprint
    assert FP.derive(huge)["is_identity"] is True


def test_over_long_values_are_truncated_not_rejected():
    """A real browser on unusual hardware can report a long GPU string. Losing that device
    entirely would be a worse outcome than trimming it."""
    long_gpu = _rich(gpu_renderer="ANGLE (" + "Z" * 10_000 + ")")
    b = FP.bounded(long_gpu)
    assert len(b["gpu_renderer"]) <= FP.MAX_VALUE_CHARS
    assert FP.derive(long_gpu)["device_id"], "an over-long value lost the whole fingerprint"


def test_structured_values_cannot_become_components():
    """str() on a dict yields a long repr that would otherwise be hashed in as a component
    value. Only scalars can be components."""
    fp = FP.derive({"gpu_renderer": {"nested": "dict"}, "cpu_cores": [1, 2],
                    "screen_w": 1920, "screen_h": 1080, "platform": "Win32"})
    assert fp["is_identity"] is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed ({len(tests)} total)")
    sys.exit(1 if failed else 0)
