"""
Tests that the device gate reaches the CARD rail, including /authorize.

WHY THIS FILE EXISTS. ("authorize", "device_gate") has been a KNOWN_GAP across many sessions on
one excuse: an ISO 8583 message carries no device. That is true for card-PRESENT and false for
card-not-present, and the fixtures say so themselves, because they set entry_mode="ecom". CNP is
where the device exists (3DS device channel, merchant SDK), and CNP is where card testing lives.

WHAT THE MEASUREMENTS SAID, 2026-08-15, and why this is not a tidiness fix:

  the card model carries ZERO device features
      CATEGORICAL = entry_mode, channel, card_type, avs_result, cvv_result, three_ds
      NUMERIC     = amount, tokenized, mcc_high_risk, bin_fraud_rate, merchant_fraud_rate,
                    amount_log
  and its top five features by importance are all verification RESULTS (avs/cvv/3ds, ~48% of
  total importance). A card tester's entire objective is finding a card that PASSES those checks,
  so at the moment the attack succeeds every one of the model's strongest signals reads clean.

  Measured recall on the challenge ledger: card_testing_bot 1.6%, model p50 0.0008. Nothing sees
  it. Device is one of the two signals that could, the other being merchant fan-in.

WHERE IT GOES, and this is an ADR-001 decision rather than a convenience. The gate is applied
inside `score_card_message_gated`, which is THE entry point every card path uses, so /authorize,
/score and build_event all get it from one edit. Wiring it onto one handler is precisely how this
codebase accumulated six controls on one path and not another.

DOUBLE APPLICATION IS SAFE AND DELIBERATE. build_event and /score already call apply_device_gate
on the post-model score. A card row on those paths now meets the gate twice. Because the contract
is escalate-only via max(), max(max(s,x),x) == max(s,x), so the second application is a no-op.
Applying it inside the scorer is the MORE correct of the two, because it lands before pricing.

Runs under pytest or standalone (python3 tests/test_card_device_gate.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_RECOVERY_SECRET", "card-device-gate-test")
os.environ.setdefault("REDWING_CARD_SALT", "card-device-gate-salt")
os.environ.setdefault("REDWING_ALLOW_OPEN", "i-understand-this-is-open")

CARD_MSG = {"amount": 900.0, "merchant_name": "Acme Supplies", "merchant_id": "m_dg",
            "cardholder_name": "Jane Roe", "entry_mode": "ecom", "mcc_code": 5999,
            "account_age_days": 30, "available_balance": 9000.0, "bin": "400000",
            "arn": "dg_arn_0001", "card_token": "tok_dg_1", "device_id": "dev_dg_1"}


class Unavailable(Exception):
    """The ML stack is genuinely absent. A legitimate skip."""


def _main():
    try:
        import main
    except Exception as e:                                        # noqa: BLE001
        raise AssertionError(f"main.py does not import ({type(e).__name__}: {e})") from e
    if not getattr(main, "MODEL_OK", False):
        raise Unavailable("models are not loaded")
    return main


# ── the gate reaches the card scorer at all ──────────────────────────────────

def test_the_card_scorer_reports_a_device_view():
    """Presence of the VIEW is what proves the gate ran. A gate that is merely mentioned in the
    source is the thing the conformance test exists to reject."""
    main = _main()
    _, detail = main.score_card_message_gated(dict(CARD_MSG))
    dv = detail.get("device_gate")
    assert isinstance(dv, dict) and "available" in dv, (
        f"no device gate view on the card scorer: {dv!r}")


def test_the_device_gate_is_escalate_only_on_the_card_rail():
    """Non-negotiable 5. It may raise a score and may never lower one, and that is a property of
    the WIRING as much as of the gate."""
    main = _main()
    ungated, _ = main.score_card_message(dict(CARD_MSG))
    gated, _ = main.score_card_message_gated(dict(CARD_MSG))
    assert gated >= ungated - 1e-9, f"the gate LOWERED a card score: {gated} < {ungated}"


def test_a_flagged_device_raises_the_card_score():
    """The gate must actually be able to change the number, not merely report. Injected rather
    than waiting for a real shared-thin device, because the real one fires on 0.0133% of traffic
    and a test that needs it would never run."""
    main = _main()
    base, _ = main.score_card_message(dict(CARD_MSG))
    orig = main.apply_device_gate
    main.apply_device_gate = lambda s, row: (max(float(s), 0.80),
                                             {"available": True, "fired": True,
                                              "escalated": max(float(s), 0.80) > float(s)})
    try:
        raised, detail = main.score_card_message_gated(dict(CARD_MSG))
    finally:
        main.apply_device_gate = orig
    assert raised >= 0.80, f"a flagged device did not raise the score: {raised}"
    assert raised > base or base >= 0.80
    assert (detail.get("device_gate") or {}).get("fired") is True


def test_a_message_with_no_device_degrades_rather_than_failing():
    """Card-present genuinely has no device. That must cost the gate, never the authorization."""
    main = _main()
    msg = {k: v for k, v in CARD_MSG.items() if k != "device_id"}
    msg["entry_mode"] = "chip"
    score, detail = main.score_card_message_gated(msg)
    assert isinstance(score, float)
    assert isinstance(detail.get("device_gate"), dict)


# ── the path that never had it ───────────────────────────────────────────────

def test_authorize_applies_the_device_gate():
    """THE gap. /authorize scored cards for its whole life with no device signal at any layer:
    none in the model's feature set, and no gate on the path."""
    main = _main()
    try:
        from fastapi.testclient import TestClient
    except Exception as e:                                        # noqa: BLE001
        raise Unavailable(str(e)) from e
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.post("/authorize", json=dict(CARD_MSG))
    assert r.status_code == 200, r.text[:200]
    sd = r.json().get("score_detail") or {}
    dv = sd.get("device_gate")
    assert isinstance(dv, dict) and "available" in dv, (
        f"/authorize produced no device gate view: {sd.keys()}")


def test_score_applies_it_on_the_card_branch_too():
    """Same control, same entry point, every card path. ADR-001."""
    main = _main()
    try:
        from fastapi.testclient import TestClient
    except Exception as e:                                        # noqa: BLE001
        raise Unavailable(str(e)) from e
    c = TestClient(main.app, raise_server_exceptions=False)
    body = {**CARD_MSG, "transaction_id": "dg_score_1", "payment_rail": "card",
            "user_id": "u_dg", "recipient_name": "Acme Supplies"}
    r = c.post("/score", json=body)
    assert r.status_code == 200, r.text[:200]
    dv = (r.json().get("card_score_detail") or {}).get("device_gate")
    assert isinstance(dv, dict) and "available" in dv


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Unavailable as e:
            print(f"  SKIP  {t.__name__}: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped ({len(tests)} total)")
    sys.exit(1 if failed else 0)
