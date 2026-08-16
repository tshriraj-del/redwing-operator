"""
Tests that /score branches to the card scorer, and that the sequence gate reaches it.

WHY THIS FILE EXISTS. `/score` computed `ml_score_row(features)` unconditionally, so a card-rail
payment arriving at this endpoint was scored by the PUSH model: a model that looks for velocity
and recipient familiarity an ISO 8583 message does not carry. `build_event()` and
`core.authorization.authorize()` both branched correctly. `/score` did not, which made it the
seventh instance of ADR-001's recurring failure, a control wired to one decision path and
forgotten on another.

Two entries in the conformance test's KNOWN_GAPS covered this, ("score", "card_model") and its
dependent ("score", "sequence_gate"), and closing the first closes the second: the gate lives
inside `score_card_message_gated`, so a path that reaches the card scorer reaches its gate and a
path that does not cannot.

THE ASSERTION THAT MATTERS IS NOT "THE KEY IS PRESENT". Adding a `card_score_detail` key to the
response would satisfy a shape check while the push model went on producing the number. So the
test below compares the reported `ml_score` against BOTH models computed independently, and
requires it to match the card one and differ from the push one. A branch that does not change
which model scored is not a branch.

AND THE RAIL IS CANONICALISED, NOT COMPARED TO A LITERAL. `debit_card` and `credit_card` are
declared synonyms of `card` in ingest_schema.RAILS. `build_event` shipped with
`rail.strip().lower() == "card"`, which silently routed both back to the push model, which is
the identical blindness this branch exists to remove. It is asserted here rather than assumed.

Runs under pytest or standalone (python3 tests/test_score_card_rail.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_RECOVERY_SECRET", "score-card-rail-test")
os.environ.setdefault("REDWING_CARD_SALT", "score-card-rail-salt")
os.environ.setdefault("REDWING_ALLOW_OPEN", "i-understand-this-is-open")  # auth fails closed; tests run the app in-process

CARD_BODY = {
    "transaction_id": "sc_card_0001", "user_id": "user_00001", "amount": 900.0,
    "payment_rail": "card", "recipient_id": "r_sc", "device_id": "d_sc",
    "recipient_name": "Acme Supplies", "entry_mode": "ecom", "mcc_code": 5999,
    "bin": "400000", "merchant_id": "m_sc", "account_age_days": 30,
    "card_token": "tok_score_rail_test",
}

PUSH_BODY = {**CARD_BODY, "transaction_id": "sc_push_0001", "payment_rail": "faster_payments"}


class Unavailable(Exception):
    """The ML stack is genuinely absent, so the branch cannot be judged. A legitimate skip."""


def _client_and_main():
    """A main.py that will not import is a FAILURE, not a skip. Only a missing model is a skip.

    Copied deliberately from test_path_conformance, which learned this the hard way: catching
    both cases together turned a whole conformance file green against a main.py that did not
    parse.
    """
    try:
        from fastapi.testclient import TestClient
    except Exception as e:                                        # noqa: BLE001
        raise Unavailable(f"fastapi test client unavailable: {type(e).__name__}") from e
    try:
        import main
    except Exception as e:                                        # noqa: BLE001
        raise AssertionError(
            f"main.py does not import ({type(e).__name__}: {e})") from e
    if not getattr(main, "MODEL_OK", False):
        raise Unavailable("models are not loaded; run the ML pipeline first")
    return TestClient(main.app, raise_server_exceptions=False), main


def _score(client, body):
    r = client.post("/score", json=body)
    assert r.status_code == 200, f"/score returned {r.status_code}: {r.text[:300]}"
    return r.json()


# ------------------------------------------------------------------ the branch

def test_a_card_rail_payment_is_scored_by_the_card_model():
    """THE gap. Evidence is the detail block the card scorer produces, which the push model
    has no way to emit."""
    client, _ = _client_and_main()
    d = _score(client, CARD_BODY)
    detail = d.get("card_score_detail")
    assert detail, "no card_score_detail on a card-rail /score response"
    assert detail.get("model") == "card_scorer", (
        f"a card payment was scored by {detail.get('model')!r}, not the card scorer")


def test_the_reported_score_is_the_card_model_s_number_and_not_the_push_model_s():
    """The assertion a shape check cannot make. Both models are computed here independently and
    the response must match the card one. Without this, adding the detail key while leaving
    `ml = ml_score_row(features)` in place would pass every other test in this file."""
    client, main = _client_and_main()
    d = _score(client, CARD_BODY)

    card_p, _ = main.score_card_message_gated(dict(CARD_BODY))
    push_p = main.ml_score_row(main.compute_features(dict(CARD_BODY)))

    assert abs(d["ml_score"] - round(card_p, 4)) < 1e-4, (
        f"ml_score {d['ml_score']} is not the card model's {round(card_p, 4)}")
    assert abs(round(card_p, 4) - round(push_p, 4)) > 1e-9, (
        "the two models returned the same number on this fixture, so this test cannot tell them "
        "apart; change the fixture rather than weakening the assertion")


def test_rail_synonyms_route_to_the_card_model_too():
    """A BUG THAT SHIPPED ON THE OTHER PATH. `debit_card` and `credit_card` are synonyms of
    `card` in ingest_schema.RAILS. Comparing to the literal "card" sent both back to the push
    model, which is exactly the blindness the branch removes."""
    client, _ = _client_and_main()
    for rail in ("debit_card", "credit_card", "CARD", " card "):
        body = {**CARD_BODY, "transaction_id": f"sc_{rail.strip()}", "payment_rail": rail}
        detail = _score(client, body).get("card_score_detail") or {}
        assert detail.get("model") == "card_scorer", (
            f"payment_rail={rail!r} did not reach the card model")


def test_a_push_rail_payment_is_untouched_by_the_branch():
    """The branch must not capture traffic it was not built for. A push payment scored by the
    card model would be the same defect pointed the other way."""
    client, _ = _client_and_main()
    d = _score(client, PUSH_BODY)
    assert not d.get("card_score_detail"), (
        "a push-rail payment came back with a card score detail")


# ------------------------------------------------------------------ the gate

def test_the_sequence_gate_is_reachable_on_this_path():
    """Closes ("score", "sequence_gate"). The gate lives inside score_card_message_gated, so
    this is not separately wired; it is reachable because the card scorer is. Asserting the view
    is present is what proves the gate RAN rather than that the code mentions it."""
    client, _ = _client_and_main()
    detail = _score(client, CARD_BODY).get("card_score_detail") or {}
    gate = detail.get("sequence_gate")
    assert isinstance(gate, dict) and "available" in gate, (
        f"no sequence gate view on the /score card path: {gate!r}")


def test_the_gate_on_this_path_is_escalate_only():
    """Non-negotiable 5. The gate may raise a score and may never lower one. Checked on THIS
    path rather than trusted from the gate's own tests, because the escalate-only property is a
    property of the wiring as much as of the gate."""
    client, main = _client_and_main()
    d = _score(client, CARD_BODY)
    ungated, _ = main.score_card_message(dict(CARD_BODY))
    assert d["ml_score"] >= round(ungated, 4) - 1e-9, (
        f"the gate LOWERED the score on /score: {d['ml_score']} < {round(ungated, 4)}")


def test_a_card_message_with_no_card_identifier_still_scores():
    """The gate degrades, it does not fail the request. A card with no token and no PAN has no
    history to read, and a gate that cannot see must cost the gate and never the score."""
    client, _ = _client_and_main()
    body = {k: v for k, v in CARD_BODY.items() if k != "card_token"}
    body["transaction_id"] = "sc_no_key"
    d = _score(client, body)
    detail = d.get("card_score_detail") or {}
    assert detail.get("model") == "card_scorer"
    assert detail["sequence_gate"]["available"] is False
    assert detail["sequence_gate"]["fired"] is False


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
