"""
Tests for the authorization decision path.

Why this file exists. Screening, two-sided pricing, the decision policy, the model registry and
the decline contract all existed and none of them were reachable from an authorization, because
there was no authorization. The platform could score a payment and could not answer the only
question a card network ever asks: approve or decline, inside two seconds, with a reason code.

The tests are about the three things an issuer owes the network, because those are what make
this a payment system rather than a scorer with card-shaped inputs:

  THE BUDGET    miss the window and the network answers on your behalf. An issuer with no
                stand-in posture has not avoided the decision, it has let the network's default
                become its policy without anyone choosing that.

  THE CODE      `HOLD` means nothing to an acquirer. The mapping from an internal action to a
                response code is where a risk vocabulary becomes something a terminal, a
                merchant and a member can act on.

  SOFT vs HARD  networks limit re-attempts on specific codes and fine violations, so this is a
                contractual fact, not a UX preference. Any recovery flow built later has to
                respect it.

Runs under pytest or standalone (python3 tests/test_authorization.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import authorization as A  # noqa: E402

os.environ.setdefault("REDWING_RECOVERY_SECRET", "test-secret")


def _msg(**kw):
    m = {"amount": 42.0, "merchant_id": "mch_1", "merchant_name": "Acme Groceries",
         "cardholder_name": "Jane Roe", "entry_mode": "chip", "mcc_code": 5411,
         "account_age_days": 900, "available_balance": 5000.0}
    m.update(kw)
    return m


def _score(p, **detail):
    return lambda msg: (p, {"scored": True, **detail})


def _clock(sequence):
    """A fake monotonic clock in ms, so latency behaviour is deterministic rather than timed."""
    it = iter(sequence)
    last = [0.0]

    def now():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]
    return now


# ------------------------------------------------------------------- the response code

def test_an_ordinary_purchase_is_approved_with_00():
    r = A.authorize(_msg(), score_fn=_score(0.01))
    assert r["approved"] is True and r["response_code"] == "00"
    assert r["retry_allowed"] is True


def test_a_step_up_is_a_soft_decline_not_do_not_honor():
    """THE mapping that matters most. Under SCA the 65/1A family tells the merchant to
    re-attempt WITH authentication. Expressing a step-up as `05` throws away the retry the
    step-up exists to enable, and the member simply loses the purchase."""
    r = A.authorize(_msg(amount=900.0, account_age_days=5), score_fn=_score(0.55))
    if r["action"] == "STEP_UP":
        assert r["response_code"] == "65"
        assert r["soft_decline"] is True and r["retry_allowed"] is True


def test_a_risk_decline_stays_deliberately_vague():
    """`05` carries no information on purpose: an explanation is also a description of the
    control. Naming the risk reason in the response is how issuers hand attackers a roadmap,
    and it is why decline_contract exists to vary disclosure per member instead."""
    r = A.authorize(_msg(amount=4000.0, account_age_days=3), score_fn=_score(0.97))
    assert r["approved"] is False
    assert r["response_code"] in ("05", "65")
    assert "fraud" not in r["reason"].lower()
    assert "model" not in r["reason"].lower()


def test_insufficient_funds_is_named_because_the_member_can_act_on_it():
    """The opposite call from a risk decline, and for the same reason. The member's own balance
    is theirs to fix, so telling them costs nothing and saves the purchase. Returning `05` here
    is how a member ends up staring at "Do Not Honor" for a balance they could have topped up."""
    r = A.authorize(_msg(amount=800.0, available_balance=100.0), score_fn=_score(0.01))
    assert r["response_code"] == "51"
    assert r["soft_decline"] is True and r["member_situation"] is True


def test_limits_are_evaluated_before_the_risk_engine():
    """A real auth path checks the member's own situation first, and the order changes what
    they are told. A velocity limit reported as a risk decline is a lie the member cannot act
    on."""
    r = A.authorize(_msg(daily_count=12, daily_count_limit=12), score_fn=_score(0.01))
    assert r["response_code"] == "65" and r["member_situation"] is True


# ------------------------------------------------------------------- soft versus hard

def test_hard_declines_forbid_a_retry_and_soft_ones_permit_it():
    """Contractual, not cosmetic: networks limit re-attempts on specific codes and fine
    violations. A recovery flow that retries a hard decline is a scheme breach."""
    for code in A.SOFT_DECLINES:
        assert A.retry_allowed(code) is True, f"{code} should be retryable"
    for code in A.HARD_DECLINES:
        assert A.retry_allowed(code) is False, f"{code} must not be retried"


def test_no_code_is_both_soft_and_hard():
    assert not (set(A.SOFT_DECLINES) & set(A.HARD_DECLINES))


# ------------------------------------------------------------------- screening first

def test_screening_blocks_before_any_score_is_computed():
    """A payment to a designated party cannot be approved at any score, so there is no point
    computing one. If the scorer runs at all here, screening has been demoted to an input."""
    called = []

    def scorer(msg):
        called.append(1)
        return 0.0, {}
    r = A.authorize(_msg(merchant_name="Vostok Marine Holdings"), score_fn=scorer)
    assert r["approved"] is False
    assert called == [], "the model ran on a payment we are prohibited from processing"
    assert r["terminal"] is True


def test_a_screening_block_is_not_do_not_honor():
    """`57 Transaction Not Permitted` is the honest code, and it is distinguishable from a risk
    decline by anyone reading the response. Hiding a prohibition inside the generic risk code
    makes the two indistinguishable to an auditor."""
    r = A.authorize(_msg(merchant_name="Vostok Marine Holdings"), score_fn=_score(0.0))
    assert r["response_code"] == "57"
    assert A.retry_allowed("57") is False


# ------------------------------------------------------------------- the latency budget

def test_a_slow_decision_falls_to_stand_in_rather_than_answering_late():
    """THE property that makes this an issuer rather than a scorer. A correct answer after the
    window is a timeout, and a timeout means the network already decided."""
    # The path calls the clock twice before the stand-in check: once for t0, once for the
    # elapsed() test after scoring. An earlier five-element sequence assumed more calls than
    # happen and the second read came back as 0, so the check never tripped.
    clock = _clock([0, 1_300, 1_310])             # t0, the STIP check, then the response
    r = A.authorize(_msg(), score_fn=_score(0.01), now_ms=clock)
    assert r["stand_in"] is True
    assert r["action"] == "STAND_IN"
    assert any(s["step"] == "stand_in" for s in r["trail"])


def test_stand_in_is_a_decided_posture_not_a_blanket_approve():
    """Without a chosen posture the network's default becomes the institution's policy and
    nobody decided that. Conservative here because stand-in runs blind to everything this
    system knows."""
    assert A.stand_in({"amount": 40, "entry_mode": "chip"})["code"] == "00"
    assert A.stand_in({"amount": 4000, "entry_mode": "chip"})["code"] == "05"
    assert A.stand_in({"amount": 40, "entry_mode": "ecom"})["code"] == "05"


def test_every_response_reports_whether_it_made_the_window():
    """An issuer that cannot see its own late responses cannot know the network has been
    answering for it."""
    r = A.authorize(_msg(), score_fn=_score(0.01))
    assert "latency_ms" in r and r["within_budget"] is True


# ------------------------------------------------------------------------- the trail

def test_the_decision_carries_the_steps_that_produced_it():
    """A decision you cannot reconstruct is one you cannot defend to a member, a network or a
    regulator months later."""
    r = A.authorize(_msg(amount=600.0), score_fn=_score(0.4))
    steps = [s["step"] for s in r["trail"]]
    assert steps[0] == "screening", "screening must be first in the trail as well as in fact"
    assert "score" in steps and "priced" in steps and "policy" in steps


def test_the_scorer_is_injected_so_the_path_is_testable_without_the_ml_stack():
    """core/ stays importable without loading models, which is what lets these run in the
    stdlib suite alongside everything else."""
    r = A.authorize(_msg(), score_fn=None)
    assert r["response_code"] == "00"
    assert r["score_detail"]["scored"] is False


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
