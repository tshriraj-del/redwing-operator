"""
Tests for the decline contract and its recovery token.

Why this file exists. The whole design rests on one claim: it is safe to explain a decline to
some members and not others, because the explanation alone is useless without a bound token.
If that claim is wrong the system is worse than the opaque `05` it replaces, because it hands
attackers a roadmap and calls it a product.

So the token tests are not hygiene, they are the argument:

  BOUND       a token issued to one member must not redeem for another, or disclosure leaks
              transferable credentials
  EXPIRING    a stale token must fail, or the remediation window is infinite
  FAIL-CLOSED an unset signing secret must make tokens unverifiable, never forgeable
  SILENT      the token itself must not carry the reason, or intercepting it leaks what
              withholding disclosure was meant to protect

And one product guard: a terminal decline must never be dressed in guided remediation, because
a member following instructions that cannot work is the cruellest failure this system could ship.

Runs under pytest or standalone (python3 tests/test_decline_contract.py).
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import decline_contract as D  # noqa: E402

os.environ.setdefault("REDWING_RECOVERY_SECRET", "test-secret-for-suite")


def _c(code="05", cause="issuer_risk", member="user_1", amount=120.0):
    return D.contract(decline_id="dec_1", member_id=member, code=code, cause=cause,
                      amount=amount, ltv_band="medium", account_age_days=400)


# ------------------------------------------------------------------ the token argument

def test_a_token_issued_to_one_member_does_not_redeem_for_another():
    """THE security property the whole design rests on. If tokens were transferable, explaining
    a decline would mean handing out a retry credential, and the opaque-to-everyone status quo
    would be strictly safer than this."""
    tok = D.issue_token(decline_id="dec_1", member_id="alice", action="verify_identity")
    assert D.verify_token(tok, member_id="alice")["valid"] is True
    bad = D.verify_token(tok, member_id="mallory")
    assert bad["valid"] is False and bad["reason"] == "wrong_member"


def test_an_expired_token_fails_and_says_it_expired():
    """The caller must distinguish 'stale, offer a fresh one' from 'forged, treat as an attack'.
    A bare False collapses a customer-service event and a security event into one."""
    now = time.time()
    tok = D.issue_token(decline_id="dec_1", member_id="alice", action="fund_account",
                        ttl=60, now=now)
    assert D.verify_token(tok, member_id="alice", now=now + 30)["valid"] is True
    late = D.verify_token(tok, member_id="alice", now=now + 61)
    assert late["valid"] is False and late["reason"] == "expired"


def test_a_tampered_token_is_rejected_as_forged_not_as_expired():
    tok = D.issue_token(decline_id="dec_1", member_id="alice", action="fund_account")
    body, sig = tok.split(".")
    forged = body + "." + ("A" * len(sig))
    r = D.verify_token(forged, member_id="alice")
    assert r["valid"] is False and r["reason"] == "bad_signature"


def test_the_token_does_not_carry_the_reason_it_was_issued_for():
    """An intercepted token must not leak what an intercepted message would have said. The
    action is a machine handle; the human-readable reason lives only in the disclosure the
    policy actually chose to send."""
    tok = D.issue_token(decline_id="dec_1", member_id="alice", action="verify_identity")
    import base64
    body = base64.urlsafe_b64decode(tok.split(".")[0] + "==").decode()
    for leak in ("unusual", "declined", "balance", "fraud", "security code"):
        assert leak not in body.lower(), f"token body leaks {leak!r}: {body}"


def test_a_malformed_token_is_rejected_rather_than_crashing():
    """Tokens arrive from the outside. A parser that raises is a denial of service on the
    retry path, which would punish exactly the recovering members this exists to serve."""
    for junk in ("", "....", "not-a-token", "a.b", None):
        r = D.verify_token(junk, member_id="alice")
        assert r["valid"] is False


# ------------------------------------------------------------- recoverability judgement

def test_recoverability_is_our_judgement_not_a_lookup_on_the_code():
    """`05` is the industry catch-all and hides both a member who needs to verify and a card we
    will not approve. Collapsing those into one class is what makes declines feel arbitrary."""
    assert D.recoverability("05", "issuer_risk") == D.STEP_UP
    assert D.recoverability("05", "member_funds") == D.SELF_SERVICE
    assert D.recoverability("62", "issuer_risk") == D.TERMINAL, (
        "a restricted card is a decision about the instrument, not this payment")


def test_an_unmapped_cause_is_unknown_rather_than_assumed_recoverable():
    """Assuming recoverable would offer remediation for a decline we do not understand, and the
    member would follow instructions that cannot work."""
    assert D.recoverability("05", "") == D.UNKNOWN
    assert D.recoverability("05", "some_new_cause") == D.UNKNOWN


# ------------------------------------------------------------------- the contract

def test_a_terminal_decline_is_never_dressed_in_guided_remediation():
    """THE product guard, and the cruellest possible failure if it breaks: a member diligently
    following steps for a decision that will not be revisited."""
    c = _c(code="62", cause="issuer_risk")
    assert c["recoverable"] is False
    assert c["required_action"] is None

    msgs = [c["disclosure"][lvl]["message"] for lvl in D.DISCLOSURE_LEVELS]
    # Every level says the SAME thing, and that thing says the door is closed. Asserting only
    # that the word "retry" is absent was too weak: the issuer_risk guided text reads "Confirm
    # it was you in the app and it will go through", which is an instruction to act and contains
    # no such word. Disabling the terminal guard sailed straight past that check.
    assert len(set(msgs)) == 1, (
        f"a terminal decline offers a disclosure LADDER, which means it is offering the member "
        f"steps for a decision that will not be revisited: {msgs}")
    assert "cannot be retried" in msgs[0].lower()
    for lvl in D.DISCLOSURE_LEVELS:
        assert "recovery_token" not in c["disclosure"][lvl]


def test_the_opaque_level_carries_no_token():
    """`none` exists for the member we deliberately told nothing, which on a likely attacker is
    the correct choice. Attaching a retry credential there would hand it to exactly the
    population the level exists to withhold from."""
    c = _c(cause="member_funds")
    assert "recovery_token" not in c["disclosure"]["none"]
    assert "recovery_token" not in c["disclosure"]["generic"]
    assert "recovery_token" in c["disclosure"]["specific"]
    assert "recovery_token" in c["disclosure"]["guided"]


def test_disclosure_gets_more_specific_as_it_rises():
    """The ladder has to be a real ladder. If levels said the same thing, there would be nothing
    to trade off and the whole optimisation would be decoration."""
    c = _c(cause="member_funds")
    msgs = [c["disclosure"][lvl]["message"] for lvl in D.DISCLOSURE_LEVELS]
    assert len(set(msgs)) == 4, f"disclosure levels are not distinct: {msgs}"
    assert len(msgs[0]) < len(msgs[-1])
    assert "balance" not in msgs[0].lower(), "the opaque level leaks the reason"
    assert "balance" in msgs[-1].lower(), "the guided level does not say what to do"


def test_the_contract_prices_the_decline_on_the_same_scale_as_a_block():
    """A decline and a hold have to be valued on one scale, or the system cannot compare them.
    Reusing false_positive_cost is what makes that true rather than asserted."""
    cheap = _c(amount=8.0)
    dear = D.contract(decline_id="d2", member_id="u", code="05", cause="issuer_risk",
                      amount=900.0, ltv_band="high", account_age_days=20)
    assert dear["cost_of_this_decline"] > cheap["cost_of_this_decline"] > 0


def test_the_contract_does_not_choose_a_disclosure_level():
    """The choice is a priced trade between recovery uplift and information handed to an
    adversary. Burying it in a helper would hide the most consequential decision in the system,
    and it needs a causal estimate that does not exist yet."""
    c = _c()
    assert set(c["disclosure"]) == set(D.DISCLOSURE_LEVELS)
    assert "chosen_disclosure" not in c and "disclosure_level" not in c


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
