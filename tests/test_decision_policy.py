"""
Tests for the decision policy layer.

Why this file exists. The live path collapsed eight possible actions into
`"HOLD" if is_alert else "ALLOW"`, so a $12 card payment and a $40,000 push to a three-day-old
payee were resolved identically. This layer bounds a PRICED action by the institution's
guardrails, and two properties have to hold or it becomes something else entirely:

  IT BOUNDS, IT DOES NOT DECIDE    the price chooses the action; policy only clamps it between a
                                   floor and a ceiling. A policy that could pick an action on
                                   its own would be a second risk opinion with no evidence
                                   behind it, which is exactly what this codebase refuses to do
                                   everywhere else.

  DE-ESCALATION IS RECORDED        a ceiling may soften an action, because real policies do that
                                   and forbidding it would push the behaviour into somebody's
                                   config. It must never be silent.

Runs under pytest or standalone (python3 tests/test_decision_policy.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import decision_policy as P  # noqa: E402


# ------------------------------------------------------------------ bounding, not deciding

def test_the_price_decides_when_policy_has_no_opinion():
    """The common case. If policy silently changed actions where it has no rule, nobody could
    tell which half of the system produced a decision."""
    d = P.decide("HOLD", 0.70, rail="ach", direction="outbound")
    assert d["action"] == "HOLD"
    assert d["bounded_by"] is None
    assert d["policy_escalated"] is False and d["policy_deescalated"] is False


def test_a_floor_lifts_an_action_the_economics_would_have_allowed():
    """THE reason a floor exists. A small push to a new payee can price out as ALLOW because the
    amount is too low to justify the false-positive cost, and an institution can still refuse to
    let an irrevocable payment go silently at that risk."""
    d = P.decide("ALLOW", 0.80, rail="zelle", direction="outbound")
    assert d["action"] == "STEP_UP"
    assert d["policy_escalated"] is True and d["bounded_by"] == "floor"
    assert "irrevocable" in d["rule"]["why"]


def test_a_ceiling_softens_an_action_and_says_so():
    """Softening must never be silent. It is a legitimate institutional choice and the first
    thing an auditor asks about, so it rides on the decision rather than in a config file."""
    d = P.decide("BLOCK", 0.70, rail="card")
    assert d["action"] == "STEP_UP"
    assert d["policy_deescalated"] is True and d["bounded_by"] == "ceiling"


def test_policy_cannot_invent_an_action_outside_the_ladder():
    """A malformed priced action must not become a decision. Defaulting to ALLOW is the
    conservative read here: the floor still applies, so risk is not lost."""
    d = P.decide("NONSENSE", 0.95, rail="zelle", direction="outbound")
    assert d["priced_action"] == "ALLOW"
    assert d["action"] == "HOLD", "the severe-band floor should still lift it"


# ---------------------------------------------------------------------- rule selection

def test_the_most_specific_rule_wins_not_the_first_one():
    """Anyone reading a decision matrix expects specificity to win. First-match-wins would make
    row ORDER load-bearing, and a policy whose meaning depends on line numbers is one nobody can
    safely edit."""
    pol = {
        "name": "t",
        "rules": [
            {"rail": "zelle", "floor": "MONITOR", "ceiling": "DECLINE", "why": "broad"},
            {"rail": "zelle", "direction": "outbound", "band": "high", "tier": "new_account",
             "floor": "HOLD", "ceiling": "HOLD", "why": "narrow"},
        ],
        "default": {"floor": "ALLOW", "ceiling": "DECLINE"},
    }
    d = P.decide("ALLOW", 0.70, rail="zelle", direction="outbound", tier="new_account",
                 policy=pol)
    assert d["rule"]["why"] == "narrow" and d["action"] == "HOLD"


def test_a_rule_that_does_not_match_is_not_applied():
    pol = {"name": "t",
           "rules": [{"rail": "zelle", "floor": "BLOCK", "ceiling": "BLOCK", "why": "zelle only"}],
           "default": {"floor": "ALLOW", "ceiling": "DECLINE"}}
    d = P.decide("ALLOW", 0.99, rail="card", policy=pol)
    assert d["action"] == "ALLOW", "a zelle rule fired on a card payment"


def test_a_pipe_list_matches_any_of_its_options():
    for rail in ("zelle", "fednow", "rtp", "wire", "crypto"):
        d = P.decide("ALLOW", 0.95, rail=rail, direction="outbound")
        assert d["action"] == "HOLD", f"{rail} did not match the irrevocable-rail rule"


def test_an_inverted_rule_honours_the_floor_and_flags_itself():
    """A ceiling below the floor is an authoring error. Silently picking one of them would make
    the matrix mean something other than what it says."""
    pol = {"name": "t",
           "rules": [{"rail": "zelle", "floor": "BLOCK", "ceiling": "ALLOW", "why": "typo"}],
           "default": {"floor": "ALLOW", "ceiling": "DECLINE"}}
    d = P.decide("ALLOW", 0.5, rail="zelle", policy=pol)
    assert d["inverted_rule"] is True
    assert d["action"] == "BLOCK", "the safety-side bound should win an authoring error"


# ------------------------------------------------------------------------- versioning

def test_the_version_tracks_content_not_a_hand_written_string():
    """A hand-maintained version drifts the moment someone edits a row and forgets to bump it,
    and a decision would then be attributed to a policy that never existed."""
    a = dict(P.DEFAULT_POLICY)
    b = {**a, "rules": a["rules"][:-1]}
    assert P.policy_version(a) != P.policy_version(b)


def test_reformatting_the_policy_is_not_a_policy_change():
    """Key order must not move the hash, or every reformat would read as a live policy change
    and the audit trail would fill with noise."""
    a = {"name": "t", "rules": [{"rail": "zelle", "floor": "HOLD", "ceiling": "BLOCK"}],
         "default": {"floor": "ALLOW", "ceiling": "BLOCK"}}
    b = {"default": {"ceiling": "BLOCK", "floor": "ALLOW"},
         "rules": [{"ceiling": "BLOCK", "floor": "HOLD", "rail": "zelle"}], "name": "t"}
    assert P.policy_version(a) == P.policy_version(b)


def test_every_decision_carries_the_version_that_produced_it():
    """The whole point of the layer. Without this an outcome change cannot be attributed to the
    policy change that caused it, and `decisions.policy_version` stays the empty column it has
    been since the substrate was built."""
    d = P.decide("HOLD", 0.7, rail="ach")
    assert d["policy_version"].startswith("redwing-default-us@")
    assert len(d["policy_version"].split("@")[1]) == 12


def test_a_broken_policy_file_falls_back_loudly_rather_than_leaving_no_policy():
    """This sits in the live path. A bad file must not mean an unbounded decision."""
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "bad.json")
    open(p, "w").write("{ not json")
    pol = P.load_policy(p)
    assert pol["rules"] == P.DEFAULT_POLICY["rules"]
    assert "load_error" in pol


# ------------------------------------------------------------------------ score bands

def test_bands_are_inclusive_at_their_lower_edge():
    assert P.band_for(0.65) == "high" and P.band_for(0.6499) == "elevated"
    assert P.band_for(0.90) == "severe"
    assert P.band_for(0.0) == "low"


def test_an_unreadable_score_lands_in_the_lowest_band_not_the_highest():
    """Failing safe here means failing OPEN, which is the right call: an unparseable score is a
    data-quality problem and must not manufacture a block on a real member's payment."""
    assert P.band_for(None) == "low" and P.band_for("abc") == "low"


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
