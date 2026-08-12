"""
Tests for the screening gate.

Why this file exists. Fifteen agency connectors existed and none of them touched a decision, and
`case_file.py` produced its sanctions verdict with `r.random() < 0.01` a few lines above a field
reporting `"sanctions_screened": True`. That is the most dangerous line the repository has had:
it asserts a control was applied while applying nothing.

Screening is not a risk signal and these tests exist to keep it from becoming one:

  IT RUNS FIRST     a payment to a designated party cannot be approved at any score, under any
                    posture, past any policy ceiling. There is no trade, so it must not sit
                    inside anything that trades.

  IT FAILS CLOSED   every advisory layer here fails silent, which is right for advisory
                    signals. An unavailable screening list must NOT mean "approve unscreened",
                    because that is the failure the control exists to prevent.

  IT NEVER OFFERS   a screening block is terminal. A system that told someone how to get a
    A WAY BACK      sanctions match through would be doing something far worse than losing a
                    sale.

Runs under pytest or standalone (python3 tests/test_screening.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import screening as S  # noqa: E402

LIST = os.path.join(ROOT, "data", "sanctions_list.txt")


def _tmp_list(text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "list.txt")
    open(p, "w").write(text)
    return p


# ------------------------------------------------------------------- fail closed

def test_a_missing_list_blocks_rather_than_approving_unscreened():
    """THE inverted failure posture, and the reason this module exists apart from the others.
    The novelty gate failing silent costs a second opinion. Screening failing silent means
    approving a payment nobody screened, which is the failure the control is for."""
    r = S.screen(counterparty="Anyone", path="/nonexistent/list.txt")
    assert r["blocked"] is True
    assert r["screened"] is False
    assert r["result"] == S.UNAVAILABLE
    assert r["code"] == S.CODE_UNAVAILABLE
    assert "not a degraded mode" in r["reason"]


def test_an_empty_list_is_not_treated_as_everybody_being_clear():
    """An empty file and a clean world are indistinguishable to a naive loader, and one of them
    approves everything."""
    r = S.screen(counterparty="Anyone", path=_tmp_list("# only comments\n\n"))
    assert r["blocked"] is True and r["result"] == S.UNAVAILABLE


def test_status_reports_whether_the_control_is_actually_in_force():
    """The question `sanctions_screened: True` used to answer without checking anything."""
    S.get_list(LIST, reload=True)
    st = S.status()
    assert st["available"] is True and st["entries"] > 0 and st["fails"] == "closed"


# ---------------------------------------------------------------------- matching

def test_an_exact_designated_party_is_a_confirmed_match():
    r = S.screen(counterparty="Vostok Marine Holdings", path=LIST)
    assert r["result"] == S.CONFIRMED
    assert r["blocked"] is True and r["reporting_obligation"] is True


def test_an_alias_matches_the_canonical_entry():
    """Designated parties use aliases and transliterations. Screening that only caught the
    canonical spelling would be theatre."""
    r = S.screen(counterparty="VMH Shipping", path=LIST)
    assert r["result"] == S.CONFIRMED and r["matched"] == "Vostok Marine Holdings"


def test_corporate_suffixes_and_punctuation_do_not_defeat_it():
    """"Vostok Marine Holdings, Ltd." is the same counterparty. An exact string match is the
    naive implementation and it fails on the first comma."""
    for variant in ("Vostok Marine Holdings Ltd", "vostok marine holdings",
                    "Vostok  Marine  Holdings."):
        r = S.screen(counterparty=variant, path=LIST)
        assert r["blocked"] is True, f"{variant!r} slipped through"


def test_a_dropped_middle_name_is_a_potential_match_not_a_clear():
    """Partial name matches are where real screening lives. Treating them as clear is how a
    designated individual gets through under a shortened name."""
    r = S.screen(counterparty="Anton Kuznetsov", path=LIST)
    assert r["result"] == S.POTENTIAL
    assert r["blocked"] is True and r["requires_human"] is True


def test_a_potential_match_is_not_an_accusation_and_says_so():
    """Most potential matches on common names are innocent people. The wording matters because
    an operations team reads it, and a system that accuses everyone gets overridden by habit."""
    r = S.screen(counterparty="Anton Kuznetsov", path=LIST)
    assert r["reporting_obligation"] is False
    assert "not real" in r["reason"] and "review" in r["reason"]


def test_an_ordinary_counterparty_is_cleared():
    r = S.screen(counterparty="Acme Groceries", path=LIST)
    assert r["result"] == S.CLEAR and r["blocked"] is False and r["screened"] is True


def test_the_member_is_screened_as_well_as_the_counterparty():
    """Sanctions apply to both ends. Screening only the payee misses a designated sender."""
    r = S.screen(counterparty="Acme Groceries", member="Vostok Marine Holdings", path=LIST)
    assert r["blocked"] is True and r["subject"] == "member"


def test_an_empty_name_does_not_match_everything():
    """Token containment on an empty set is vacuously true, which would block every payment
    with a missing counterparty name."""
    assert S.screen(counterparty="", member="", path=LIST)["blocked"] is False


# --------------------------------------------------------------------- terminal

def test_a_screening_block_is_terminal_and_carries_its_own_code():
    """It must be distinguishable from a fraud decline by an analyst, an auditor and a support
    agent. "We think this is fraud" and "we are prohibited from processing this" are different
    conversations."""
    for who, code in (("Vostok Marine Holdings", S.CODE_SANCTIONS),
                      ("Anton Kuznetsov", S.CODE_WATCHLIST)):
        r = S.screen(counterparty=who, path=LIST)
        assert r["terminal"] is True and r["code"] == code
        assert not r["code"].isdigit(), "a screening code must not look like a network decline"


def test_the_decline_contract_refuses_to_offer_a_way_back_from_a_screening_block():
    """The join that matters. A recovery path that could resurrect a sanctioned payment would be
    a compliance failure, not a product one."""
    from core import decline_contract as D
    c = D.contract(decline_id="d1", member_id="u1", code=S.CODE_SANCTIONS,
                   cause="screening", amount=500.0)
    assert c["recoverable"] is False
    assert c["required_action"] is None
    for lvl in D.DISCLOSURE_LEVELS:
        assert "recovery_token" not in c["disclosure"][lvl]


# ------------------------------------------------------------- the coin flip is gone

def test_the_case_file_no_longer_decides_sanctions_with_a_random_number():
    """`sanctions_hit = r.random() < 0.01` sat above `"sanctions_screened": True`. An
    investigator reading "Sanctions/watchlist potential match" had no way to know it was noise."""
    import ast
    # Parsed, not grepped. The first version matched the COMMENT that quotes the old line while
    # explaining why it was removed, which is the same false positive the challenge-set
    # evaluator's `.fit()` guard hit. A test that fails on its own documentation teaches people
    # to delete the documentation.
    tree = ast.parse(open(os.path.join(ROOT, "case_file.py")).read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "sanctions_hit" in names:
                src = ast.dump(node.value)
                assert "random" not in src, "sanctions_hit is still decided by a random draw"

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "sanctions_screened"
                        and isinstance(v, ast.Constant) and v.value is True):
                    assert False, "the panel still hardcodes that screening happened"


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
