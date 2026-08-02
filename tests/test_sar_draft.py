"""
Tests for the SAR grounding gate.

The document this guards is filed with a regulator under a named person's signature. The
failure it exists to prevent is a narrative that reads perfectly and states something the case
file does not: an amount never moved, an account never opened, a beneficiary that appears
nowhere in the evidence.

Two properties matter and they pull against each other:
  - it must catch invented facts, or it is decoration
  - it must NOT flag correct text, because a validator that cries wolf gets switched off, and
    the first version of this module failed its own draft by re-reading the "200" inside
    "$4,200" as an unsupported number

Runs under pytest or standalone (python3 tests/test_sar_draft.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import sar_draft as S  # noqa: E402


def _case():
    return {
        "transaction_id": "txn_88421",
        "transaction": {"transaction_id": "txn_88421", "user_id": "user_00042",
                        "amount": 4200.0, "payment_rail": "Zelle",
                        "recipient_id": "recv_019233", "fraud_typology": "APP_scam"},
        "disposition": {"action": "ESCALATE", "reasons": ["new payee", "amount 12x baseline"]},
    }


def test_an_invented_amount_is_refused():
    g = S.check_grounding("Subject moved $9,750 over Zelle.", _case())
    assert not g["grounded"]
    assert any(u["claim"] == "9,750" for u in g["unsupported"])


def test_an_invented_beneficiary_is_refused():
    g = S.check_grounding("Funds were sent to recv_555555.", _case())
    assert not g["grounded"]
    assert any(u["kind"] == "identifier" for u in g["unsupported"])


def test_an_invented_date_is_refused():
    g = S.check_grounding("The account was opened on 2024-03-02.", _case())
    assert not g["grounded"]
    assert any(u["kind"] == "date" for u in g["unsupported"])


def test_the_modules_own_draft_passes_its_own_gate():
    """THE bug this file caught first. A drafter whose output its own validator rejects is
    worse than no validator: it trains everyone to ignore the result."""
    d = S.draft_narrative(_case())
    assert d["grounding"]["grounded"], (
        f"our own draft was refused: {d['grounding']['unsupported']}")
    assert d["grounding"]["checked"] > 0, "nothing was checked, so passing means nothing"


def test_money_and_dates_are_not_re_read_as_bare_numbers():
    """The specific defect: '$4,200' contains '200' and '2024-03-02' contains '2024'. Both were
    reported as unsupported standalone numbers before the specific patterns masked their spans."""
    claims = S.claims_in("A payment of $4,200 on 2024-03-02.")
    numbers = [c for c in claims if c[0] == "number"]
    assert not numbers, f"fragments re-read as numbers: {numbers}"
    assert any(c[0] == "money" for c in claims)
    assert any(c[0] == "date" for c in claims)


def test_small_ordinals_in_prose_are_not_treated_as_claims():
    """'stage 3' and 'the second payment' are prose, not assertions of fact. Requiring them to
    be grounded produces noise rather than safety."""
    g = S.check_grounding("This is consistent with stage 3 of the typology.", _case())
    assert g["grounded"], f"ordinary prose was refused: {g['unsupported']}"


def test_a_grounded_narrative_passes_even_when_freehand():
    """The gate is drafter-agnostic on purpose: template, LLM or analyst, same check."""
    text = "Subject user_00042 sent $4,200 via Zelle to recv_019233 under txn_88421."
    assert S.check_grounding(text, _case())["grounded"]


def test_the_attestation_binds_to_the_exact_text():
    """A draft must not be editable between sign-off and filing, so the digest changes if a
    single character does."""
    a = S.narrative_sha("Subject sent $4,200.")
    b = S.narrative_sha("Subject sent $4,300.")
    assert a != b and len(a) == 16
    assert S.narrative_sha("Subject sent $4,200.") == a, "digest is not stable"


def test_an_empty_narrative_is_not_silently_grounded():
    """Nothing to check must not read as 'checked and fine'. It is reported as zero claims so a
    caller can refuse it, rather than passing an empty filing."""
    g = S.check_grounding("", _case())
    assert g["checked"] == 0


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
