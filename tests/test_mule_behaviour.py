"""
Tests for the observable witting-ness tells.

Why this file exists. The graduation gate reported paired_with_heuristic = 0 for every target,
and it could not rise: the witting-ness heuristic never ran, because its twelve tells and the
session telemetry the scorer collects have exactly zero overlap. A gate that can never fire is
not a gate, so the heuristic needed inputs it can actually get.

The danger in giving it those inputs is the reason most of these tests exist. The two easiest
tells to compute both point at guilt, and a heuristic that can only ever see evidence of guilt
finds guilt in everyone. Those labels would then seed a classifier, which is how a system
launders its own assumptions into training data. So the exculpatory direction is tested as
carefully as the incriminating one.

Runs under pytest or standalone (python3 tests/test_mule_behaviour.py).
"""

import os
import sys
from types import SimpleNamespace as D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import mule_behaviour as B          # noqa: E402
from core.mule_network import classify_mule   # noqa: E402


def test_repeated_warnings_read_as_witting():
    """The tell is REPEATED friction, not "was warned once and kept banking". Keying on the
    latter measured 68% of all accounts as witting, because continuing to transact after a
    single step-up is what every ordinary customer does."""
    t = B.warning_tells([D(ts="2026-01-01", action="STEP_UP"),
                         D(ts="2026-01-02", action="ALLOW"),
                         D(ts="2026-01-03", action="HOLD"),
                         D(ts="2026-01-04", action="BLOCK")])
    assert "continues_after_warning" in t
    assert classify_mule(t)["role"] == "witting"


def test_a_single_warning_is_not_evidence_of_defiance():
    """One step-up followed by ordinary activity describes almost everybody. Reading it as
    defiance is how this module first put 68% of accounts at witting."""
    t = B.warning_tells([D(ts="2026-01-01", action="STEP_UP"),
                         D(ts="2026-01-02", action="ALLOW"),
                         D(ts="2026-01-03", action="ALLOW")])
    assert "continues_after_warning" not in t


def test_stopping_after_a_warning_reads_as_unwitting():
    """THE balance test. If only the incriminating half of this evidence were derived, the
    heuristic could never exonerate anyone, and every label it seeded would point one way."""
    t = B.warning_tells([D(ts="2026-01-01", action="ALLOW"),
                         D(ts="2026-01-02", action="STEP_UP")])
    assert "stops_on_warning" in t, "the exculpatory half of the same evidence is missing"
    assert classify_mule(t)["role"] == "unwitting"


def test_a_customer_who_was_never_warned_produces_no_evidence_either_way():
    """Absence of a warning is not absence of guilt, and it is not evidence of it. The tell
    simply does not exist for that account, which classify_mule turns into 'undetermined'."""
    t = B.warning_tells([D(ts="2026-01-01", action="ALLOW"), D(ts="2026-01-02", action="ALLOW")])
    assert t == {}
    assert classify_mule(t)["role"] == "undetermined"


def test_stopping_is_weighted_weaker_than_carrying_on():
    """We cannot tell "they stopped" from "we blocked them and they had no chance to continue",
    so the exculpatory read is deliberately held weak rather than pretended to be certain."""
    stop = B.warning_tells([D(ts="2026-01-01", action="ALLOW"), D(ts="2026-01-02", action="BLOCK")])
    cont = B.warning_tells([D(ts=f"2026-01-{d:02d}", action="BLOCK") for d in range(1, 6)])
    assert stop["stops_on_warning"] < cont["continues_after_warning"]


def test_fanin_counts_distinct_senders_not_payments():
    """One victim groomed over twenty payments is ONE source. Counting transactions would turn
    a pig-butchering victim's own escalating ramp into evidence that their payee is a collector,
    which inverts who is being accused."""
    assert B.fanin_tell([("user:v1", 1)] * 20) == {}
    many = B.fanin_tell([(f"user:v{i}", 1) for i in range(12)])
    assert many.get("many_victim_sources", 0) > 0.9


def test_an_ordinary_payee_with_a_few_senders_is_not_a_collector():
    assert B.fanin_tell([(f"user:v{i}", 0) for i in range(3)]) == {}


def test_no_evidence_yields_undetermined_rather_than_a_guess():
    """The correct answer far more often than not. A heuristic that always produces a role is
    producing noise, and the substrate would record it as if it were a prediction."""
    r = classify_mule({})
    assert r["role"] == "undetermined" and r["confidence"] == 0.0


def test_one_directional_evidence_refuses_to_return_a_verdict():
    """THE rule this module exists to enforce, arrived at by getting it wrong twice.

    High fan-in alone points only at guilt. Reading a role from it produced 99% "witting" on
    real data, then 68% after a fix, and both numbers described our own data rather than any
    person. A heuristic that cannot see exculpatory evidence must not conclude."""
    class _S:
        def recipient_sender_labels(self, _): return [(f"user:v{i}", 1) for i in range(12)]
        def decisions_for_entity(self, _, limit=200): return []
    r = B.observable_role(_S(), "user:u1", "recipient:r1")
    assert r["role"] == "undetermined", "a one-directional read returned a verdict"
    assert r["evidence_basis"] == "one_directional_refused"
    assert "many_victim_sources" in r["tells_used"], "it should still say what it saw"
    assert r["why"]


def test_a_verdict_is_allowed_once_exculpatory_evidence_can_fire():
    """The refusal is about the SHAPE of available evidence, not a blanket ban: when a tell
    that could exonerate is in play, the heuristic is allowed to read the balance."""
    class _S:
        def recipient_sender_labels(self, _): return []
        def decisions_for_entity(self, _, limit=200):
            return [D(ts="2026-01-01", action="ALLOW"), D(ts="2026-01-02", action="HOLD")]
    r = B.observable_role(_S(), "user:u1", "recipient:r1")
    assert r["evidence_basis"] == "observable_only"
    assert r["role"] == "unwitting"


def test_a_broken_store_does_not_take_the_score_path_down():
    """This runs inside scoring. A substrate query that fails must cost a training label, never
    a decision."""
    class _Boom:
        def recipient_sender_labels(self, _): raise RuntimeError("db gone")
        def decisions_for_entity(self, _, limit=200): raise RuntimeError("db gone")
    assert B.behavioural_tells(_Boom(), "user:u1", "recipient:r1") == {}


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
