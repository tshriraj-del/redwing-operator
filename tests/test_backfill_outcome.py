"""
Tests for the outcome-label backfill.

The point of the backfill is to make human judgment usable, so the way it can fail worst is by
destroying some. store.add_label() SUPERSEDES the current label for a target, and the first
version of this script wrote a machine call onto every decision including ones an analyst had
already judged, marking two of the five gold labels in the store superseded. These pin that
shut, and pin the idempotency that makes re-running safe.

Runs under pytest or standalone (python3 tests/test_backfill_outcome.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.store import Store                      # noqa: E402
from core import backfill_outcome_labels as B     # noqa: E402


def _store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd); os.unlink(path)
    return Store(path)


def _decision(s, sub, score):
    s.log_decision(decision_id=f"dec:{sub}", subject_ref=sub, entity_id=f"user:{sub}",
                   action="ALLOW", module="model", score=score)


def test_it_never_supersedes_a_human_label():
    """THE regression. A subject an analyst has already judged is skipped entirely, because
    writing a machine call there marks the human's judgment superseded, which is the opposite
    of what a script for making human judgment usable should do."""
    s = _store()
    _decision(s, "judged", 0.9)
    s.add_label("outcome", "is_fraud", 1, source="analyst", subject_ref="judged",
                annotator="investigator")
    p = B.plan(s)
    assert not any(t["subject_ref"] == "judged" for t in p["to_write"]), (
        "the backfill planned to write over a subject an analyst had already labelled")
    B.apply(s, p)
    rows = s._conn.execute(
        "SELECT superseded_by FROM labels WHERE subject_ref='judged' AND source='analyst'"
    ).fetchall()
    assert all(not r["superseded_by"] for r in rows), "a human label was superseded"


def test_it_writes_the_machine_call_for_unjudged_decisions():
    s = _store()
    _decision(s, "hot", 0.9)     # above the 0.65 alert threshold
    _decision(s, "cold", 0.1)
    B.apply(s, B.plan(s))
    got = {r["subject_ref"]: r["label_value"] for r in s._conn.execute(
        "SELECT subject_ref,label_value FROM labels WHERE source='heuristic'").fetchall()}
    assert str(got.get("hot")) == "1" and str(got.get("cold")) == "0"


def test_it_is_idempotent():
    s = _store()
    _decision(s, "a", 0.9)
    n1 = B.apply(s, B.plan(s))
    n2 = B.apply(s, B.plan(s))
    assert n1 == 1 and n2 == 0, f"re-running wrote {n2} more labels"


def test_backfilled_rows_are_distinguishable_from_live_ones():
    """A backfill reads cascade_score (this bank's own book) while the live path records the
    call after the network view. Two different quantities, so anything computing agreement over
    them has to be able to tell which it is holding."""
    s = _store()
    _decision(s, "a", 0.9)
    B.apply(s, B.plan(s))
    r = s._conn.execute("SELECT annotator FROM labels WHERE source='heuristic'").fetchone()
    assert r["annotator"] == B.ANNOTATOR != "model_score_call"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}"); failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed ({len(tests)} total)")
    sys.exit(1 if failed else 0)
