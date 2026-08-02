"""
Tests for the outcome ledger and for source precedence in the label write path.

Why this file exists. Of the five sources the graduation gate trusts as ground truth, exactly
one had a live production path: `analyst`, a human clicking. The other four appeared in a
constant tuple and nowhere else. So the gold supply was one person's afternoon, and this module
is the path in for the outcomes the business already produces.

The precedence tests are the load-bearing half, and they exist because this repo has already
suffered the failure they prevent: `backfill_outcome_labels.py` wrote machine calls over two of
the five analyst gold labels in the live store and marked the humans superseded. That was fixed
with a skip-guard inside that one script, which protects nothing from the next writer. A nightly
chargeback feed is the next writer, so the rule now lives in `store.add_label()` where every
writer meets it.

Runs under pytest or standalone (python3 tests/test_outcome_ledger.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import outcome_ledger as L                        # noqa: E402
from core.store import FRAUD_FALSE, FRAUD_TRUE, Store       # noqa: E402


def _fresh_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


def _scored(s, subj="txn_1", score=0.8):
    """A decision, so the outcome has point-in-time features to attach to."""
    s.log_decision(subject_ref=subj, action="ALLOW", module="model", score=score,
                   features={"amount": 4200.0})


def _current(s, subj):
    for l in s.current_labels(subject_ref=subj):
        if l.label_space == "outcome" and l.label_key == "is_fraud":
            return l
    return None


# ------------------------------------------------------------------------- precedence

def test_a_weaker_source_does_not_overwrite_a_stronger_one():
    """THE regression, and it is not hypothetical: the backfill did this to two of five analyst
    gold labels in the live store. A skip-guard in one script does not generalise, so the rule
    lives in the write path where every writer meets it."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_TRUE, source="analyst", subject_ref="t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="heuristic", subject_ref="t1")
    cur = _current(s, "t1")
    assert cur.source == "analyst", f"a machine call took over from an analyst ({cur.source})"
    assert cur.label_value == FRAUD_TRUE
    s.close()


def test_the_outranked_label_is_kept_as_evidence_not_discarded():
    """It arrives already superseded rather than being dropped. Two reasons: evidence should
    never be thrown away, and the graduation gate recovers the machine's own prediction from
    exactly this history to pair it against the gold."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_TRUE, source="analyst", subject_ref="t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="heuristic", subject_ref="t1")
    hist = [l for l in s.label_history(subject_ref="t1") if l.label_space == "outcome"]
    assert len(hist) == 2, "the outranked label was dropped instead of recorded"
    loser = [l for l in hist if l.source == "heuristic"][0]
    assert loser.superseded_by, "the loser should arrive already superseded"
    assert "outranked" in loser.notes
    s.close()


def test_a_stronger_source_does_take_over():
    """Precedence is a ranking, not a freeze. A chargeback outranks an analyst."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="analyst", subject_ref="t1")
    r = L.record_outcome(s, {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback"})
    assert r["resolution"] == "accepted"
    assert _current(s, "t1").source == "chargeback"
    s.close()


def test_an_explicit_override_lets_a_weaker_source_win_and_records_why():
    """Real work needs this. An analyst who reviews a chargeback and finds first-party abuse is
    a weaker source correctly overturning a stronger one. Default refuse, explicit override,
    always recorded, so nobody can do it silently."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_TRUE, source="chargeback", subject_ref="t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="analyst", subject_ref="t1",
                override_reason="reviewed: first-party abuse, delivery evidence on file")
    cur = _current(s, "t1")
    assert cur.source == "analyst" and cur.label_value == FRAUD_FALSE
    assert "first-party abuse" in cur.notes and "override of chargeback" in cur.notes
    s.close()


def test_precedence_does_not_disturb_the_ordinary_case():
    """The overwhelmingly common path: machine call at score time, analyst adjudicates later.
    The analyst must still win, and the heuristic must still be recoverable for pairing."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="heuristic", subject_ref="t1")
    s.add_label("outcome", "is_fraud", FRAUD_TRUE, source="analyst", subject_ref="t1")
    assert _current(s, "t1").source == "analyst"
    hist = [l for l in s.label_history(subject_ref="t1") if l.source == "heuristic"]
    assert hist and hist[0].superseded_by, "the heuristic label must remain, superseded"
    s.close()


def test_an_unknown_source_can_neither_be_ignored_nor_bulldoze_an_adjudication():
    """A new integration lands with a source nobody has ranked yet. It must not be silently
    discarded, and it must not outrank a human. It sits just above the machine."""
    from core.store import precedence_of
    assert precedence_of("heuristic") < precedence_of("brand_new_feed")
    assert precedence_of("brand_new_feed") < precedence_of("analyst")


# ------------------------------------------------------------------------ the ledger

def test_an_unreadable_outcome_is_refused_rather_than_called_legitimate():
    """Defaulting an unparseable column to 'not fraud' would bury real losses in the negative
    class, which is the same one-directional error label maturity exists to prevent."""
    assert L.normalise_outcome("banana") is None
    ok, err = L.validate({"subject_ref": "t1", "source": "chargeback", "outcome": "banana"})
    assert ok is None and "unreadable" in err


def test_the_machines_own_call_cannot_enter_through_the_outcome_door():
    """The ledger is for reports from the world. A heuristic arriving here would be the system
    laundering its own prediction into the gold sources."""
    ok, err = L.validate({"subject_ref": "t1", "source": "heuristic", "outcome": "fraud"})
    assert ok is None and "machine" in err


def test_re_reading_yesterdays_file_writes_nothing_new():
    """Outcome files are re-sent, replayed and backfilled constantly. A second read must not
    manufacture a second label, or every re-ingest would look like fresh corroboration."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    rec = {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback",
           "effective_ts": "2026-06-01T00:00:00Z", "reference": "cb_1"}
    assert L.record_outcome(s, rec)["resolution"] == "accepted"
    assert L.record_outcome(s, rec)["resolution"] == "duplicate"
    hist = [l for l in s.label_history(subject_ref="t1") if l.label_space == "outcome"]
    assert len(hist) == 1, f"re-ingest wrote {len(hist)} labels"
    s.close()


def test_a_disagreement_between_two_gold_sources_is_surfaced_as_a_missed_fraud():
    """THE point of the module. The analyst cleared it; a chargeback arrived three weeks later
    saying fraud. That is a labelled false negative found by the world rather than by us, with
    the point-in-time features still attached. Before this it sat in label_history unread."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="analyst", subject_ref="t1")
    L.record_outcome(s, {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback",
                         "effective_ts": "2026-06-21T00:00:00Z"})
    d = L.disagreements(s)
    assert len(d) == 1, d
    assert d[0]["kind"] == "missed_fraud"
    assert d[0]["first"]["source"] == "analyst" and d[0]["current"]["source"] == "chargeback"
    assert d[0]["reversal"] is False, "two different sources disputing is not a reversal"
    s.close()


def test_a_source_reversing_itself_is_named_a_reversal_not_a_dispute():
    """A chargeback represented and won back is one party changing its mind. A model trained
    between the two versions was trained on a label that no longer exists, and nothing else in
    the system would notice."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    L.record_outcome(s, {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback",
                         "effective_ts": "2026-05-01T00:00:00Z", "reference": "cb_1"})
    L.record_outcome(s, {"subject_ref": "t1", "outcome": "legit", "source": "chargeback",
                         "effective_ts": "2026-06-01T00:00:00Z", "reference": "cb_1_rev"})
    rev = L.reversals(s)
    assert len(rev) == 1 and rev[0]["reversal"] is True
    assert rev[0]["kind"] == "false_alarm"
    assert _current(s, "t1").label_value == FRAUD_FALSE, "the reversal should stand"
    s.close()


def test_agreement_between_sources_is_not_reported_as_a_disagreement():
    """Corroboration is the common case and must stay quiet, or the queue fills with noise."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_TRUE, source="analyst", subject_ref="t1")
    L.record_outcome(s, {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback"})
    assert L.disagreements(s) == []
    s.close()


def test_an_outranked_report_still_counts_as_a_disagreement():
    """It lost the vote; it did not stop being evidence that two sources disagree. Reading only
    the winners would hide every dispute a weaker source raised."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_TRUE, source="law_enforcement", subject_ref="t1")
    r = L.record_outcome(s, {"subject_ref": "t1", "outcome": "legit", "source": "victim_report"})
    assert r["resolution"] == "outranked"
    assert r["disagrees_with_previous"] is True
    assert _current(s, "t1").source == "law_enforcement"
    assert len(L.disagreements(s)) == 1
    s.close()


def test_the_outcome_record_survives_alongside_the_label():
    """The label is what we believe; the event is what arrived, with the amount, reason code and
    reference the label has no room for. Losing the second makes the first unauditable."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    L.record_outcome(s, {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback",
                         "reference": "cb_9", "reason_code": "10.4", "amount": 4200.0})
    evs = [e for e in s.recent_events(limit=50) if e.event_type == "outcome"]
    assert len(evs) == 1
    assert evs[0].payload["reason_code"] == "10.4" and evs[0].payload["amount"] == 4200.0
    s.close()


def test_contradicting_the_machine_is_not_counted_as_a_gold_dispute():
    """These were both briefly called 'disagreements', and a single run reported 10 of one and
    0 of the other. Contradicting the machine's own call is the ordinary case and is what the
    gate already measures as kappa. A dispute between two GROUND-TRUTH sources is the rare one
    that needs a human, because there is no opinion available to dismiss."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    s.add_label("outcome", "is_fraud", FRAUD_FALSE, source="heuristic", subject_ref="t1")
    r = L.ingest_outcomes(s, [{"subject_ref": "t1", "outcome": "fraud",
                               "source": "chargeback"}])
    assert len(r["contradicted_standing"]) == 1, "the machine was contradicted, say so"
    assert r["gold_vs_gold_disputes"] == 0, "the machine's call is not a second ground truth"
    assert L.disagreements(s) == [], "gold-only reporting must ignore the heuristic"
    s.close()


def test_a_batch_reports_what_it_could_not_use():
    """A rejected row must be visible. Silent drops in an outcome feed are how a bank discovers
    six months later that a whole reason code never landed."""
    s = Store(_fresh_db())
    _scored(s, "t1")
    r = L.ingest_outcomes(s, [
        {"subject_ref": "t1", "outcome": "fraud", "source": "chargeback"},
        {"subject_ref": "t2", "outcome": "???", "source": "chargeback"},
        {"outcome": "fraud", "source": "chargeback"},
    ])
    assert r["counts"]["accepted"] == 1
    assert r["counts"]["rejected"] == 2
    assert len(r["errors"]) == 2 and all("error" in e for e in r["errors"])
    s.close()


def test_the_simulated_feed_is_labelled_as_simulated():
    """REDWING has no real chargeback file. Everything else here is real code on real input;
    this is the part that pretends, and it must be impossible to mistake downstream."""
    s = Store(_fresh_db())
    for i in range(40):                                    # decided a year ago, so the lags land
        s.log_decision(subject_ref=f"t{i}", action="ALLOW", module="model", score=0.9,
                       features={"amount": 4200.0}, ts="2025-08-02T00:00:00Z")
    feed = L.simulate_outcome_feed(s, n=20, seed=3)
    assert feed, "the simulator produced nothing to check"
    assert all(r["simulated"] is True for r in feed)
    assert all(r["effective_ts"] for r in feed), "records must carry when the fact became true"
    s.close()


def test_the_simulator_will_not_report_an_outcome_that_has_not_happened_yet():
    """A payment made this morning cannot have a chargeback dated three weeks out. Emitting one
    would put future-dated facts in the substrate and hand the maturity curve lags it could
    never observe, which is the exact artefact that module exists to keep out."""
    s = Store(_fresh_db())
    for i in range(40):
        _scored(s, f"t{i}", score=0.9)                     # decided now
    assert L.simulate_outcome_feed(s, n=20, seed=3) == []
    s.close()


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
