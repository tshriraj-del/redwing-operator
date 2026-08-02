"""
Tests for label maturity: how complete a window's labels are, and what may be trained on.

Why this file exists. Fraud labels arrive late and the lateness is not random: the scams that
do the damage on an irrevocable rail surface weeks or months after the payment, because the
victim does not know yet. So the recent window is systematically short of its positives, and
every one of those missing labels would have been a fraud. Train there and the model learns
recent traffic is safe. Measure drift there and drift improves while things get worse.

Two mistakes made while building this are pinned here, because both are the kind that pass
review:

  - the first batch detector used an absolute span threshold and MISSED the 378-label backfill
    in this repo's own store, whose decisions happened to span only 14 days
  - the second one caught it, and would also have thrown away a bank's daily chargeback file,
    which has the identical shape and entirely real lag

The second mistake is the more instructive, and the test named for it is the reason the module
reports bulk arrivals rather than excluding them.

Runs under pytest or standalone (python3 tests/test_label_maturity.py).
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import label_maturity as M   # noqa: E402
from core.store import Store           # noqa: E402

# Fixed clock so every assertion below is deterministic.
AS_OF = datetime(2026, 8, 2, tzinfo=timezone.utc)


def _fresh_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


def _iso(days_ago: float) -> str:
    return (AS_OF - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _case(s, subj, decided_days_ago, labeled_days_ago, *, source="analyst",
          value="True", effective_days_ago=None):
    """One decision plus the label that later landed on it."""
    s.log_decision(subject_ref=subj, action="ALLOW", module="model", score=0.5,
                   features={"amount": 100.0}, ts=_iso(decided_days_ago))
    s.add_label("outcome", "is_fraud", value, source=source, confidence=0.9,
                subject_ref=subj, ts=_iso(labeled_days_ago),
                effective_ts=(_iso(effective_days_ago)
                              if effective_days_ago is not None else ""))


def _uniform_lags(s, n=100, decided_days_ago=300, source="analyst"):
    """n gold labels whose lags are exactly 1..n days, so F(d) = d/n and every
    percentile is known in closed form."""
    for i in range(1, n + 1):
        _case(s, f"u{i}", decided_days_ago, decided_days_ago - i, source=source)


# --------------------------------------------------------------------------- the curve

def test_the_floor_is_the_date_by_which_coverage_is_met():
    """The arithmetic. Lags 1..100 give F(d) = d/100, so 90% coverage is reached at day 90
    and the floor sits 90 days before now. Anything decided after that is still collecting."""
    s = Store(_fresh_db())
    _uniform_lags(s, n=100, decided_days_ago=300)
    mf = M.maturity_floor(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    assert mf["known"], mf["reason"]
    assert abs(mf["days_to_coverage"] - 90) < 1e-6, mf["days_to_coverage"]
    assert mf["floor"].startswith((AS_OF - timedelta(days=90)).date().isoformat())
    s.close()


def test_the_machines_own_call_is_never_in_the_denominator():
    """THE load-bearing filter. A heuristic label is written at score time, so its lag is zero
    by construction: it is a prediction, not a report from the world. Let those in and the
    curve collapses toward zero and pronounces every cohort mature the instant it was scored,
    which is the exact false assurance this module exists to deny."""
    s = Store(_fresh_db())
    _uniform_lags(s, n=60, decided_days_ago=300)              # real gold, lags 1..60
    for i in range(400):                                       # machine calls, lag 0
        _case(s, f"h{i}", 300, 300, source="heuristic")

    gold_only = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    everything = M.lag_curve(s, "outcome", "is_fraud", sources=None,
                             as_of=AS_OF, horizon_days=180)
    assert gold_only["lag_days"]["p50"] > 25, "gold lags were diluted"
    assert everything["lag_days"]["p50"] == 0.0, "sanity: machine calls do have zero lag"
    assert everything["days_to_coverage"] < gold_only["days_to_coverage"], (
        "including the machine's own calls did not shorten the curve, so this test is not "
        "measuring the thing it claims to")
    s.close()


def test_a_curve_is_refused_rather_than_fitted_to_too_little():
    """A default curve would license training on the immature window it exists to withhold, so
    too little evidence returns a refusal that names what is missing."""
    s = Store(_fresh_db())
    _uniform_lags(s, n=5, decided_days_ago=300)
    c = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    assert c["derivable"] is False
    assert "curve" not in c and "days_to_coverage" not in c, "a refusal leaked a curve"
    assert str(M.MIN_LABELS_FOR_CURVE) in c["reason"]
    s.close()


def test_unsettled_cohorts_are_excluded_from_the_curve():
    """The circularity that cannot be broken, only declared. A recent cohort is still
    receiving labels, and the ones it is missing are precisely the long-lag ones being
    measured, so including it would bias the curve toward short lags and overstate maturity."""
    s = Store(_fresh_db())
    _uniform_lags(s, n=60, decided_days_ago=300)          # settled
    for i in range(60):                                    # decided last week, 1-day lags
        _case(s, f"r{i}", 7, 6)
    c = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    assert c["settled_cohort_labels"] == 60, c["settled_cohort_labels"]
    assert c["lag_days"]["p50"] > 25, "the unsettled cohort's short lags leaked in"
    s.close()


def test_a_horizon_that_truncates_its_own_tail_says_so():
    """The observable symptom of a horizon set too short: labels arriving past it were never
    seen, so coverage is overstated. Reported rather than silently corrected, because the fix
    is a judgement about the portfolio, not something to infer from the data."""
    s = Store(_fresh_db())
    _uniform_lags(s, n=100, decided_days_ago=300)
    c = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=100)
    assert c["horizon_truncated"] is True
    assert "horizon_warning" in c and "truncated" in c["horizon_warning"]
    s.close()


# ------------------------------------------------------------------ bulk arrival handling

def test_a_bulk_arrival_is_reported_by_its_lag_identity():
    """When one labeled_ts is stamped across a bucket, each row's lag is fixed by its decision
    date alone, so the lag range equals the decision span exactly. That identity is the
    signature, and it needs no tuned threshold: the first version of this used an absolute
    30-day span and sailed straight past the 378-label backfill in this repo's own store,
    whose decisions spanned 14."""
    s = Store(_fresh_db())
    for i in range(40):                       # decided across 40 days, all labelled one day
        _case(s, f"b{i}", 200 + i, 150)
    rows = M.lag_rows(s, "outcome", "is_fraud", sources=list(M.GOLD_SOURCES))
    found = M.single_arrival_buckets(rows)
    assert len(found) == 1, found
    b = found[0]
    assert b["labels"] == 40
    assert abs(b["lag_range_days"] - b["decision_span_days"]) < 0.01, (
        "the identity that defines a single arrival does not hold in the fixture")
    s.close()


def test_a_daily_chargeback_file_is_not_thrown_away():
    """THE design correction, and the reason this module reports bulk arrivals instead of
    excluding them.

    A real bank's best label source is a file that lands every morning. Every row in Tuesday's
    file carries Tuesday, and the decisions inside it span months, so it has exactly the shape
    of a backfill AND entirely real lag: the chargeback genuinely did arrive on Tuesday.
    Nothing in the timestamps can tell those apart, because the difference is in the pipeline
    rather than the distribution. An earlier version excluded this shape and would have
    discarded the single most valuable label supply in the building."""
    s = Store(_fresh_db())
    for day in range(10):                     # ten daily files, 30 chargebacks each
        for i in range(30):
            _case(s, f"cb{day}_{i}", 200 + i, 150 - day, source="chargeback")
    c = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    assert c["derivable"] is True, c.get("reason")
    assert c["n"] == 300, "the daily files were dropped from the curve"
    assert len(c["single_arrival_buckets"]) == 10, "the buckets should still be REPORTED"
    s.close()


def test_effective_ts_makes_write_time_irrelevant():
    """The incentive to populate the column, and the actual answer to bulk arrivals. Import a
    year of chargebacks in one go: written at one instant, but each carrying when it truly
    became a chargeback. The lag is then a property of the world, so the bucket report has
    nothing to say about them and the curve is unaffected by when the job ran."""
    s = Store(_fresh_db())
    for i in range(1, 61):                    # all WRITTEN today, true lag i days
        _case(s, f"e{i}", 300, 0, source="chargeback", effective_days_ago=300 - i)
    rows = M.lag_rows(s, "outcome", "is_fraud", sources=list(M.GOLD_SOURCES))
    assert all(r["lag_basis"] == "effective_ts" for r in rows)
    assert M.single_arrival_buckets(rows) == [], "effective_ts rows should be exempt"
    c = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    assert c["derivable"] and abs(c["lag_days"]["max"] - 60) < 1e-6, c["lag_days"]
    assert c["lag_from_effective_ts"] == 60
    s.close()


# ------------------------------------------------------------------------- what it gates

def test_maturity_is_a_property_of_the_cohort_not_of_the_label():
    """A label written today about a payment made today is not mature. It is one early report
    from a cohort whose other labels have not arrived, and the ones still missing are the
    frauds. So the partition keys on the DECISION date, never on the label's own."""
    # Both were LABELLED today. The only thing separating them is when the payment was
    # decided, which is exactly the axis that must decide maturity: keying on the label's own
    # date would call the old case immature too, purely because someone got to it this morning.
    fresh = {"subject_ref": "a", "decided_ts": _iso(1), "labeled_ts": _iso(0)}
    old = {"subject_ref": "b", "decided_ts": _iso(400), "labeled_ts": _iso(0)}
    p = M.partition([fresh, old], _iso(90))
    assert [r["subject_ref"] for r in p["mature"]] == ["b"], (
        "a long-settled case was called immature because its label happens to be recent")
    assert [r["subject_ref"] for r in p["immature"]] == ["a"]


def test_mature_only_refuses_rather_than_training_on_everything():
    """THE trap. A filter that quietly does nothing when it cannot compute is worse than no
    filter at all, because the caller reads the result as maturity-enforced and it is not."""
    from core.train import train_target
    s = Store(_fresh_db())
    for i in range(80):                       # plenty to train on, but nothing settled
        _case(s, f"n{i}", 2, 1, value=("True" if i % 3 == 0 else "False"))
    r = train_target(s, "outcome", "is_fraud", mature_only=True, as_of=AS_OF)
    assert r["trained"] is False
    assert r["maturity_known"] is False
    assert "silently" in r["reason"], r["reason"]
    s.close()


def test_the_gate_will_not_certify_on_immature_gold():
    """The graduation gate counted gold and pairs and never asked whether the cohort had
    settled. The bar here is the SAME MIN_GOLD rather than a second threshold: the question is
    only whether enough of the gold is old enough to count."""
    from core.graduation import evaluate_target
    from core.loop import record_decision
    s = Store(_fresh_db())

    # a settled cohort, enough to derive a curve with ~30d lags
    for i in range(1, 41):
        _case(s, f"s{i}", 300, 300 - (i % 30) - 1, source="chargeback")

    # 60 recent adjudicated cases, paired with a heuristic that agrees ~80% of the time
    for i in range(60):
        gold = "True" if i % 2 == 0 else "False"
        heur = gold if i >= 12 else ("False" if gold == "True" else "True")
        record_decision(s, f"g{i}", module="model",
                        heuristic_labels=[{"space": "outcome", "key": "is_fraud",
                                           "value": heur, "confidence": 0.3}])
        s.add_label("outcome", "is_fraud", gold, source="analyst", confidence=0.9,
                    subject_ref=f"g{i}")

    rep = evaluate_target(s, "outcome", "is_fraud", as_of=AS_OF)
    assert rep["verdict"] == "immature_evidence", rep["verdict"]
    assert rep["maturity"]["known"] is True
    assert "still arriving" in rep["reason"]
    assert "do not label harder" in rep["reason"], (
        "the reason must say that the fix is time, or an analyst will be sent to grind out "
        "labels that cannot help")
    s.close()


def test_an_unknown_maturity_does_not_fail_the_gate_but_is_stated_on_it():
    """The balance. An unknown maturity is not evidence of immaturity, so it must not block;
    but a caveat parked in a separate field is a caveat nobody reads, so it rides on the
    verdict's own reason string."""
    from core.graduation import evaluate_target
    from core.loop import record_decision
    s = Store(_fresh_db())
    for i in range(60):
        gold = "survival" if i % 2 == 0 else "organized_malicious"
        heur = gold if i >= 12 else ("organized_malicious" if gold == "survival" else "survival")
        record_decision(s, f"m{i}", module="motive",
                        heuristic_labels=[{"space": "intent", "key": "motive",
                                           "value": heur, "confidence": 0.3}])
        s.add_label("intent", "motive", gold, source="analyst", confidence=0.9,
                    subject_ref=f"m{i}")
    rep = evaluate_target(s, "intent", "motive", as_of=AS_OF)
    assert rep["verdict"] == "ready_to_train", rep["verdict"]
    assert rep["maturity"]["known"] is False
    assert "MATURITY UNKNOWN" in rep["reason"]
    s.close()


def test_a_malformed_timestamp_costs_a_row_not_the_run():
    """These strings arrive from connectors, CSVs and other institutions."""
    assert M._parse("not-a-date") is None
    assert M._parse("") is None
    assert M._parse(None) is None
    assert M._parse("2026-08-02T00:00:00Z").year == 2026
    s = Store(_fresh_db())
    _uniform_lags(s, n=40, decided_days_ago=300)
    s.log_decision(subject_ref="bad", action="ALLOW", features={"a": 1.0}, ts="garbage")
    s.add_label("outcome", "is_fraud", "True", source="analyst", subject_ref="bad")
    c = M.lag_curve(s, "outcome", "is_fraud", as_of=AS_OF, horizon_days=180)
    assert c["derivable"] is True and c["n"] == 40
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
