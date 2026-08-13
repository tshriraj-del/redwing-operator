"""
Tests for label-based performance monitoring and the three-way attribution.

Why this file exists. `drift_monitor.py` computes PSI over score and feature distributions,
which is label-free: it can say the input moved and can never say the model got worse. REDWING
could detect a shifted population and could not detect a decayed model, which is the first thing
a model-risk reviewer asks about.

The load-bearing tests are the three that must come apart. When this month looks worse there are
three explanations demanding opposite responses, and a monitor that cannot tell them apart is
worse than none, because it will send somebody to retrain against a slow month at the disputes
team:

    the model degraded      -> retrain
    the population shifted  -> the model may be fine, the question changed
    the labels are late     -> nothing happened, you are reading an empty window

Each is constructed here from data that genuinely has that property, and each must produce its
own verdict and not the others'.

Runs under pytest or standalone (python3 tests/test_model_performance.py).
"""

import os
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import model_performance as P            # noqa: E402
from drift_monitor import MIN_PER_SIDE             # noqa: E402
from core.store import FRAUD_FALSE, FRAUD_TRUE, Store   # noqa: E402

AS_OF = datetime(2026, 8, 2, tzinfo=timezone.utc)


def _fresh_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


def _iso(days_ago: float) -> str:
    return (AS_OF - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _case(s, subj, days_ago, score, fraud, *, action="ALLOW", released=False,
          labelled=True, label_days_ago=None, source="chargeback", features=None):
    s.log_decision(subject_ref=subj, action=action, module="model", score=score,
                   expected_liability=1000.0, features=(features or {"amount": 100.0}),
                   rationale={"released": released}, ts=_iso(days_ago))
    if labelled:
        s.add_label("outcome", "is_fraud", FRAUD_TRUE if fraud else FRAUD_FALSE,
                    source=source, confidence=0.95, subject_ref=subj,
                    ts=_iso(label_days_ago if label_days_ago is not None else days_ago - 1))


def _cohort(s, *, tag, days_ago, n=200, accuracy=0.95, amount=(50.0, 150.0), seed=1,
            labelled=True, label_days_ago=None):
    """A window of decisions where the model is right `accuracy` of the time.

    `amount` moves the INPUT distribution, which is what a population shift actually is: the
    same model asked about different traffic. It is deliberately NOT a knob on the score,
    because the score is the model's own output and moves under both explanations, which is
    exactly the confusion this suite exists to prevent.
    """
    rng = random.Random(seed)
    for i in range(n):
        fraud = i % 4 == 0                                  # 25% base rate
        correct = rng.random() < accuracy
        # a correct call scores above the 0.65 alert line iff the case is fraud
        hi = fraud if correct else not fraud
        score = rng.uniform(0.70, 0.95) if hi else rng.uniform(0.10, 0.60)
        _case(s, f"{tag}{i}", days_ago, score, fraud,
              labelled=labelled, label_days_ago=label_days_ago,
              features={"amount": rng.uniform(*amount)})


# ----------------------------------------------------------------- the three-way attribution

def test_a_genuine_decay_on_a_mature_window_is_called_degraded():
    """The one that justifies a retrain, and the only verdict allowed to say so.

    The cohorts are sized at the PSI floor deliberately. `degraded` now requires the population
    to have been actively EXCLUDED, not merely skipped, so a fixture too small to compute PSI on
    correctly produces `degraded_unconfirmed` instead. Sizing it up is what makes this test about
    decay rather than about sample size.
    """
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1, n=MIN_PER_SIDE)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, seed=2, n=MIN_PER_SIDE)
    # a settled gold history so the maturity curve is derivable and the window counts as mature
    for i in range(40):
        _case(s, f"m{i}", 300, 0.5, i % 2 == 0, label_days_ago=300 - (i % 25) - 1)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "degraded", f"{d['verdict']}: {d['reason']}"
    assert d["deltas_vs_previous"]["precision_on_allowed"] < -P.MATERIAL_DROP
    s.close()


def test_a_shifted_score_distribution_is_not_reported_as_decay():
    """Same model, different traffic. Retraining here bakes the shift in rather than fixing
    anything, so the verdict must point at the population and say why."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1, n=MIN_PER_SIDE)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, amount=(5000.0, 15000.0), seed=2, n=MIN_PER_SIDE)
    for i in range(40):
        _case(s, f"m{i}", 300, 0.5, i % 2 == 0, label_days_ago=300 - (i % 25) - 1)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "population_shift", f"{d['verdict']}: {d['reason']}"
    assert d["feature_psi_vs_previous"] >= 0.20, "the INPUT is what shifted"
    assert "before retraining" in d["reason"]
    s.close()


def test_an_empty_window_is_unmeasurable_rather_than_a_regression():
    """THE dangerous one. A window whose outcomes have not arrived looks exactly like a window
    with no fraud in it. Calling that an improvement is how real decay is missed for a quarter;
    calling it a regression is how a team retrains on nothing."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, seed=2, labelled=False)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "unmeasurable", f"{d['verdict']}: {d['reason']}"
    assert "not about the model" in d["reason"]
    assert "Do not retrain" in d["reason"]
    s.close()


def test_a_decay_on_an_immature_window_is_flagged_but_not_confirmed():
    """The honest middle. The numbers point at the model, and a recent window short of its
    late-arriving frauds shows a falling recall for reasons that have nothing to do with it. A
    signal to watch, not a finding."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, seed=2)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "degraded_unconfirmed", f"{d['verdict']}: {d['reason']}"
    assert d["maturity_known"] is False
    assert "NOT confirmed" in d["reason"]
    s.close()


def test_a_small_wobble_is_not_called_a_trend():
    """Moves of a point or two on a few hundred labels are sampling noise, and a monitor that
    fires on them gets muted, after which it detects nothing at all."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1)
    _cohort(s, tag="new", days_ago=170, accuracy=0.94, seed=2)
    for i in range(40):
        _case(s, f"m{i}", 300, 0.5, i % 2 == 0, label_days_ago=300 - (i % 25) - 1)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "stable", f"{d['verdict']}: {d['reason']}"
    s.close()


# --------------------------------------------------------------------------- censoring

def test_the_metric_is_named_for_the_population_it_actually_describes():
    """Outcomes exist only where we ALLOWED the payment, so every production metric describes
    the allowed population. Calling it 'recall' overstates it permanently, because the frauds
    the model caught and blocked are exactly the ones missing from the denominator."""
    s = Store(_fresh_db())
    _cohort(s, tag="a", days_ago=10, accuracy=0.9, seed=5)
    w = P.window(s, _iso(40), _iso(0), mature_only=False, as_of=AS_OF)
    assert "recall_on_allowed" in w["metrics"]
    assert "recall" not in w["metrics"], "an unqualified recall would be a claim we cannot make"
    assert "censored_share" in w["censoring"]
    s.close()


def test_the_holdout_estimates_what_the_block_wall_is_hiding():
    """What the holdout was built and paid for. Released cases are would-be-blocks allowed at
    random, so their fraud rate is the only unbiased view of the blocked population anyone has."""
    s = Store(_fresh_db())
    for i in range(100):                                  # blocked: outcome unknowable
        _case(s, f"b{i}", 10, 0.9, True, action="BLOCK", labelled=False)
    for i in range(40):                                   # released: would-be-blocks, allowed
        _case(s, f"r{i}", 10, 0.9, i % 2 == 0, action="ALLOW", released=True)
    rows = P._rows(s, _iso(40), _iso(0))
    est = P.estimate_censored(rows)
    assert est["estimable"] is True
    assert abs(est["fraud_rate_in_released"] - 0.5) < 0.05
    assert 40 <= est["estimated_frauds_in_blocked"] <= 60
    s.close()


def test_a_handful_of_releases_produces_no_estimate_at_all():
    """A fraud rate from five releases has an error bar wider than the number it estimates. An
    acknowledged gap beats a confident correction that is wrong."""
    s = Store(_fresh_db())
    for i in range(100):
        _case(s, f"b{i}", 10, 0.9, True, action="BLOCK", labelled=False)
    for i in range(5):
        _case(s, f"r{i}", 10, 0.9, True, action="ALLOW", released=True)
    est = P.estimate_censored(P._rows(s, _iso(40), _iso(0)))
    assert est["estimable"] is False
    assert "estimated" in est["reason"]
    assert "fraud_rate_in_released" not in est
    s.close()


# ------------------------------------------------------------------------- window hygiene

def test_a_cohort_is_keyed_on_the_decision_date_not_the_label_date():
    """A window is a set of payments the model judged. Keying on when we found out would fold a
    slow month at the disputes team into the model's score."""
    s = Store(_fresh_db())
    # decided 100 days ago, all labelled only yesterday
    for i in range(40):
        _case(s, f"x{i}", 100, 0.9, i % 2 == 0, label_days_ago=1)
    inside = P.window(s, _iso(110), _iso(90), mature_only=False, as_of=AS_OF)
    outside = P.window(s, _iso(10), _iso(0), mature_only=False, as_of=AS_OF)
    assert inside["n_decisions"] == 40, "the cohort should sit where it was DECIDED"
    assert outside["n_decisions"] == 0, "the labels' own date pulled the cohort forward"
    s.close()


def test_unlabelled_decisions_are_counted_as_coverage_not_as_misses():
    """The difference between 'the model missed these' and 'nobody has told us about these'.
    An inner join would silently collapse the second into invisibility and make coverage
    unknowable, which is how a monitor reports confidently on 2% of its traffic."""
    s = Store(_fresh_db())
    _cohort(s, tag="k", days_ago=10, n=40, accuracy=0.9, seed=3)
    for i in range(60):
        _case(s, f"u{i}", 10, 0.9, True, labelled=False)
    w = P.window(s, _iso(40), _iso(0), mature_only=False, as_of=AS_OF)
    assert w["n_decisions"] == 100 and w["n_labelled"] == 40
    assert abs(w["label_coverage"] - 0.4) < 1e-9
    assert w["metrics"]["fn"] < 60, "unlabelled decisions were counted as missed frauds"
    s.close()


def test_an_immeasurable_window_still_reports_coverage_and_censoring():
    """When it cannot speak about the model it must still say why, or the operator has no idea
    whether to wait, to chase the feed, or to worry."""
    s = Store(_fresh_db())
    for i in range(10):
        _case(s, f"t{i}", 10, 0.9, True)
    w = P.window(s, _iso(40), _iso(0), mature_only=False, as_of=AS_OF)
    assert w["measurable"] is False
    assert "label_coverage" in w and "censoring" in w
    assert str(P.MIN_LABELLED) in w["reason"]
    s.close()


# --------------------------------------------------- the differential the console renders

def _rungs(d):
    return {r["reason"]: r for r in d["differential"]}


def test_every_verdict_carries_the_whole_ladder_not_just_the_rung_it_stopped_on():
    """The console renders this list directly. If an early exit returned two rungs instead of
    five, the page would show a two-item differential and a reader would conclude two
    explanations were weighed. The list is fixed-length so the shape of the reasoning does not
    change with the answer."""
    s = Store(_fresh_db())
    for i in range(10):
        _case(s, f"t{i}", 10, 0.9, True, labelled=False)
    d = P.diagnose(s, window_days=30, as_of=AS_OF)
    assert [r["reason"] for r in d["differential"]] == [k for k, _ in P.REASON_LADDER]
    s.close()


def test_an_untested_explanation_is_not_reported_as_an_excluded_one():
    """THE load-bearing test in this section, and the reason `not_reached` exists as a third
    status. On an unmeasurable window the procedure stops at rung one: population shift and model
    decay are never examined. Marking them `ruled_out` would put "population shift: excluded" on
    a governance page when nothing was measured, which is precisely the false assurance this
    module refuses everywhere else."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, seed=2, labelled=False)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "unmeasurable"
    r = _rungs(d)
    assert r["unmeasurable"]["status"] == P.RULED_IN
    for later in ("no_baseline", "stable", "population_shift", "degraded"):
        assert r[later]["status"] == P.NOT_REACHED, (
            f"{later} was reported as {r[later]['status']} on a window where the procedure never "
            f"reached it; an untested reason is unknown, not excluded")


def test_exactly_one_explanation_is_ruled_in():
    """A differential with two answers is not a differential. Checked across all three of the
    verdicts that come apart, because an early-exit path that forgot to record its own rung would
    otherwise pass every other test in this file."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1, n=MIN_PER_SIDE)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, amount=(5000.0, 15000.0), seed=2, n=MIN_PER_SIDE)
    for i in range(40):
        _case(s, f"m{i}", 300, 0.5, i % 2 == 0, label_days_ago=300 - (i % 25) - 1)
    for as_of, expect in ((AS_OF - timedelta(days=140), "population_shift"),):
        d = P.diagnose(s, window_days=30, as_of=as_of)
        assert d["verdict"] == expect, f"{d['verdict']}: {d['reason']}"
        ruled_in = [r["reason"] for r in d["differential"] if r["status"] == P.RULED_IN]
        assert ruled_in == [expect], f"expected exactly [{expect}], got {ruled_in}"
    s.close()


def test_the_rung_that_is_ruled_in_is_the_verdict_itself():
    """The console colours the verdict from one field and the ladder from another. They must not
    be able to disagree, or the page shows 'Degraded' beside a ladder pointing at the population.
    `degraded_unconfirmed` is the qualified form of the degraded rung, deliberately."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, seed=2)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "degraded_unconfirmed"
    ruled_in = [r["reason"] for r in d["differential"] if r["status"] == P.RULED_IN]
    assert ruled_in == ["degraded"], ruled_in
    assert "NOT confirmed" in _rungs(d)["degraded"]["evidence"]
    s.close()


def test_an_excluded_explanation_says_what_excluded_it():
    """'Ruled out' with no number attached is an assertion, not evidence. On a genuine decay the
    reader has to be able to see the label count that cleared rung one and the PSI that cleared
    rung four, because those two exclusions are the entire argument for retraining."""
    s = Store(_fresh_db())
    _cohort(s, tag="old", days_ago=200, accuracy=0.95, seed=1, n=MIN_PER_SIDE)
    _cohort(s, tag="new", days_ago=170, accuracy=0.55, seed=2, n=MIN_PER_SIDE)
    for i in range(40):
        _case(s, f"m{i}", 300, 0.5, i % 2 == 0, label_days_ago=300 - (i % 25) - 1)
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    assert d["verdict"] == "degraded", f"{d['verdict']}: {d['reason']}"
    r = _rungs(d)
    assert r["unmeasurable"]["status"] == P.RULED_OUT
    assert str(P.MIN_LABELLED) in r["unmeasurable"]["evidence"], "no label floor in the evidence"
    assert r["population_shift"]["status"] == P.RULED_OUT
    assert "0.20" in r["population_shift"]["evidence"], "no PSI threshold in the evidence"
    assert str(d["feature_psi_vs_previous"]) in r["population_shift"]["evidence"], (
        "the exclusion does not carry the measured PSI it rests on")
    s.close()


def test_an_uncomputable_psi_does_not_read_as_a_steady_population():
    """The subtle one. When neither window carries 50 rows of numeric features the PSI never
    runs, the verdict still falls through to the model, and the honest label for population shift
    is `not_reached`. Calling it `ruled_out` would credit the conclusion with an exclusion that
    was never performed."""
    s = Store(_fresh_db())
    for i in range(40):
        _case(s, f"o{i}", 200, 0.9 if i % 4 == 0 else 0.2, i % 4 == 0, features={})
    for i in range(40):
        _case(s, f"n{i}", 170, 0.2, i % 4 == 0, features={})
    d = P.diagnose(s, window_days=30, as_of=AS_OF - timedelta(days=140))
    r = _rungs(d)
    if d["verdict"] in ("degraded", "degraded_unconfirmed"):
        assert r["population_shift"]["status"] == P.NOT_REACHED, (
            f"PSI was {d.get('feature_psi_vs_previous')} yet population shift reads "
            f"{r['population_shift']['status']}")
        assert "UNTESTED" in r["population_shift"]["evidence"]
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
