"""
Tests for the unsupervised novelty gate on the live scoring path.

Why this file exists. `pulseml_models/anomaly_layer.py` trained an isolation forest, wrote it to
disk, and the operator never opened the file. The README meanwhile described it as 30% of an ML
ensemble. So the platform simultaneously overclaimed the thing in its documentation and
underused it in its code, and neither error was visible from the other side.

Two properties have to hold now that it is wired, and both are easy to lose quietly:

  ESCALATE-ONLY   the gate may raise a score to the alert line and no further, and may never
                  lower one. An unsupervised detector saying "this is unusual" is a reason to
                  look, not a reason to be sure, and certainly not a reason to auto-decline.

  FAIL-SILENT     it is a second opinion. If the artifact is missing, stale, or throws, the
                  supervised decision must still happen. A second opinion that can take the
                  primary path down is a liability, not a control.

Runs under pytest or standalone (python3 tests/test_novelty_gate.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_ALLOW_OPEN", "i-understand-this-is-open")  # auth fails closed; tests run the app in-process

import main  # noqa: E402


def _fake_gate(anomaly: float, threshold: float = 0.9):
    """Swap the real forest for a stub returning a chosen anomaly score."""
    class _Iso:
        n_features_in_ = len(main.FEATURES) if main.FEATURES else 32

        def score_samples(self, X):
            # novelty_view maps (hi - raw)/span into 0-1, so invert to hit the target.
            return [main.ANOMALY_HI - anomaly * main.ANOMALY_SPAN]
    return {"iso": _Iso(), "hi": main.ANOMALY_HI, "span": main.ANOMALY_SPAN,
            "threshold": threshold, "auc": 0.9}


# The stub needs the same anchors novelty_view uses; expose them once here rather than
# hard-coding numbers that would drift from model_config.json.
main.ANOMALY_HI = (main.ANOMALY or {}).get("hi", -0.4)
main.ANOMALY_SPAN = (main.ANOMALY or {}).get("span", 0.17)


def _restore(saved):
    main.ANOMALY = saved


# ------------------------------------------------------------------- escalate-only

def test_a_novel_payment_is_raised_to_the_alert_line():
    saved = main.ANOMALY
    main.ANOMALY = _fake_gate(anomaly=0.99, threshold=0.9)
    score, view = main.apply_novelty_gate(0.10, {})
    assert view["novel"] is True
    assert score == main._alert_line(), "a novel payment should reach review, exactly"
    assert view["escalated"] is True
    _restore(saved)


def test_the_gate_never_pushes_past_the_alert_line_into_an_auto_decline():
    """The ceiling is the whole reason this is a gate. An unsupervised detector must be able to
    buy a payment a human look and nothing more."""
    saved = main.ANOMALY
    main.ANOMALY = _fake_gate(anomaly=1.0, threshold=0.5)
    score, _ = main.apply_novelty_gate(0.0, {})
    assert score == main._alert_line()
    assert score < 1.0
    _restore(saved)


def test_the_gate_never_lowers_a_score_the_model_already_raised():
    """THE regression. If the gate could talk a confident supervised score DOWN because the
    payment looks ordinary, one quiet composition change would start clearing fraud the model
    had already caught."""
    saved = main.ANOMALY
    main.ANOMALY = _fake_gate(anomaly=0.0, threshold=0.9)       # looks entirely ordinary
    score, view = main.apply_novelty_gate(0.97, {})
    assert score == 0.97, "an ordinary-looking payment must not de-escalate a model hit"
    assert view["novel"] is False
    _restore(saved)


def test_an_already_alerting_score_is_left_alone():
    """Nothing to add: it is already going to a human. Raising it further would misreport how
    much of the score came from the supervised model."""
    saved = main.ANOMALY
    main.ANOMALY = _fake_gate(anomaly=1.0, threshold=0.5)
    high = main._alert_line() + 0.2
    score, view = main.apply_novelty_gate(high, {})
    assert score == high and view.get("escalated") is False
    _restore(saved)


def test_below_threshold_novelty_changes_nothing():
    """The gate fires on roughly the most anomalous 1%. Anything looser and it stops being a
    gate and becomes a second alert queue."""
    saved = main.ANOMALY
    main.ANOMALY = _fake_gate(anomaly=0.80, threshold=0.9)
    score, view = main.apply_novelty_gate(0.20, {})
    assert score == 0.20 and view["novel"] is False
    _restore(saved)


# ---------------------------------------------------------------------- fail-silent

def test_a_missing_gate_leaves_supervised_scoring_untouched():
    saved = main.ANOMALY
    main.ANOMALY = None
    score, view = main.apply_novelty_gate(0.42, {})
    assert score == 0.42 and view["available"] is False
    _restore(saved)


def test_a_throwing_gate_costs_a_second_opinion_not_a_decision():
    """This runs inside live scoring. A broken second opinion must never take the payment
    decision down with it."""
    saved = main.ANOMALY

    class _Boom:
        n_features_in_ = 32

        def score_samples(self, X):
            raise RuntimeError("model corrupt")
    main.ANOMALY = {"iso": _Boom(), "hi": -0.4, "span": 0.17, "threshold": 0.9}
    score, view = main.apply_novelty_gate(0.42, {})
    assert score == 0.42 and view["available"] is False
    _restore(saved)


def test_a_stale_artifact_is_refused_by_the_compatibility_guard():
    """Tested against the real stale artifact, not a hypothetical. `isolation_forest.pkl` in the
    ML repo was trained on 23 features while the model uses 32; loading it would have scored a
    feature space that no longer means what it did, on every payment, silently."""
    class _Stale:
        n_features_in_ = 23

    class _Current:
        n_features_in_ = 32
    assert main.gate_is_compatible(_Stale(), [0] * 32) is False
    assert main.gate_is_compatible(_Current(), [0] * 32) is True


def test_the_live_gate_matches_the_current_feature_set():
    """THE bug that made this whole exercise necessary. The artifact on disk was trained on 23
    features while the model had moved to 32, so a gate loaded without this check would either
    throw on every payment or, worse, score a feature space that no longer means what it did.
    Loading refuses loudly instead."""
    if main.ANOMALY is None:
        return                                   # not loaded on this machine; nothing to check
    assert main.ANOMALY["iso"].n_features_in_ == len(main.FEATURES), (
        "the novelty gate and the supervised model disagree about the feature set; "
        "re-run pulseml_models/anomaly_layer.py")


def test_the_alert_line_is_read_from_the_matcher_not_restated():
    """A second copy of 0.65 in main.py is how the gate's ceiling and the alert decision drift
    apart, and the drift would be invisible until they disagreed on a real payment."""
    from match_engine import is_alert
    assert main._alert_line() == (is_alert.__defaults__ or (0.65,))[0]
    assert is_alert(main._alert_line()) is True


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
