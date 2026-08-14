"""
Tests for the PSI drift monitor, and specifically for its transition history.

WHY THIS FILE EXISTS. `drift_monitor` had no tests, and the gap that surfaced is the kind that
only shows up in use: the operator was restarted, the ledger replayed through /ingest, and the
monitor came up reading `drift` with a rail_risk PSI of 0.54 and an EMPTY event history. The
recording condition was `new_state in ("warning","drift") and state == "stable"`, so a monitor
that never passed through `stable` on its way up recorded nothing.

That matters more than it first looks, because the history is what the console renders. A list
that is silently partial is read as complete, so an empty one says "no drift has ever occurred"
when what happened is "drift occurred by a route this code did not check for". Three transitions
were being dropped, and one of them is the most important entry the monitor can produce:

    warming_up -> drift     came up already drifting
    warning    -> drift     the ESCALATION
    anything   -> stable    recovery, without which a timeline only ever gets worse

PSI itself is the standard estimator and is not re-derived here; what is guarded is the state
machine around it and the arithmetic that must not silently return a confident zero.

Runs under pytest or standalone (python3 tests/test_drift_monitor.py).
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import drift_monitor as D  # noqa: E402


def _feed(n, *, score, feature=None, seed=0):
    """Push n samples. `score` and `feature` are callables of the index, so a test can move the
    distribution partway through a buffer rather than between two buffers."""
    rng = random.Random(seed)
    for i in range(n):
        feats = {f: (feature(i, rng) if feature else 0.5) for f in D.TRACKED_FEATURES}
        D.record(score(i, rng), feats)


def _steady(i, rng):
    return rng.uniform(0.10, 0.30)


def _shifted(i, rng):
    return rng.uniform(0.70, 0.95)


# ------------------------------------------------------------------ the transition history

def test_a_monitor_that_comes_up_already_drifting_records_the_event():
    """THE regression. A restart mid-incident goes warming_up -> drift without ever touching
    `stable`, and the old condition required `stable` as the previous state, so the whole
    incident left no trace in the history the console renders."""
    D.reset()
    # both halves of the buffer differ from the start: never passes through stable
    _feed(D.WARMUP_MIN, score=lambda i, r: _steady(i, r) if i < D.WARMUP_MIN // 2 else _shifted(i, r),
          feature=lambda i, r: 0.1 if i < D.WARMUP_MIN // 2 else 0.9)
    st = D.get_status()
    assert st["state"] in ("warning", "drift"), f"fixture did not drift: {st['state']}"
    assert st["drift_events"], (
        f"state is {st['state']} and the history is empty; a monitor that came up drifting "
        f"recorded nothing")
    assert st["drift_events"][-1]["from_state"] == "warming_up"


def test_an_escalation_from_warning_to_drift_is_recorded():
    """The most consequential entry the monitor can produce, and the old condition dropped it:
    the previous state was `warning`, not `stable`, so the escalation to `drift` was invisible.
    A reviewer reading the history would see the warning and never learn it got worse."""
    D.reset()
    _feed(D.WARMUP_MIN, score=_steady, feature=lambda i, r: 0.5)
    before = len(D.get_status()["drift_events"])
    D._status["state"] = "warning"          # the state the old guard refused to escalate from
    _feed(D.CHECK_EVERY, score=_shifted, feature=lambda i, r: 0.9)
    events = D.get_status()["drift_events"]
    assert len(events) > before, "an escalation out of `warning` recorded nothing"
    assert events[-1]["from_state"] == "warning"


def test_a_recovery_is_recorded_so_the_timeline_can_go_both_ways():
    """Without this the history only ever accumulates bad news, and a panel showing three
    escalations and no recoveries reads as a system that has been broken since the first one."""
    D.reset()
    _feed(D.WARMUP_MIN, score=_steady, feature=lambda i, r: 0.5)
    D._status["state"] = "drift"
    _feed(D.CHECK_EVERY, score=_steady, feature=lambda i, r: 0.5)
    events = D.get_status()["drift_events"]
    assert events, "a return to stable recorded nothing"
    assert events[-1]["state"] == "stable" and events[-1]["from_state"] == "drift"


def test_a_steady_stream_records_no_events_at_all():
    """The other half. A history that fires on every check is noise, and a monitor whose events
    are noise gets muted, which is how the real one gets missed."""
    D.reset()
    _feed(D.WARMUP_MIN * 3, score=_steady, feature=lambda i, r: 0.5)
    st = D.get_status()
    assert st["state"] == "stable", st["state"]
    assert [e for e in st["drift_events"] if e["state"] != "stable"] == []


def test_the_history_is_bounded():
    """It lives in memory for the life of the process. Unbounded growth on a monitor that flaps
    is a slow leak in the one component that is supposed to be watching for problems."""
    D.reset()
    D._drift_events.extend({"timestamp": str(i), "state": "drift", "from_state": "stable",
                            "score_psi": 0.3, "top_feature": None, "top_feat_psi": 0.0}
                           for i in range(20))
    _feed(D.WARMUP_MIN, score=lambda i, r: _steady(i, r) if i < D.WARMUP_MIN // 2 else _shifted(i, r),
          feature=lambda i, r: 0.1 if i < D.WARMUP_MIN // 2 else 0.9)
    assert len(D._drift_events) <= 20, len(D._drift_events)
    assert len(D.get_status()["drift_events"]) <= 10, "the status view is capped at 10"


# ------------------------------------------------------------------ the estimator's floors

def test_too_few_samples_return_no_measurement_rather_than_a_confident_zero():
    """THE second regression, and the more dangerous one. `_compute_psi` used to floor at 30 per
    side and return 0.0 below it, and zero reads as `stable`: a monitor with no data reported
    that the population had not moved. Under the floor the answer is None, which is not a PSI
    value and cannot be classified as anything."""
    assert D._compute_psi([0.1] * 10, [0.9] * 10) is None, "a confident zero on 10 rows a side"
    assert D._compute_psi([0.1] * (D.MIN_PER_SIDE - 1), [0.9] * (D.MIN_PER_SIDE - 1)) is None
    D.reset()
    _feed(D.WARMUP_MIN - 1, score=_shifted, feature=lambda i, r: 0.9)
    assert D.get_status()["state"] == "warming_up", (
        "the monitor left warmup early and would have reported a state from too few samples")


def test_the_sample_floor_is_where_the_null_distribution_stops_firing():
    """The floor is a measured quantity, so it is checked by measurement. One population split at
    random has, by construction, no drift in it; the monitor must not find any. At the old floor
    of 30 per side this fails on better than 99 runs in 100, which is what the constant is for."""
    rng = random.Random(3)
    false_drift = 0
    for _ in range(60):
        pool = [rng.gauss(0, 1) for _ in range(2 * D.MIN_PER_SIDE)]
        rng.shuffle(pool)
        p = D._compute_psi(pool[:D.MIN_PER_SIDE], pool[D.MIN_PER_SIDE:])
        assert p is not None, "the floor should admit a sample of exactly MIN_PER_SIDE"
        false_drift += p >= D.PSI_DRIFT
    assert false_drift == 0, f"{false_drift}/60 splits of ONE population were called drift"


def test_an_unmeasured_feature_does_not_vote_toward_stable():
    """Feature buffers fill at different rates because a feature is only recorded on rows that
    carry it. Folding a short one in as 0.0 lets a signal nobody measured pull the worst-case
    down, which is how a real drift on one feature gets averaged into calm by four unknowns."""
    assert D._classify(None, {"a": None, "b": None}) == "warming_up"
    assert D._classify(None, {"a": 0.31, "b": None}) == "drift", (
        "a known drift was outvoted by unmeasured features")
    assert D._classify(0.01, {"a": None}) == "stable"


def test_an_identical_distribution_scores_zero_and_a_moved_one_does_not():
    """The estimator has to actually discriminate, or every test above passes vacuously."""
    a = [i / 1000 for i in range(D.MIN_PER_SIDE)]
    assert D._compute_psi(a, list(a)) == 0.0
    assert D._compute_psi(a, [x + 0.8 for x in a]) > D.PSI_DRIFT


def test_reset_clears_the_history_as_well_as_the_buffers():
    """reset() is called after a retrain. Carrying the old model's drift events into the new
    model's timeline would attribute one model's incident to another."""
    D.reset()
    _feed(D.WARMUP_MIN, score=lambda i, r: _steady(i, r) if i < D.WARMUP_MIN // 2 else _shifted(i, r),
          feature=lambda i, r: 0.1 if i < D.WARMUP_MIN // 2 else 0.9)
    assert D.get_status()["drift_events"], "fixture produced no events to clear"
    D.reset()
    st = D.get_status()
    assert st["drift_events"] == [] and st["samples"] == 0 and st["state"] == "warming_up"


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
