"""
ADWIN-inspired concept drift monitor for the RedWing scoring pipeline.

Tracks Population Stability Index (PSI) on model score distributions and
key feature distributions. Flags when incoming traffic drifts from the
historical baseline - indicating concept drift, data pipeline shift, or
an adversarial probing campaign.

PSI interpretation (industry standard):
  < 0.10  - stable
  0.10-0.20 - warning (monitor closely)
  > 0.20  - drift (consider retraining)

Reference: ADWIN (Bifet & Gavalda, 2007); PSI as used in SR 11-7 / Fed guidance.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from threading import Lock

# -- Configuration -------------------------------------------------------------

BUFFER_SIZE = 2000   # rolling window size (transactions)
CHECK_EVERY = 50     # check every N new samples
PSI_WARNING = 0.10
PSI_DRIFT   = 0.20
N_BINS      = 10

# Minimum samples PER SIDE of the comparison before PSI means anything.
#
# MEASURED, not chosen. PSI over too few samples measures the sampler, not the population. The
# null distribution was built by drawing one population, splitting it at random, and scoring the
# two halves against each other 400 times per size: any PSI there is pure binning noise, because
# nothing changed. With N_BINS = 10:
#
#     n/side   median PSI    reads "warning"    reads "drift"
#         30       1.3535            100.0%           99.0%
#         50       0.5702             99.0%           90.8%
#        100       0.2227             91.2%           56.2%
#        200       0.1017             50.7%           12.2%
#        300       0.0678             21.2%            3.2%
#        500       0.0371              4.2%            0.0%
#       1000       0.0174              0.5%            0.0%
#
# The previous settings sat in the unusable band: `_compute_psi` floored at 30 per side (100%
# false drift), warmup began at 200 samples which is 100 per side (56% false drift), and the
# per-feature gate was 60 samples, or 30 per side. The monitor was a coin flip at its own
# operating point, and it was found by restarting the operator and replaying a steady stream:
# it reported `drift` with a score PSI of 0.33 on data drawn from one distribution.
#
# 500 per side is the first size where a false DRIFT does not occur in 400 trials. The residual
# 4.2% false-warning rate is accepted: warning is "look at this", drift is "consider retraining".
MIN_PER_SIDE = 500
WARMUP_MIN   = 2 * MIN_PER_SIDE   # the buffer is split in half, so this is the binding number

# Features tracked alongside the model score
TRACKED_FEATURES = [
    "amount_zscore",
    "velocity_1h",
    "rail_risk",
    "recipient_familiarity",
    "device_familiarity",
]

# -- State ---------------------------------------------------------------------

_lock = Lock()
_score_buf: deque               = deque(maxlen=BUFFER_SIZE)
_feature_bufs: dict             = {f: deque(maxlen=BUFFER_SIZE) for f in TRACKED_FEATURES}
_since_last_check: int          = 0
_drift_events: list             = []

_status: dict = {
    "state":         "warming_up",   # warming_up | stable | warning | drift
    "score_psi":     None,
    "feature_psi":   {f: None for f in TRACKED_FEATURES},
    "samples":       0,
    "last_checked":  None,
    "drift_events":  [],
    "baseline_size": 0,
    "current_size":  0,
}

# -- PSI core ------------------------------------------------------------------

def _compute_psi(reference: list, current: list) -> float | None:
    """PSI between two samples, or None when there is not enough of either to say.

    RETURNS None, NOT 0.0, when the floor is not met. A zero is a measurement: it says the two
    distributions match. Returning it for "I could not tell" is the confident-zero failure, and
    here it is the expensive direction, because zero reads as `stable` and a monitor that reports
    stable on no data is worse than one that reports nothing at all.
    """
    if len(reference) < MIN_PER_SIDE or len(current) < MIN_PER_SIDE:
        return None

    lo, hi = min(reference), max(reference)
    if lo == hi:
        return 0.0

    width = (hi - lo) / N_BINS
    edges = [lo + i * width for i in range(N_BINS + 1)]
    edges[-1] += 1e-9  # ensure max value falls in last bin

    def bin_dist(data: list) -> list:
        counts = [0] * N_BINS
        n = len(data)
        for x in data:
            i = min(int((x - lo) / width), N_BINS - 1)
            counts[i] += 1
        return [c / n for c in counts]

    p_ref = bin_dist(reference)
    p_cur = bin_dist(current)

    psi = 0.0
    for pr, pc in zip(p_ref, p_cur):
        pr = max(pr, 1e-6)
        pc = max(pc, 1e-6)
        psi += (pc - pr) * math.log(pc / pr)

    return round(abs(psi), 4)


def _classify(score_psi: float | None, feat_psi: dict) -> str:
    """Worst PSI across score and features decides the state; None values do not vote.

    A feature whose buffer has not filled is UNKNOWN, and folding it in as a zero would let an
    unmeasured signal pull the state toward stable. When nothing at all could be computed the
    answer is `warming_up`, which is the only honest state for a monitor with no measurement.
    """
    known = [v for v in [score_psi, *feat_psi.values()] if v is not None]
    if not known:
        return "warming_up"
    worst = max(known)
    if worst >= PSI_DRIFT:
        return "drift"
    if worst >= PSI_WARNING:
        return "warning"
    return "stable"


def _check() -> None:
    buf = list(_score_buf)
    n   = len(buf)
    if n < WARMUP_MIN:
        return

    mid       = n // 2
    reference = buf[:mid]
    current   = buf[mid:]

    score_psi = _compute_psi(reference, current)

    # A feature buffer fills only when that feature is actually present on incoming rows, so
    # these run ahead of and behind each other and behind the score buffer. `_compute_psi`
    # returns None for the ones that are still short, and `_classify` does not let them vote.
    feat_psi = {}
    for f, dbuf in _feature_bufs.items():
        arr = list(dbuf)
        m = len(arr) // 2
        feat_psi[f] = _compute_psi(arr[:m], arr[m:])

    new_state = _classify(score_psi, feat_psi)
    now       = datetime.utcnow().isoformat() + "Z"

    # EVERY state change is recorded, not only the ones leaving `stable`.
    #
    # The condition here used to be `new_state in ("warning","drift") and state == "stable"`,
    # which silently dropped three transitions a reviewer actually needs:
    #
    #   warming_up -> drift   a monitor that came up already drifting logged nothing at all
    #   warning    -> drift   the ESCALATION, the single most important entry in the history
    #   * -> stable           recovery, so a timeline could only ever show things getting worse
    #
    # Found by restarting the operator and replaying the ledger: the state read `drift` with a
    # rail_risk PSI of 0.54 and the event history was empty. A history that is silently partial
    # is worse than no history, because it is read as complete.
    old_state = _status["state"]
    if new_state != old_state:
        top_feat = max(feat_psi, key=feat_psi.get) if feat_psi else None
        _drift_events.append({
            "timestamp":   now,
            "state":       new_state,
            "from_state":  old_state,
            "score_psi":   score_psi,
            "top_feature": top_feat,
            "top_feat_psi": feat_psi.get(top_feat, 0.0) if top_feat else 0.0,
        })
        if len(_drift_events) > 20:
            _drift_events.pop(0)

    _status.update({
        "state":         new_state,
        "score_psi":     score_psi,
        "feature_psi":   feat_psi,
        "last_checked":  now,
        "baseline_size": mid,
        "current_size":  n - mid,
        "drift_events":  list(_drift_events[-10:]),
    })

# -- Public API ----------------------------------------------------------------

def record(score: float, features: dict | None = None) -> None:
    """Record a scored transaction. Called from build_event() in main.py."""
    global _since_last_check
    with _lock:
        _score_buf.append(float(score))
        if features:
            for f in TRACKED_FEATURES:
                v = features.get(f)
                if v is not None:
                    _feature_bufs[f].append(float(v))
        _status["samples"] += 1
        _since_last_check  += 1
        if _status["samples"] >= WARMUP_MIN and _since_last_check >= CHECK_EVERY:
            _check()
            _since_last_check = 0


def get_status() -> dict:
    with _lock:
        return dict(_status)


def reset() -> None:
    """Clear all buffers - call after retraining the model."""
    global _since_last_check, _drift_events
    with _lock:
        _score_buf.clear()
        for buf in _feature_bufs.values():
            buf.clear()
        _drift_events = []
        _since_last_check = 0
        _status.update({
            "state":         "warming_up",
            "score_psi":     None,
            "feature_psi":   {f: None for f in TRACKED_FEATURES},
            "samples":       0,
            "last_checked":  None,
            "drift_events":  [],
            "baseline_size": 0,
            "current_size":  0,
        })
