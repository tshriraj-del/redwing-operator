"""
ADWIN-inspired concept drift monitor for the RedWing scoring pipeline.

Tracks Population Stability Index (PSI) on model score distributions and
key feature distributions. Flags when incoming traffic drifts from the
historical baseline — indicating concept drift, data pipeline shift, or
an adversarial probing campaign.

PSI interpretation (industry standard):
  < 0.10  — stable
  0.10–0.20 — warning (monitor closely)
  > 0.20  — drift (consider retraining)

Reference: ADWIN (Bifet & Gavalda, 2007); PSI as used in SR 11-7 / Fed guidance.
"""

import math
from collections import deque
from datetime import datetime
from threading import Lock

# ── Configuration ─────────────────────────────────────────────────────────────

BUFFER_SIZE = 2000   # rolling window size (transactions)
WARMUP_MIN  = 200    # samples needed before drift checks begin
CHECK_EVERY = 50     # check every N new samples
PSI_WARNING = 0.10
PSI_DRIFT   = 0.20
N_BINS      = 10

# Features tracked alongside the model score
TRACKED_FEATURES = [
    "amount_zscore",
    "velocity_1h",
    "rail_risk",
    "recipient_familiarity",
    "device_familiarity",
]

# ── State ─────────────────────────────────────────────────────────────────────

_lock = Lock()
_score_buf: deque               = deque(maxlen=BUFFER_SIZE)
_feature_bufs: dict             = {f: deque(maxlen=BUFFER_SIZE) for f in TRACKED_FEATURES}
_since_last_check: int          = 0
_drift_events: list             = []

_status: dict = {
    "state":         "warming_up",   # warming_up | stable | warning | drift
    "score_psi":     0.0,
    "feature_psi":   {f: 0.0 for f in TRACKED_FEATURES},
    "samples":       0,
    "last_checked":  None,
    "drift_events":  [],
    "baseline_size": 0,
    "current_size":  0,
}

# ── PSI core ──────────────────────────────────────────────────────────────────

def _compute_psi(reference: list, current: list) -> float:
    if len(reference) < 30 or len(current) < 30:
        return 0.0

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


def _classify(score_psi: float, feat_psi: dict) -> str:
    worst = max(score_psi, max(feat_psi.values(), default=0.0))
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

    feat_psi = {}
    for f, dbuf in _feature_bufs.items():
        arr = list(dbuf)
        if len(arr) >= 60:
            m = len(arr) // 2
            feat_psi[f] = _compute_psi(arr[:m], arr[m:])
        else:
            feat_psi[f] = 0.0

    new_state = _classify(score_psi, feat_psi)
    now       = datetime.utcnow().isoformat() + "Z"

    if new_state in ("warning", "drift") and _status["state"] == "stable":
        top_feat = max(feat_psi, key=feat_psi.get) if feat_psi else None
        _drift_events.append({
            "timestamp":   now,
            "state":       new_state,
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

# ── Public API ────────────────────────────────────────────────────────────────

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
    """Clear all buffers — call after retraining the model."""
    global _since_last_check, _drift_events
    with _lock:
        _score_buf.clear()
        for buf in _feature_bufs.values():
            buf.clear()
        _drift_events = []
        _since_last_check = 0
        _status.update({
            "state":         "warming_up",
            "score_psi":     0.0,
            "feature_psi":   {f: 0.0 for f in TRACKED_FEATURES},
            "samples":       0,
            "last_checked":  None,
            "drift_events":  [],
            "baseline_size": 0,
            "current_size":  0,
        })
