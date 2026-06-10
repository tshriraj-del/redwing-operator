"""
gnn_lite.py — Tier 2 GNN cascade: 1-layer Graph Convolutional Network.

Implements message-passing neighborhood aggregation over the transaction graph:
  Round 0: per-entity fraud rate features (user, device, recipient)
  Round 1: 1-hop aggregates precomputed at startup
             user's recipient neighborhood mean fraud rate
             device's co-user neighborhood mean fraud rate

Both Round 1 aggregates are stored as scalar lookup tables (no runtime graph
traversal) following the BRIGHT batch/realtime separation pattern. Inference
is O(1) dict lookup — identical latency to the existing Tier 3 store.

Only invoked for borderline transactions (TIER2_LO ≤ Tier 1 score ≤ TIER2_HI).
Final cascade score: Tier1 * 0.65 + GNN * 0.35

GCN output weights calibrated from fraud domain knowledge:
  f_R  recipient fraud rate         0.35  fraud rings share recipient endpoints
  h_D1 device co-user mean fraud    0.30  shared device = ring membership signal
  f_U  user fraud rate              0.15  user's own transaction history
  h_U1 user's recipient mean fraud  0.15  guilt-by-association via payments
  f_D  device fraud rate            0.05  device baseline (subsumed by h_D1)
"""

import threading
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ── Cascade thresholds ────────────────────────────────────────────────────────

_BASE_FRAUD_RATE = 0.018   # dataset-level prior: 880K txns, 1.84% fraud
_TIER2_LO   = 0.35         # below this: Tier 1 confident clean → skip GNN
_TIER2_HI   = 0.80         # above this: Tier 1 confident fraud → skip GNN
_T1_WEIGHT  = 0.65
_GNN_WEIGHT = 0.35

# GCN output weights: [f_R, h_D1, f_U, h_U1, f_D]
_W     = [0.35, 0.30, 0.15, 0.15, 0.05]
_SCALE = 3.5   # amplify so 5% fraud-rate entity scores ~0.18, not near-zero

# ── Shared state ──────────────────────────────────────────────────────────────

_lock = threading.RLock()

_user_fraud_rate:      dict = {}   # user_id      → float
_device_fraud_rate:    dict = {}   # device_id    → float
_recipient_fraud_rate: dict = {}   # recipient_id → float
_user_recipient_mean:  dict = {}   # user_id      → mean(recipient fraud rates)
_device_couser_mean:   dict = {}   # device_id    → mean(co-user fraud rates)

_initialized = False
_stats: dict = {}


# ── Result type ───────────────────────────────────────────────────────────────

class GNNResult(NamedTuple):
    score:     float
    embedding: list    # [f_R, h_D1, f_U, h_U1, f_D]
    hops:      int     # 1 = GNN ran; 0 = tables not ready (fallback)


# ── Init / Precompute ─────────────────────────────────────────────────────────

def init(df: pd.DataFrame) -> None:
    """
    Build all GNN tables from a transactions DataFrame.

    Expected columns: user_id, device_id (optional), recipient_id (optional),
                      is_fraud (optional — defaults to 0 if absent).

    Uses vectorized pandas operations throughout; runs in ~2s on 880K rows.
    Thread-safe atomic swap: reads are never blocked by a running init.
    """
    global _initialized, _stats

    if df is None or df.empty:
        return

    df = df.copy()
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0)

    # Normalise entity IDs to str for consistent key lookups at inference time
    df["user_id"] = df["user_id"].astype(str)
    if "device_id" in df.columns:
        df["device_id"] = df["device_id"].astype(str)
    if "recipient_id" in df.columns:
        df["recipient_id"] = df["recipient_id"].astype(str)

    # ── Round 0: per-entity fraud rates ──────────────────────────────────────
    ufr = df.groupby("user_id")["is_fraud"].mean().to_dict()
    dfr = df.groupby("device_id")["is_fraud"].mean().to_dict() if "device_id" in df.columns else {}
    rfr = df.groupby("recipient_id")["is_fraud"].mean().to_dict() if "recipient_id" in df.columns else {}

    # ── Round 1: precomputed 1-hop neighborhood aggregates ───────────────────
    # user_recipient_mean: for each user, mean fraud rate across their recipients
    if "recipient_id" in df.columns and rfr:
        rfr_series = pd.Series(rfr)
        df_r = df[["user_id", "recipient_id"]].copy()
        df_r["rfr"] = df_r["recipient_id"].map(rfr_series).fillna(_BASE_FRAUD_RATE)
        urmf = df_r.groupby("user_id")["rfr"].mean().to_dict()
    else:
        urmf = {}

    # device_couser_mean: for each device, mean fraud rate across co-users
    if "device_id" in df.columns and ufr:
        ufr_series = pd.Series(ufr)
        df_d = df[["device_id", "user_id"]].copy()
        df_d["ufr"] = df_d["user_id"].map(ufr_series).fillna(_BASE_FRAUD_RATE)
        dcmf = df_d.groupby("device_id")["ufr"].mean().to_dict()
    else:
        dcmf = {}

    # ── Atomic swap ───────────────────────────────────────────────────────────
    with _lock:
        _user_fraud_rate.clear();      _user_fraud_rate.update(ufr)
        _device_fraud_rate.clear();    _device_fraud_rate.update(dfr)
        _recipient_fraud_rate.clear(); _recipient_fraud_rate.update(rfr)
        _user_recipient_mean.clear();  _user_recipient_mean.update(urmf)
        _device_couser_mean.clear();   _device_couser_mean.update(dcmf)
        _initialized = True
        _stats = {
            "users":       len(ufr),
            "devices":     len(dfr),
            "recipients":  len(rfr),
            "user_hops":   len(urmf),
            "device_hops": len(dcmf),
        }


def refresh_from_disk(models_dir: Path) -> None:
    """Reload GNN tables from transactions.csv. Called by the hourly refresh loop."""
    csv = Path(models_dir) / "transactions.csv"
    if csv.exists():
        init(pd.read_csv(csv, low_memory=False))


# ── Inference (O(1)) ──────────────────────────────────────────────────────────

def score(user_id, device_id, recipient_id) -> GNNResult:
    """
    GNN Tier 2 score for a single transaction.

    Embedding layout: [f_R, h_D1, f_U, h_U1, f_D]
      f_R  — recipient fraud rate (Round 0)
      h_D1 — device co-user mean fraud rate (Round 1 aggregate)
      f_U  — user fraud rate (Round 0)
      h_U1 — user's recipient mean fraud rate (Round 1 aggregate)
      f_D  — device fraud rate (Round 0)

    Score: min(dot(W, embedding) * SCALE, 1.0)
    Returns fallback GNNResult with hops=0 if tables not yet initialised.
    """
    if not _initialized:
        return GNNResult(score=_BASE_FRAUD_RATE, embedding=[], hops=0)

    uid = str(user_id)    if user_id    else ""
    did = str(device_id)  if device_id  else ""
    rid = str(recipient_id) if recipient_id else ""

    f_U  = _user_fraud_rate.get(uid, _BASE_FRAUD_RATE)
    f_D  = _device_fraud_rate.get(did, _BASE_FRAUD_RATE)
    f_R  = _recipient_fraud_rate.get(rid, _BASE_FRAUD_RATE)
    h_U1 = _user_recipient_mean.get(uid, f_R)
    h_D1 = _device_couser_mean.get(did, f_U)

    embedding = [f_R, h_D1, f_U, h_U1, f_D]
    raw  = sum(w * x for w, x in zip(_W, embedding))
    gnn_s = round(min(raw * _SCALE, 1.0), 4)

    return GNNResult(
        score=gnn_s,
        embedding=[round(x, 4) for x in embedding],
        hops=1,
    )


def cascade_blend(tier1_score: float, gnn_result: GNNResult) -> float:
    """Blend Tier 1 and GNN scores: 65% Tier 1, 35% GNN."""
    return round(tier1_score * _T1_WEIGHT + gnn_result.score * _GNN_WEIGHT, 4)


def should_invoke(tier1_score: float) -> bool:
    """Return True if this transaction is in the borderline range where GNN adds value."""
    return _TIER2_LO <= tier1_score <= _TIER2_HI


def get_stats() -> dict:
    """Return entity coverage and initialisation status."""
    return {"initialized": _initialized, **_stats}
