"""
Offline graph feature store — Tier 3 of the RedWing 4-tier cascade.

Precomputes per-entity statistics from transactions.csv at startup and
refreshes every hour. At score time, get_features() is an O(1) dict lookup
with zero latency overhead.

This implements the "batch-precomputed embeddings" half of the BRIGHT
architecture (arXiv 2205.13084) — real node features rather than learned
embeddings, but the same latency-separation pattern.

Features per entity type:
  device   → distinct users sharing it, historical fraud rate
  recipient → historical fraud hit rate, transaction volume
  user      → historical fraud rate, fraud neighbor count (1-hop ring signal)

The aggregate graph_risk_score feeds into the autonomous_decision() blend
in agent.py alongside the XGBoost score and AI behavioral confidence.
"""

import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── State ─────────────────────────────────────────────────────────────────────

_lock = threading.Lock()

_device_user_count:   dict = {}   # device_id → int (distinct users)
_device_fraud_rate:   dict = {}   # device_id → float
_recipient_fraud_rate: dict = {}  # recipient_id → float
_recipient_tx_count:  dict = {}   # recipient_id → int
_user_fraud_rate:     dict = {}   # user_id → float
_user_fraud_neighbors: dict = {}  # user_id → int (ring membership proxy)

_stats: dict = {"entities": 0, "transactions": 0, "last_refresh": None}


# ── Precomputation ─────────────────────────────────────────────────────────────

def precompute(df: pd.DataFrame) -> None:
    """
    Build all lookup tables from a transactions DataFrame.
    Thread-safe atomic swap — reads are never blocked by a running precompute.
    Called at startup and by the hourly refresh task.
    """
    if df is None or df.empty:
        return

    has_fraud = "is_fraud" in df.columns

    # ── Collect per-entity fraud tallies in one pass ──────────────────────────
    device_users: dict  = defaultdict(set)
    device_fraud: dict  = defaultdict(list)
    recip_fraud:  dict  = defaultdict(list)
    user_fraud:   dict  = defaultdict(list)

    for row in df.itertuples(index=False):
        uid  = str(getattr(row, "user_id",      ""))
        did  = str(getattr(row, "device_id",    "")) if hasattr(row, "device_id")    else ""
        rid  = str(getattr(row, "recipient_id", "")) if hasattr(row, "recipient_id") else ""
        is_f = int(bool(getattr(row, "is_fraud", 0))) if has_fraud else 0

        if did and did not in ("nan", "None", ""):
            device_users[did].add(uid)
            device_fraud[did].append(is_f)

        if rid and rid not in ("nan", "None", ""):
            recip_fraud[rid].append(is_f)

        if uid:
            user_fraud[uid].append(is_f)

    # ── Derived tables ────────────────────────────────────────────────────────
    new_duc = {d: len(u) for d, u in device_users.items()}

    new_dfr = {
        d: round(sum(v) / len(v), 4)
        for d, v in device_fraud.items() if v
    }

    new_rfr = {
        r: round(sum(v) / len(v), 4)
        for r, v in recip_fraud.items() if v
    }
    new_rtx = {r: len(v) for r, v in recip_fraud.items()}

    new_ufr = {
        u: round(sum(v) / len(v), 4)
        for u, v in user_fraud.items() if v
    }

    # ── 1-hop fraud neighbor count (ring membership proxy) ────────────────────
    # For each device shared by multiple users, count how many users on that
    # device have a non-zero fraud rate. Clean users on a "dirty" device are
    # flagged with the count of those dirty co-users — a simplified GNN
    # 1-hop neighbourhood aggregation.
    new_ufn: dict = defaultdict(int)
    for did, users in device_users.items():
        fraud_users = [u for u in users if new_ufr.get(u, 0.0) > 0]
        if fraud_users:
            for u in users:
                if new_ufr.get(u, 0.0) == 0.0:
                    new_ufn[u] += len(fraud_users)

    # ── Atomic swap ──────────────────────────────────────────────────────────
    now = datetime.utcnow()
    with _lock:
        _device_user_count.clear();   _device_user_count.update(new_duc)
        _device_fraud_rate.clear();   _device_fraud_rate.update(new_dfr)
        _recipient_fraud_rate.clear();_recipient_fraud_rate.update(new_rfr)
        _recipient_tx_count.clear();  _recipient_tx_count.update(new_rtx)
        _user_fraud_rate.clear();     _user_fraud_rate.update(new_ufr)
        _user_fraud_neighbors.clear();_user_fraud_neighbors.update(new_ufn)

    _stats.update({
        "entities":     len(new_duc) + len(new_rfr) + len(new_ufr),
        "transactions": len(df),
        "last_refresh": now.isoformat() + "Z",
    })


def refresh_from_disk(models_dir: Path) -> None:
    """Reload from transactions.csv. Run in executor — reads 880K rows."""
    try:
        df = pd.read_csv(models_dir / "transactions.csv")
        precompute(df)
    except Exception:
        pass


# ── Feature lookup ─────────────────────────────────────────────────────────────

def get_features(
    user_id:      Optional[str],
    device_id:    Optional[str],
    recipient_id: Optional[str],
) -> dict:
    """
    O(1) graph context lookup for a transaction.
    Returns a dict that build_event() attaches to the event payload.
    """
    uid = str(user_id)      if user_id      else ""
    did = str(device_id)    if device_id    else ""
    rid = str(recipient_id) if recipient_id else ""

    with _lock:
        duc = _device_user_count.get(did, 1)
        dfr = _device_fraud_rate.get(did, 0.0)
        rfr = _recipient_fraud_rate.get(rid, 0.0)
        rtx = _recipient_tx_count.get(rid, 0)
        ufr = _user_fraud_rate.get(uid, 0.0)
        ufn = _user_fraud_neighbors.get(uid, 0)

    return {
        "device_shared_users":   duc,
        "device_fraud_rate":     dfr,
        "recipient_fraud_rate":  rfr,
        "recipient_tx_count":    rtx,
        "user_historical_fraud": ufr,
        "user_fraud_neighbors":  ufn,
        "graph_risk_score":      _aggregate(duc, dfr, rfr, ufr, ufn),
    }


def _aggregate(
    device_shared_users: int,
    device_fraud_rate:   float,
    recipient_fraud_rate: float,
    user_hist_fraud:     float,
    user_fraud_neighbors: int,
) -> float:
    """
    Aggregate per-entity graph stats into a single 0–1 risk score.

    Weights match the information value of each signal for fraud ring detection:
      30% recipient fraud rate  — who is this money going to?
      25% device fraud rate     — is this device contaminated?
      20% shared device risk    — ≥4 co-users on device = suspicious
      15% user historical fraud — this user's own fraud history
      10% fraud neighbors       — ring membership (normalized at 5+ neighbors)
    """
    shared_risk   = min((device_shared_users - 1) / 4.0, 1.0) if device_shared_users > 1 else 0.0
    neighbor_risk = min(user_fraud_neighbors / 5.0, 1.0)

    score = (
          0.30 * min(recipient_fraud_rate * 3.0, 1.0)
        + 0.25 * min(device_fraud_rate    * 3.0, 1.0)
        + 0.20 * shared_risk
        + 0.15 * min(user_hist_fraud      * 5.0, 1.0)
        + 0.10 * neighbor_risk
    )
    return round(min(score, 1.0), 4)


def get_stats() -> dict:
    return dict(_stats)
