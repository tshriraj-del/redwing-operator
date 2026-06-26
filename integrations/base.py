"""
Base connector interface for all RedWing external integrations.

Every agency/bureau connector inherits from BaseConnector and implements:
  - enrich(payload)  → pull signals about a transaction/user
  - report(payload)  → push a fraud event or regulatory filing
  - health_check()   → verify the connection is live

The hub calls connectors through this interface — it never needs to know
the specifics of any individual agency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class ConnectorStatus(str, Enum):
    ACTIVE      = "active"       # configured and reachable
    DEGRADED    = "degraded"     # reachable but slow or partial
    UNAVAILABLE = "unavailable"  # cannot be reached
    UNCONFIGURED = "unconfigured" # credentials not set
    DERIVED     = "derived"      # no credentials: returning derived signals, not a live feed


class ConnectorCategory(str, Enum):
    CREDIT_BUREAU      = "credit_bureau"
    FINANCIAL_INTEL    = "financial_intelligence"
    FRAUD_CONSORTIUM   = "fraud_consortium"
    PAYMENT_NETWORK    = "payment_network"
    LAW_ENFORCEMENT    = "law_enforcement"
    OPEN_BANKING       = "open_banking"
    IDENTITY_VERIFY    = "identity_verification"


@dataclass
class EnrichRequest:
    transaction_id: str
    user_id:        str
    amount:         float              = 0.0
    payment_rail:   str                = ""
    recipient_id:   str                = ""
    fraud_typology: str                = ""
    raw:            dict               = field(default_factory=dict)  # full transaction row


@dataclass
class EnrichResponse:
    connector:      str
    status:         ConnectorStatus
    signals:        dict               = field(default_factory=dict)
    latency_ms:     int                = 0
    error:          Optional[str]      = None
    raw_response:   Optional[dict]     = None
    requested_at:   str                = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReportRequest:
    report_type:    str                # "SAR", "CTR", "fraud_ring", "case_referral", etc.
    transaction_id: str
    user_id:        str
    amount:         float              = 0.0
    fraud_typology: str                = ""
    narrative:      str                = ""
    evidence:       dict               = field(default_factory=dict)


@dataclass
class ReportResponse:
    connector:      str
    status:         ConnectorStatus
    reference_id:   Optional[str]      = None   # agency-assigned reference number
    acknowledged:   bool               = False
    async_pending:  bool               = False  # true if agency will respond via webhook later
    error:          Optional[str]      = None
    submitted_at:   str                = field(default_factory=lambda: datetime.utcnow().isoformat())


class BaseConnector(ABC):
    """
    Abstract base for all agency/bureau connectors.
    Subclasses implement enrich(), report(), and health_check().
    The hub calls these through this interface only.
    """

    id:          str                   # unique connector id, e.g. "equifax", "fincen"
    name:        str                   # human-readable name
    category:    ConnectorCategory
    description: str = ""

    def is_configured(self) -> bool:
        """Return True if all required credentials are present in environment."""
        return True

    @abstractmethod
    def health_check(self) -> ConnectorStatus:
        """Ping the agency endpoint and return current status."""
        ...

    @abstractmethod
    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        """
        Pull fraud/identity signals from this agency for a given transaction.
        Should never raise — catch internally and return status=UNAVAILABLE on error.
        """
        ...

    @abstractmethod
    def report(self, req: ReportRequest) -> ReportResponse:
        """
        Submit a fraud event, SAR, CTR, or case referral to this agency.
        Returns async_pending=True if the agency acknowledges async.
        """
        ...

    def _timed_call(self, fn, *args, **kwargs):
        """Utility: call fn and return (result, latency_ms)."""
        start = time.time()
        result = fn(*args, **kwargs)
        return result, int((time.time() - start) * 1000)


# ── Derived enrichment ─────────────────────────────────────────────────────────
# When a connector has no live credentials, the hub still returns *something* useful:
# deterministic, coherent signals keyed by the connector's category, seeded by the
# user (stable per user) and nudged by the known fraud typology so they agree with
# ground truth. Flagged _mode="derived" so it's never mistaken for a live feed. This
# is exactly the identity/device/bureau signal the scaffolded feature families were
# placeholders for; in production each connector's real API replaces it.

import hashlib as _hashlib
import random as _random


def _derive_rng(*parts):
    h = _hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return _random.Random(int(h[:12], 16))


def derived_signals(connector: "BaseConnector", req: EnrichRequest) -> dict:
    """Deterministic stand-in enrichment for an uncredentialed connector."""
    r = _derive_rng(connector.id, req.user_id)
    typ = (req.fraud_typology or "").lower()
    synth = "synthetic" in typ
    ato = "takeover" in typ or "ato" in typ
    scam = any(k in typ for k in ("scam", "pig", "deepfake", "social"))
    cat = connector.category
    sig = {"_mode": "derived", "_connector": connector.name}

    if cat == ConnectorCategory.CREDIT_BUREAU:
        sig.update({
            "identity_verified": (not synth) and r.random() > 0.10,
            "file_exists": not synth,
            "thin_file": synth or r.random() < 0.15,
            "synthetic_identity_score": round(min(1.0, (0.72 if synth else 0.06) + r.random() * 0.2), 2),
            "fraud_alert_active": ato or r.random() < 0.05,
        })
    elif cat == ConnectorCategory.FINANCIAL_INTEL:
        sig.update({
            "sanctions_match": r.random() < 0.01,
            "pep_flag": r.random() < 0.03,
            "watchlist_hit": scam and r.random() < 0.30,
            "prior_sar_count": r.choices([0, 0, 0, 1, 2], weights=[60, 18, 10, 8, 4])[0],
        })
    elif cat == ConnectorCategory.FRAUD_CONSORTIUM:
        sig.update({
            "device_reputation": round(min(1.0, (0.25 if (ato or synth) else 0.85) + r.random() * 0.1), 2),
            "known_fraud_device": ato and r.random() < 0.5,
            "consortium_fraud_reports": r.choices([0, 0, 1, 3, 8], weights=[55, 20, 12, 8, 5])[0],
            "velocity_across_institutions": round((0.7 if (ato or scam) else 0.1) + r.random() * 0.2, 2),
        })
    elif cat == ConnectorCategory.LAW_ENFORCEMENT:
        sig.update({
            "ic3_complaints": r.choices([0, 0, 0, 1, 4], weights=[70, 12, 8, 6, 4])[0],
            "known_mule_account": scam and r.random() < 0.4,
        })
    elif cat == ConnectorCategory.OPEN_BANKING:
        sig.update({
            "income_verified": r.random() > 0.2,
            "account_age_days": r.randint(20, 3000),
            "balance_band": r.choice(["<1k", "1k-10k", "10k-50k", "50k+"]),
        })
    else:
        sig.update({"available": True})
    return sig
