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
