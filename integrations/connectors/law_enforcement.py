"""
Law Enforcement Connectors — FBI IC3, INTERPOL, Europol EC3.

Purpose:
  enrich()  → check if recipient/device/IP appears in known fraud databases,
               cross-reference against active case files (where API access permits)
  report()  → submit fraud ring referrals, cybercrime complaints, cross-border cases

Escalation thresholds (guidelines, not hard rules):
  FBI IC3:   losses > $10,000 OR organised ring indicators
  INTERPOL:  cross-border fraud, losses > $50,000
  Europol:   EU-based victims or perpetrators

Important:
  - Law enforcement referrals are one-way — no real-time enrichment signal returned
  - Most responses are async (acknowledgement within days, not seconds)
  - Maintain a local case reference log for follow-up

Credentials needed (in operator/.env):
  FBI_IC3_API_KEY       (Internet Crime Complaint Center)
  INTERPOL_API_KEY, INTERPOL_NCB_CODE   (National Central Bureau code)
  EUROPOL_API_KEY, EUROPOL_ORG_ID
"""

import os
from ..base import (
    BaseConnector, ConnectorCategory, ConnectorStatus,
    EnrichRequest, EnrichResponse, ReportRequest, ReportResponse,
)


class FBII3Connector(BaseConnector):
    id          = "fbi_ic3"
    name        = "FBI IC3"
    category    = ConnectorCategory.LAW_ENFORCEMENT
    description = "Internet Crime Complaint Center — cybercrime and fraud referrals"

    REFERRAL_THRESHOLD = 10_000  # USD

    def is_configured(self) -> bool:
        return bool(os.environ.get("FBI_IC3_API_KEY"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        # TODO: GET https://api.ic3.gov/v1/health
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        # IC3 does not offer real-time enrichment — reporting only
        return EnrichResponse(connector=self.id, status=ConnectorStatus.ACTIVE,
                              signals={"referral_threshold_met": req.amount >= self.REFERRAL_THRESHOLD})

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="FBI_IC3_API_KEY not set")
        # TODO:
        # POST https://api.ic3.gov/v1/complaint/submit
        # Required fields: victim info, perpetrator info, fraud type, loss amount, narrative
        # Returns: IC3 complaint number (reference_id)
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              async_pending=True)


class INTERPOLConnector(BaseConnector):
    id          = "interpol"
    name        = "INTERPOL I-24/7"
    category    = ConnectorCategory.LAW_ENFORCEMENT
    description = "Cross-border fraud referrals via INTERPOL secure communications network"

    REFERRAL_THRESHOLD = 50_000  # USD

    def is_configured(self) -> bool:
        return bool(os.environ.get("INTERPOL_API_KEY") and os.environ.get("INTERPOL_NCB_CODE"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        # INTERPOL provides no real-time enrichment via API
        return EnrichResponse(connector=self.id, status=ConnectorStatus.ACTIVE,
                              signals={"referral_threshold_met": req.amount >= self.REFERRAL_THRESHOLD,
                                       "cross_border_indicator": True})

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="INTERPOL_API_KEY / INTERPOL_NCB_CODE not set")
        # TODO:
        # Submit via INTERPOL I-24/7 secure network
        # Route through National Central Bureau (NCB_CODE) of the reporting country
        # Returns: INTERPOL case reference number
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              async_pending=True)


class EuropolConnector(BaseConnector):
    id          = "europol"
    name        = "Europol EC3"
    category    = ConnectorCategory.LAW_ENFORCEMENT
    description = "European Cybercrime Centre — EU fraud ring referrals and intelligence sharing"

    def is_configured(self) -> bool:
        return bool(os.environ.get("EUROPOL_API_KEY") and os.environ.get("EUROPOL_ORG_ID"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="EUROPOL_API_KEY / EUROPOL_ORG_ID not set")
        # TODO: Submit via Europol Secure Information Exchange Network Application (SIENA)
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              async_pending=True)
