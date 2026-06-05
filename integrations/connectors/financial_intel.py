"""
Financial Intelligence Connectors — FinCEN, FCA, AUSTRAC, RBI.

Purpose:
  enrich()  → sanctions screening, PEP (politically exposed person) check,
               adverse media, watchlist lookup
  report()  → SAR (Suspicious Activity Report), CTR (Currency Transaction Report),
               TSAR (Terrorist Financing SAR), regulatory disclosure

Key compliance notes:
  - SAR filings are legally mandated above certain thresholds (US: $5,000+)
  - CTR filings required for cash transactions > $10,000
  - Filing must occur within 30 days of detection (FinCEN)
  - Reports are confidential — do NOT notify the subject (tipping-off offense)
  - All filings must be audit-logged with analyst name, timestamp, rationale

Credentials needed (in operator/.env):
  FINCEN_API_KEY, FINCEN_ORG_ID
  FCA_API_KEY
  OFAC_API_KEY  (OFAC SDN list screening)
"""

import os
from ..base import (
    BaseConnector, ConnectorCategory, ConnectorStatus,
    EnrichRequest, EnrichResponse, ReportRequest, ReportResponse,
)


class FinCENConnector(BaseConnector):
    id          = "fincen"
    name        = "FinCEN (BSA E-Filing)"
    category    = ConnectorCategory.FINANCIAL_INTEL
    description = "SAR and CTR filing via BSA E-Filing System. Mandatory for US financial institutions."

    # SAR threshold: transactions >= $5,000 with fraud indicators
    SAR_THRESHOLD = 5_000
    # CTR threshold: cash transactions >= $10,000
    CTR_THRESHOLD = 10_000

    def is_configured(self) -> bool:
        return bool(os.environ.get("FINCEN_API_KEY") and os.environ.get("FINCEN_ORG_ID"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        # TODO: GET https://bsaefiling1.fincen.treas.gov/api/v1/health
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="FINCEN_API_KEY / FINCEN_ORG_ID not set")
        # TODO: Check if transaction_id or user_id has prior SAR history
        # GET /api/v1/sar/lookup?ref={transaction_id}
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              signals={"sar_threshold_met": req.amount >= self.SAR_THRESHOLD,
                                       "ctr_threshold_met": req.amount >= self.CTR_THRESHOLD})

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="Credentials not configured")
        # TODO:
        # 1. Build FinXML payload per BSA E-Filing schema
        # 2. POST https://bsaefiling1.fincen.treas.gov/api/v1/sar/submit
        # 3. Store BSA ID (reference_id) returned by FinCEN
        # 4. Set async_pending=True — FinCEN acknowledges within 24-48h
        # NOTE: DO NOT notify the subject. Tipping-off is a federal offense.
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              async_pending=True)


class OFACConnector(BaseConnector):
    id          = "ofac"
    name        = "OFAC SDN Screening"
    category    = ConnectorCategory.FINANCIAL_INTEL
    description = "Sanctions screening against OFAC Specially Designated Nationals list"

    def is_configured(self) -> bool:
        return bool(os.environ.get("OFAC_API_KEY"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        # TODO: GET https://api.ofac-api.com/v3/health
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="OFAC_API_KEY not set")
        # TODO:
        # POST https://api.ofac-api.com/v3/screen
        # Body: { "cases": [{ "name": user_id, "type": "individual" }] }
        # Return: ofac_match, match_score, matched_entity
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        # OFAC matches are reported to FinCEN, not to OFAC directly
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              error="OFAC matches: report via FinCEN connector")


class FCAConnector(BaseConnector):
    id          = "fca"
    name        = "FCA (UK Financial Conduct Authority)"
    category    = ConnectorCategory.FINANCIAL_INTEL
    description = "UK suspicious activity reporting, FCA register verification"

    def is_configured(self) -> bool:
        return bool(os.environ.get("FCA_API_KEY"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="FCA_API_KEY not set")
        # TODO: GET https://register.fca.org.uk/services/V0.1/Firm/{firm_id}
        # Check if recipient is an authorised firm
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="Credentials not configured")
        # TODO: Submit SAR via UK NCA UKFIU reporting portal
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                              async_pending=True)
