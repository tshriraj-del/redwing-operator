"""
Fraud Consortium Connectors — Early Warning Services, ThreatMetrix, NICE Actimize.

Purpose:
  enrich()  → cross-institution fraud signals, device reputation, email/phone risk,
               velocity across the consortium network
  report()  → contribute confirmed fraud signals back to the consortium

These are the highest-value enrichment sources for real-time fraud decisions
because they see signals across thousands of institutions simultaneously.

Credentials needed (in operator/.env):
  EWS_API_KEY, EWS_ORG_ID           (Early Warning Services — Zelle network)
  THREATMETRIX_API_KEY, THREATMETRIX_ORG_ID
  ACTIMIZE_API_KEY, ACTIMIZE_TENANT
"""

import os
from ..base import (
    BaseConnector, ConnectorCategory, ConnectorStatus,
    EnrichRequest, EnrichResponse, ReportRequest, ReportResponse,
)


class EarlyWarningConnector(BaseConnector):
    id          = "early_warning"
    name        = "Early Warning Services"
    category    = ConnectorCategory.FRAUD_CONSORTIUM
    description = "Cross-bank fraud signals, Zelle network velocity, account verification"

    def is_configured(self) -> bool:
        return bool(os.environ.get("EWS_API_KEY") and os.environ.get("EWS_ORG_ID"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        # TODO: GET https://api.earlywarning.com/v1/health
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="EWS_API_KEY / EWS_ORG_ID not set")
        # TODO:
        # POST https://api.earlywarning.com/v1/account/verify
        # Body: { "account_token": recipient_id, "transaction_amount": amount }
        # Return: account_verified, recipient_fraud_history, network_velocity_score
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="Credentials not configured")
        # TODO: POST /fraud-report — contribute confirmed fraud signal to network
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)


class ThreatMetrixConnector(BaseConnector):
    id          = "threatmetrix"
    name        = "ThreatMetrix (LexisNexis)"
    category    = ConnectorCategory.FRAUD_CONSORTIUM
    description = "Device fingerprint reputation, email/phone risk, global fraud network signals"

    def is_configured(self) -> bool:
        return bool(os.environ.get("THREATMETRIX_API_KEY") and os.environ.get("THREATMETRIX_ORG_ID"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="THREATMETRIX_API_KEY not set")
        # TODO:
        # POST https://h.online-metrix.net/api/session-query
        # Body: { "org_id": ..., "session_id": device_id, "api_key": ... }
        # Return: device_risk_score, device_seen_before, proxy_detected,
        #         bot_detected, account_email_risk, network_risk
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="Credentials not configured")
        # TODO: Contribute device + account fraud signal back to ThreatMetrix network
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)


class ActimizeConnector(BaseConnector):
    id          = "actimize"
    name        = "NICE Actimize"
    category    = ConnectorCategory.FRAUD_CONSORTIUM
    description = "Enterprise fraud intelligence, AML signals, cross-channel fraud patterns"

    def is_configured(self) -> bool:
        return bool(os.environ.get("ACTIMIZE_API_KEY") and os.environ.get("ACTIMIZE_TENANT"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="ACTIMIZE_API_KEY / ACTIMIZE_TENANT not set")
        # TODO: POST https://{tenant}.actimize.com/api/v2/transaction/score
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        if not self.is_configured():
            return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="Credentials not configured")
        return ReportResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)
