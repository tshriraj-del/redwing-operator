"""
Open Banking Connectors — Plaid, Finicity, TrueLayer.

Purpose:
  enrich()  → account verification, balance check, transaction history,
               income verification, account age, real-time account status
  report()  → not applicable (read-only data sources)

These connectors are the primary source for transaction enrichment signals
that aren't available from the payment rail alone.

Credentials needed (in operator/.env):
  PLAID_CLIENT_ID, PLAID_SECRET, PLAID_ENV   (sandbox|development|production)
  FINICITY_APP_KEY, FINICITY_PARTNER_ID, FINICITY_PARTNER_SECRET
  TRUELAYER_CLIENT_ID, TRUELAYER_CLIENT_SECRET
"""

import os
from ..base import (
    BaseConnector, ConnectorCategory, ConnectorStatus,
    EnrichRequest, EnrichResponse, ReportRequest, ReportResponse,
)


class PlaidConnector(BaseConnector):
    id          = "plaid"
    name        = "Plaid"
    category    = ConnectorCategory.OPEN_BANKING
    description = "Account verification, transaction history, income, balance, account age"

    def is_configured(self) -> bool:
        return bool(os.environ.get("PLAID_CLIENT_ID") and os.environ.get("PLAID_SECRET"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        env = os.environ.get("PLAID_ENV", "sandbox")
        # TODO: GET https://{env}.plaid.com/health
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="PLAID_CLIENT_ID / PLAID_SECRET not set")
        # TODO:
        # 1. Exchange public_token for access_token (user must have linked account via Plaid Link)
        # 2. POST /accounts/balance/get → current balance, account age
        # 3. POST /transactions/get → 30-day transaction history
        # 4. POST /identity/get → owner name, address for identity match
        # Return: account_verified, account_age_days, balance, avg_monthly_transactions
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        return ReportResponse(connector=self.id, status=ConnectorStatus.ACTIVE,
                              error="Plaid is read-only — no reporting endpoint")


class FinicityConnector(BaseConnector):
    id          = "finicity"
    name        = "Finicity (Mastercard)"
    category    = ConnectorCategory.OPEN_BANKING
    description = "Account aggregation, income verification, cash flow analysis"

    def is_configured(self) -> bool:
        return bool(os.environ.get("FINICITY_APP_KEY") and os.environ.get("FINICITY_PARTNER_ID"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="FINICITY_APP_KEY / FINICITY_PARTNER_ID not set")
        # TODO:
        # POST https://api.finicity.com/aggregation/v2/customers/{customerId}/accounts
        # Return: account_balance, cash_flow_score, income_verified
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        return ReportResponse(connector=self.id, status=ConnectorStatus.ACTIVE,
                              error="Finicity is read-only — no reporting endpoint")


class TrueLayerConnector(BaseConnector):
    id          = "truelayer"
    name        = "TrueLayer"
    category    = ConnectorCategory.OPEN_BANKING
    description = "UK/EU open banking — account info, transaction history, payment initiation"

    def is_configured(self) -> bool:
        return bool(os.environ.get("TRUELAYER_CLIENT_ID") and os.environ.get("TRUELAYER_CLIENT_SECRET"))

    def health_check(self) -> ConnectorStatus:
        if not self.is_configured():
            return ConnectorStatus.UNCONFIGURED
        # TODO: GET https://auth.truelayer.com/.well-known/openid-configuration
        return ConnectorStatus.UNCONFIGURED

    def enrich(self, req: EnrichRequest) -> EnrichResponse:
        if not self.is_configured():
            return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED,
                                  error="TRUELAYER_CLIENT_ID / TRUELAYER_CLIENT_SECRET not set")
        # TODO:
        # GET https://api.truelayer.com/data/v1/accounts
        # GET https://api.truelayer.com/data/v1/accounts/{id}/transactions
        return EnrichResponse(connector=self.id, status=ConnectorStatus.UNCONFIGURED)

    def report(self, req: ReportRequest) -> ReportResponse:
        return ReportResponse(connector=self.id, status=ConnectorStatus.ACTIVE,
                              error="TrueLayer is read-only for enrichment — payment initiation is separate")
