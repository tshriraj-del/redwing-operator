"""
Integration Hub — central registry and orchestrator for all external connectors.

The operator never calls individual connectors directly.
It calls the hub with a list of signal types needed, and the hub:
  1. Routes to the right connectors
  2. Runs them concurrently
  3. Normalises responses into a single signals dict
  4. Audit-logs every call
  5. Returns within a timeout budget

Usage:
    from integrations.hub import hub

    signals = hub.enrich(
        req=EnrichRequest(transaction_id="txn_001", user_id="u_123", amount=4500),
        connectors=["equifax", "ofac", "threatmetrix"]
    )

    result = hub.report(
        req=ReportRequest(report_type="SAR", transaction_id="txn_001", ...),
        connectors=["fincen"]
    )
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import (
    BaseConnector, ConnectorCategory, ConnectorStatus,
    EnrichRequest, EnrichResponse, ReportRequest, ReportResponse,
    derived_signals,
)

# ── Import all connectors ──────────────────────────────────────────────────
from .connectors.credit_bureaus    import EquifaxConnector, ExperianConnector, TransUnionConnector
from .connectors.financial_intel   import FinCENConnector, OFACConnector, FCAConnector
from .connectors.fraud_consortiums import EarlyWarningConnector, ThreatMetrixConnector, ActimizeConnector
from .connectors.law_enforcement   import FBII3Connector, INTERPOLConnector, EuropolConnector
from .connectors.open_banking      import PlaidConnector, FinicityConnector, TrueLayerConnector

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = Path.home() / "pulseml_models" / "integration_audit.jsonl"
ENRICH_TIMEOUT = 5   # seconds per connector for enrichment
REPORT_TIMEOUT = 15  # seconds per connector for reporting


class IntegrationHub:
    """
    Central registry of all external connectors.
    Call hub.enrich() or hub.report() — never instantiate connectors directly.
    """

    def __init__(self):
        self._registry: dict[str, BaseConnector] = {}
        self._register_all()

    def _register_all(self):
        connectors = [
            # Credit Bureaus
            EquifaxConnector(), ExperianConnector(), TransUnionConnector(),
            # Financial Intelligence
            FinCENConnector(), OFACConnector(), FCAConnector(),
            # Fraud Consortiums
            EarlyWarningConnector(), ThreatMetrixConnector(), ActimizeConnector(),
            # Law Enforcement
            FBII3Connector(), INTERPOLConnector(), EuropolConnector(),
            # Open Banking
            PlaidConnector(), FinicityConnector(), TrueLayerConnector(),
        ]
        for c in connectors:
            self._registry[c.id] = c
        logger.info(f"Integration Hub: {len(self._registry)} connectors registered")

    # ── Public API ─────────────────────────────────────────────────────────

    def list_connectors(self) -> list[dict]:
        """Return all connectors with their current configuration status."""
        return [
            {
                "id":          c.id,
                "name":        c.name,
                "category":    c.category.value,
                "description": c.description,
                "configured":  c.is_configured(),
                "status":      c.health_check().value if c.is_configured() else ConnectorStatus.UNCONFIGURED.value,
            }
            for c in self._registry.values()
        ]

    def enrich(
        self,
        req: EnrichRequest,
        connectors: Optional[list[str]] = None,
        categories: Optional[list[ConnectorCategory]] = None,
        timeout: int = ENRICH_TIMEOUT,
    ) -> dict:
        """
        Enrich a transaction by running multiple connectors concurrently.

        Args:
            req:        The transaction to enrich.
            connectors: Specific connector IDs to run. If None, run all configured.
            categories: Filter by category instead of specific IDs.
            timeout:    Per-connector timeout in seconds.

        Returns:
            {
              "signals": { connector_id: { ...signals } },
              "errors":  { connector_id: "error message" },
              "latency_ms": int,
              "connectors_run": int,
            }
        """
        # Include uncredentialed connectors: they return derived enrichment, not nothing.
        targets = self._resolve_targets(connectors, categories, configured_only=False)
        start   = time.time()
        results = {"signals": {}, "errors": {}, "latency_ms": 0, "connectors_run": len(targets)}

        with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as pool:
            futures = {pool.submit(self._safe_enrich, c, req, timeout): c.id for c in targets}
            for future in as_completed(futures, timeout=timeout + 1):
                cid = futures[future]
                try:
                    resp: EnrichResponse = future.result()
                    if resp.signals:
                        results["signals"][cid] = resp.signals
                    if resp.error:
                        results["errors"][cid] = resp.error
                except Exception as e:
                    results["errors"][cid] = str(e)

        results["latency_ms"] = int((time.time() - start) * 1000)
        self._audit_log("enrich", req.transaction_id, req.user_id, targets, results)
        return results

    def report(
        self,
        req: ReportRequest,
        connectors: list[str],
        timeout: int = REPORT_TIMEOUT,
    ) -> dict:
        """
        Submit a fraud report or regulatory filing to specified connectors.

        Args:
            req:        The report payload.
            connectors: Specific connector IDs to report to.
            timeout:    Per-connector timeout in seconds.

        Returns:
            {
              "submitted":  { connector_id: { reference_id, async_pending } },
              "errors":     { connector_id: "error message" },
              "latency_ms": int,
            }
        """
        targets = self._resolve_targets(connectors, configured_only=True)
        start   = time.time()
        results = {"submitted": {}, "errors": {}, "latency_ms": 0}

        with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as pool:
            futures = {pool.submit(self._safe_report, c, req, timeout): c.id for c in targets}
            for future in as_completed(futures, timeout=timeout + 1):
                cid = futures[future]
                try:
                    resp: ReportResponse = future.result()
                    if resp.acknowledged or resp.async_pending:
                        results["submitted"][cid] = {
                            "reference_id":  resp.reference_id,
                            "async_pending": resp.async_pending,
                            "submitted_at":  resp.submitted_at,
                        }
                    if resp.error:
                        results["errors"][cid] = resp.error
                except Exception as e:
                    results["errors"][cid] = str(e)

        results["latency_ms"] = int((time.time() - start) * 1000)
        self._audit_log("report", req.transaction_id, req.user_id, targets, results,
                        report_type=req.report_type)
        return results

    def health(self) -> dict:
        """Return health status of all connectors."""
        return {
            c.id: (ConnectorStatus.DERIVED.value if not c.is_configured()
                   else c.health_check().value)
            for c in self._registry.values()
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _resolve_targets(
        self,
        connector_ids: Optional[list[str]] = None,
        categories: Optional[list[ConnectorCategory]] = None,
        configured_only: bool = True,
    ) -> list[BaseConnector]:
        if connector_ids:
            targets = [self._registry[cid] for cid in connector_ids if cid in self._registry]
        elif categories:
            targets = [c for c in self._registry.values() if c.category in categories]
        else:
            targets = list(self._registry.values())

        if configured_only:
            targets = [c for c in targets if c.is_configured()]

        return targets

    def _safe_enrich(self, connector: BaseConnector, req: EnrichRequest, timeout: int) -> EnrichResponse:
        # No live credentials: return deterministic derived enrichment, clearly flagged.
        if not connector.is_configured():
            return EnrichResponse(connector=connector.id, status=ConnectorStatus.DERIVED,
                                  signals=derived_signals(connector, req))
        try:
            resp = connector.enrich(req)
            # A configured-but-stubbed connector may still report UNCONFIGURED; derive then.
            if resp.status == ConnectorStatus.UNCONFIGURED or not resp.signals:
                return EnrichResponse(connector=connector.id, status=ConnectorStatus.DERIVED,
                                      signals=derived_signals(connector, req))
            return resp
        except Exception as e:
            logger.warning(f"Connector {connector.id} enrich failed: {e}")
            return EnrichResponse(connector=connector.id, status=ConnectorStatus.UNAVAILABLE, error=str(e))

    def _safe_report(self, connector: BaseConnector, req: ReportRequest, timeout: int) -> ReportResponse:
        try:
            return connector.report(req)
        except Exception as e:
            logger.warning(f"Connector {connector.id} report failed: {e}")
            return ReportResponse(connector=connector.id, status=ConnectorStatus.UNAVAILABLE, error=str(e))

    def _audit_log(self, action: str, transaction_id: str, user_id: str,
                   targets: list[BaseConnector], result: dict, **extra):
        entry = {
            "timestamp":      datetime.utcnow().isoformat(),
            "action":         action,
            "transaction_id": transaction_id,
            "user_id":        user_id,
            "connectors":     [c.id for c in targets],
            "result_summary": {
                "signals_received": list(result.get("signals", {}).keys()),
                "errors":           list(result.get("errors", {}).keys()),
                "latency_ms":       result.get("latency_ms", 0),
            },
            **extra,
        }
        try:
            AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")


# Singleton — import this everywhere
hub = IntegrationHub()
