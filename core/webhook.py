"""
core/webhook.py - a source-authenticated webhook receiver (real-time push connector).

The pull connectors (file, DB table) read from a source on our schedule. A webhook is the
real-time PUSH source: a payment processor posts each authorisation to us as it happens, which
is what low-latency fraud scoring needs (decide before the payment settles).

The concern a push source adds is AUTHENTICITY. An open push endpoint lets anyone inject
events, and in a fraud platform that is not a nuisance, it is an attack: fabricated "clean"
events poison scores and, worse, poison the labels the model will train on. So this receiver
authenticates the source by an HMAC signature over the raw body before the event is allowed
anywhere near the pipeline. Only a caller holding the source's shared secret can publish.

Flow: verify the signature -> parse -> schema-validate -> publish to the durable transport.
A failure at any step is a typed rejection (401 unknown source / bad signature, 400 bad JSON,
422 bad schema, 429 backpressure), never a silent accept.

Pure stdlib (hmac, hashlib). Unit-testable without the web layer.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from .ingest_schema import validate_event
from .stream import BackpressureError


def sign(secret: str, body) -> str:
    """The signature a legitimate producer sends: 'sha256=' + HMAC-SHA256(secret, raw_body)."""
    b = body if isinstance(body, bytes) else str(body).encode()
    return "sha256=" + hmac.new(str(secret).encode(), b, hashlib.sha256).hexdigest()


def verify_signature(secret: str, body, signature: str) -> bool:
    """Constant-time verify. A missing secret or signature fails closed."""
    if not secret or not signature:
        return False
    return hmac.compare_digest(sign(secret, body), str(signature))


class WebhookReceiver:
    """Authenticates a push source by its shared secret, then validates and publishes."""

    def __init__(self, transport, secrets: dict, topic: str = "ingest"):
        self.transport = transport
        self.secrets = dict(secrets or {})        # source_id -> shared secret
        self.topic = topic

    def sources(self) -> list:
        """Registered source ids (never the secrets)."""
        return sorted(self.secrets.keys())

    def accept(self, source: str, body, signature: str) -> dict:
        """Authenticate + validate + publish one pushed event. `body` is the RAW bytes/str the
        signature was computed over. Returns a typed result with an HTTP-ish status."""
        secret = self.secrets.get(source)
        if secret is None:
            return {"accepted": False, "status": 401, "reason": "unknown source"}
        if not verify_signature(secret, body, signature):
            return {"accepted": False, "status": 401, "reason": "signature verification failed"}
        try:
            raw = json.loads(body if isinstance(body, str) else body.decode())
        except (ValueError, UnicodeDecodeError):
            return {"accepted": False, "status": 400, "reason": "body is not valid JSON"}
        if not isinstance(raw, dict):
            return {"accepted": False, "status": 400, "reason": "body must be a JSON object"}

        v = validate_event(raw, source=f"webhook:{source}")
        if not v["valid"]:
            return {"accepted": False, "status": 422, "reason": "schema_validation_failed",
                    "errors": v["errors"]}
        try:
            seq = self.transport.publish(self.topic, str(v["event"].get("transaction_id") or ""),
                                         v["event"])
        except BackpressureError as e:
            return {"accepted": False, "status": 429, "reason": str(e)}
        return {"accepted": True, "status": 202, "reason": "queued", "seq": seq,
                "deduped": seq is None, "warnings": v["warnings"]}
