"""
core/ingest_schema.py - the ingestion contract: validate and normalise inbound events.

The ingestion surface used to take a free-form dict and let missing fields default to 0.0
at scoring time. That silently corrupts signal: a transaction with no amount does not fail,
it scores as if the amount were zero. This module is the schema-first fix, the foundation an
ingestion pipeline is built on.

validate_event(raw):
  * REJECTS genuinely broken input instead of coercing it. A non-numeric or negative amount,
    or an event with neither an amount nor precomputed features, is an error, not a silent 0.
  * NORMALISES onto a copy of the raw event, so every passthrough field the feature engine
    reads is preserved, while amount / rail / currency / timestamp are cleaned and a
    provenance block is stamped on.
  * FLAGS label-only fields (fraud_typology, is_fraud) as a leakage risk: they describe the
    answer and must never be fed to the model as features.
  * WARNS (does not fail) on missing-but-recommended fields (rail, a subject id), so an
    operator sees degraded signal without dropping the event.

Returns {valid, event, errors, warnings}. Pure stdlib, unit-testable without the ML stack.
"""

from __future__ import annotations

import uuid
from datetime import datetime

SCHEMA_VERSION = "1.0"

# Canonical payment rail -> the raw synonyms that normalise to it.
RAILS = {
    "card":  {"card", "credit_card", "debit_card", "cards", "card_present", "card_not_present"},
    "ach":   {"ach", "ach_debit", "ach_credit"},
    "wire":  {"wire", "swift", "fedwire"},
    "zelle": {"zelle"},
    "rtp":   {"rtp", "realtime", "real_time", "real-time"},
    "fps":   {"fps", "faster_payments", "faster-payments"},
    "sepa":  {"sepa", "sepa_ct", "sepa_inst"},
    "crypto": {"crypto", "cryptocurrency", "btc", "eth", "usdt", "onchain", "on_chain"},
    "paypal": {"paypal"},
    "check": {"check", "cheque"},
}
_RAIL_LOOKUP = {syn: canon for canon, syns in RAILS.items() for syn in syns}

# Fields that describe the ANSWER, not the transaction. They are kept for labelling / liability
# but must never be used as model features (that would be leakage).
LABEL_FIELDS = ("fraud_typology", "is_fraud", "label")

# A subject id is needed for entity linkage and reputation; at least one is recommended.
SUBJECT_FIELDS = ("user_id", "account_id")


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _err(field: str, code: str, message: str) -> dict:
    return {"field": field, "code": code, "message": message}


def _warn(field: str, message: str) -> dict:
    return {"field": field, "message": message}


def normalize_rail(value) -> tuple:
    """Return (canonical_rail, recognised?). Unknown rails pass through lowercased."""
    v = str(value or "").strip().lower()
    if not v:
        return "", False
    if v in _RAIL_LOOKUP:
        return _RAIL_LOOKUP[v], True
    return v, False


def _norm_timestamp(ts, warnings) -> str:
    if ts is None or ts == "":
        return _now()
    if isinstance(ts, (int, float)):                      # epoch seconds
        try:
            return datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
        except (ValueError, OSError, OverflowError):
            warnings.append(_warn("timestamp", f"uninterpretable epoch {ts!r}; used ingest time"))
            return _now()
    return str(ts)                                        # trust an ISO-ish string as given


def validate_event(raw: dict, source: str = "") -> dict:
    """Validate and normalise one inbound event. See module docstring for the contract."""
    errors: list = []
    warnings: list = []
    raw = raw if isinstance(raw, dict) else {}
    ev = dict(raw)                                        # normalise onto a copy; preserve passthrough

    # -- transaction id (idempotency key) --
    tid = str(raw.get("transaction_id") or raw.get("txn_id") or "").strip()
    if not tid:
        tid = "txn_" + uuid.uuid4().hex[:12]
        warnings.append(_warn("transaction_id",
                              "missing; generated a synthetic id (replay/idempotency will be weaker)"))
    ev["transaction_id"] = tid

    # -- amount: the field that used to default to 0 and kill signal --
    has_features = isinstance(raw.get("features"), dict) and bool(raw.get("features"))
    amt_raw = raw.get("amount", None)
    if amt_raw is None or amt_raw == "":
        if not has_features:
            errors.append(_err("amount", "required",
                               "no amount and no precomputed features; nothing to score (refusing to assume 0)"))
    else:
        try:
            amt = float(amt_raw)
            if amt < 0:
                errors.append(_err("amount", "invalid", f"amount is negative ({amt})"))
            else:
                ev["amount"] = round(amt, 2)
        except (TypeError, ValueError):
            errors.append(_err("amount", "invalid",
                               f"amount is not numeric: {amt_raw!r} (refusing to default to 0)"))

    # -- currency --
    cur = str(raw.get("currency", "USD") or "USD").upper()
    if len(cur) != 3 or not cur.isalpha():
        warnings.append(_warn("currency", f"unusual currency code {cur!r}"))
    ev["currency"] = cur

    # -- timestamp --
    ev["timestamp"] = _norm_timestamp(raw.get("timestamp"), warnings)

    # -- payment rail (keep the key the feature engine reads) --
    rail_raw = raw.get("payment_rail") or raw.get("rail")
    if rail_raw:
        canon, known = normalize_rail(rail_raw)
        ev["payment_rail"] = canon
        if not known:
            warnings.append(_warn("payment_rail", f"unknown rail {rail_raw!r}; passed through unnormalised"))
    else:
        warnings.append(_warn("payment_rail", "missing; rail-specific liability pricing will be generic"))

    # -- subject ids --
    if not any(str(raw.get(k) or "").strip() for k in SUBJECT_FIELDS):
        warnings.append(_warn("user_id", "no subject id (user_id/account_id); entity linkage and reputation will be weak"))

    # -- precomputed features must be numeric --
    if has_features:
        clean = {}
        for k, v in raw["features"].items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                warnings.append(_warn(f"features.{k}", f"non-numeric feature dropped: {v!r}"))
        ev["features"] = clean

    # -- leakage guard: flag label-only fields (kept, but never to be used as features) --
    present_labels = [k for k in LABEL_FIELDS if raw.get(k) not in (None, "")]
    if present_labels:
        ev["_label_fields"] = present_labels
        warnings.append(_warn(",".join(present_labels),
                              "label-only field present; must not be used as a model feature (leakage)"))

    ev["_ingest"] = {"schema_version": SCHEMA_VERSION, "ingest_ts": _now(),
                     "source": source or "unknown", "raw_field_count": len(raw)}

    return {
        "valid": not errors,
        "event": ev if not errors else None,
        "errors": errors,
        "warnings": warnings,
    }


def contract() -> dict:
    """The self-documenting ingestion contract (served at GET /ingest/schema)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "required": {
            "amount_or_features": "a numeric amount >= 0, OR a precomputed `features` object",
        },
        "recommended": {
            "transaction_id": "stable idempotency key (auto-generated if absent)",
            "payment_rail": f"one of {sorted(RAILS.keys())} (synonyms normalised)",
            "subject_id": f"at least one of {list(SUBJECT_FIELDS)}",
            "timestamp": "ISO-8601 or epoch seconds (defaults to ingest time)",
        },
        "optional_passthrough": ["device_id", "recipient_id", "institution_id", "currency",
                                 "merchant", "mcc", "ip", "geo"],
        "label_only_do_not_feature": list(LABEL_FIELDS),
        "rails": {canon: sorted(syns) for canon, syns in RAILS.items()},
    }
