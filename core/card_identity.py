"""
core/card_identity.py - a stable card key that is never a PAN.

WHY THIS EXISTS. The sequence gate asks "how has THIS card behaved in the last 24 hours", and
the authorization path could not answer it. What the message carried was `bin` (the issuer and
product, shared by millions of cards) and `cardholder_name` (neither unique nor stable). Neither
identifies a card, so every per-card control was unbuildable, not merely unbuilt.

THE PAN IS NEVER STORED, AND THAT IS NOT A STYLE PREFERENCE. A stored PAN puts the whole system
in PCI DSS scope and turns the decision substrate into cardholder data. Everything downstream of
this file, the decisions table included, holds only a SALTED HASH. The raw value exists for the
length of one function call and is never written, logged, or returned.

SALTED, NOT PLAIN. A bare SHA-256 of a PAN is reversible in practice: the space is ~10^15 with a
known checksum and known BIN ranges, so an attacker with the table can enumerate it in hours. The
salt comes from the environment and never from the code, so a leaked database without the salt
does not yield card numbers.

PREFERENCE ORDER, and it is deliberate. A network TOKEN is better than a PAN: it is already the
industry's answer to this problem, it is per-merchant or per-device, and it does not need the
salt to be safe. If the message carries one, that is what identifies the card here.
"""

from __future__ import annotations

import hashlib
import os
import re

# Environment-supplied and never defaulted to a constant in code. A hardcoded fallback would make
# every deployment share one salt, which is the same as having none.
_SALT_ENV = "REDWING_CARD_SALT"

# Fields that already identify a card WITHOUT being a PAN. Preferred, in this order.
TOKEN_FIELDS = ("card_token", "network_token", "token_pan", "card_id", "card_key")

# Fields that may carry a real PAN. Hashed immediately, never retained.
PAN_FIELDS = ("pan", "card_number", "primary_account_number")

_DIGITS = re.compile(r"\D")


def _salt() -> str:
    """The salt, or empty. An absent salt is reported by `salt_configured()` rather than being
    silently replaced, because a system quietly hashing with no salt looks identical to one doing
    it properly right up until the database leaks."""
    return os.environ.get(_SALT_ENV, "")


def salt_configured() -> bool:
    return bool(_salt())


def _hash(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(f"{_salt()}:{prefix}:{value}".encode()).hexdigest()[:24]


def card_key(msg: dict) -> str:
    """A stable, non-reversible key for the card this authorization is on, or "".

    Returns "" when the message carries nothing that identifies a card. That is a real and
    common case (an ISO 8583 message routed without a token, a test harness), and the callers
    treat it as "no per-card history available" rather than inventing one. A synthesised key
    would silently give every unidentified card a shared history, which is worse than none:
    the sequence gate would then see one enormous card bursting constantly and fire on
    everything.
    """
    for f in TOKEN_FIELDS:
        v = str(msg.get(f, "") or "").strip()
        if v:
            # A token is already a safe surrogate, but it is still hashed so that one storage
            # format covers both paths and nothing downstream has to know which was used.
            return _hash("tok_", v)

    for f in PAN_FIELDS:
        raw = _DIGITS.sub("", str(msg.get(f, "") or ""))
        if len(raw) >= 12:                       # shortest real PAN is 12 digits
            # AN UNSALTED PAN HASH IS A REVERSIBLE PAN, so refuse rather than write a weak key.
            #
            # `salt_configured()` existed for exactly this and had ZERO production callers, so
            # nothing ever stopped it. Unsalted, `_hash` is a global constant function of the card
            # number: an attacker with the table precomputes sha256 over the Luhn-valid space for
            # the BINs they care about, and the BIN is stored in the clear in the same rationale
            # block. That turns a database leak into a cardholder-data breach, which is precisely
            # what this module's docstring promises is impossible.
            #
            # Returning "" rather than raising: this sits on a live authorization path, and every
            # caller already handles "no card identifier" (it is a real and common case). The
            # sequence gate degrades and says so. A raise here would fail the authorization, and
            # a misconfiguration must not decline a customer's payment.
            if not salt_configured():
                return ""
            key = _hash("pan_", raw)
            del raw                              # not a security control, a statement of intent
            return key
    return ""


def last4(msg: dict) -> str:
    """The last four digits, which are safe to display and are NOT part of the key.

    Kept separate on purpose. Last four is shown to analysts and printed on receipts, so it is
    not sensitive, but it is also not identifying: many cards share any given four digits. Using
    it as the key would merge unrelated cards into one history.
    """
    for f in PAN_FIELDS:
        raw = _DIGITS.sub("", str(msg.get(f, "") or ""))
        if len(raw) >= 4:
            return raw[-4:]
    return ""
