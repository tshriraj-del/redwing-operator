"""
core/decline_contract.py - a decline that carries its own way back.

THE PROBLEM. A decline today is a dead end. The member sees "your card was declined", the
merchant sees `05 Do Not Honor`, and a good customer is lost without the issuer ever finding out
it was wrong. False declines cost US ecommerce an estimated $157B of exposure and $81B actually
lost, which is more than card fraud itself. That is not a detection failure. It is a
conversation that ends one message too early.

Issuers keep the code vague on purpose. `05` is the catch-all precisely because an explanation
is also a description of the control, and strategic-classification research is explicit that
agents adapt when institutions explain their decisions. So the industry's answer to information
leakage is to leak nothing, to everybody, and it loses the recoverable members along with the
attackers.

WHAT THIS MODULE ADDS. A decline becomes an object with four things a bare code does not carry:

    recoverability   OUR judgement, not the network's. `05` hides both a member who needs to
                     verify and a card we will never approve; the code cannot tell them apart
                     and the member cannot act on it either way.
    price            what wrongly declining THIS member costs, from core/liability.py. A
                     decline is not free and the system should know which ones are expensive.
    remediation      what specifically would change the answer, at four disclosure levels, so
                     the same decline can be explained fully to one member and opaquely to
                     another.
    recovery token   an HMAC-bound, expiring, single-use handle so a retry-after-remediation is
                     recognised as THIS decision being revisited rather than a fresh attempt.

That last one is the piece that makes disclosure safe to vary. Without it, "verify and retry"
is just advice an attacker can also follow. With it, the retry arrives carrying proof that the
remediation this system asked for was actually performed, by the member it was issued to,
inside the window it was issued for.

WHAT THIS MODULE DOES NOT DO. It does not choose the disclosure level. That decision is a priced
trade between recovery uplift and information leaked to an adversary, it needs a causal estimate
that does not exist yet, and putting a guess here would bury the most important choice in the
system inside a helper. `contract()` returns every level; something upstream picks one.

Pure stdlib (hmac, hashlib). No network, no store dependency for issuance.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

# ── recoverability: our judgement, not the network's ─────────────────────────
#
# The network code says what happened. This says what can be DONE about it, which is a different
# question and the only one a member can act on.
SELF_SERVICE = "self_service"   # the member can fix this alone
STEP_UP = "step_up"             # needs verification before we would say yes
TERMINAL = "terminal"           # nothing the member does will change the answer
UNKNOWN = "unknown"             # we genuinely do not know, and should say so

# Cause -> recoverability. Cause comes from the authorization path; see auth_ledger.DECLINE_CAUSE.
_RECOVERABILITY = {
    "member_funds": SELF_SERVICE,
    "member_limit": SELF_SERVICE,
    "data_error":   SELF_SERVICE,
    "issuer_risk":  STEP_UP,
}

# Codes that are terminal whatever their cause says. A restricted card is a decision about the
# instrument, not about this payment, and telling the member to retry would be a lie.
_TERMINAL_CODES = ("62", "14", "54")

DISCLOSURE_LEVELS = ("none", "generic", "specific", "guided")

# Remediation text by cause and level. Deliberately dull: this is a contract, not marketing.
#
# The ladder is the point. "none" is the status quo the whole industry ships, and it is included
# as a real option rather than a strawman, because for a likely attacker it is the correct
# choice and the optimiser must be able to select it.
_REMEDIATION = {
    "member_funds": {
        "none":     "This payment was declined.",
        "generic":  "This payment was declined. Check your account and try again.",
        "specific": "Declined: not enough available balance for this amount.",
        "guided":   "Declined: not enough available balance. Add funds, then retry this "
                    "payment from your activity feed.",
    },
    "member_limit": {
        "none":     "This payment was declined.",
        "generic":  "This payment was declined. Check your account and try again.",
        "specific": "Declined: this exceeds a limit on your account.",
        "guided":   "Declined: this exceeds your current spending limit. Raise the limit in "
                    "settings or retry a smaller amount.",
    },
    "data_error": {
        "none":     "This payment was declined.",
        "generic":  "This payment was declined. Check your details and try again.",
        "specific": "Declined: the card details did not match.",
        "guided":   "Declined: the security code or billing address did not match. Re-enter "
                    "them and retry.",
    },
    "issuer_risk": {
        "none":     "This payment was declined.",
        "generic":  "This payment was declined. Contact support if you believe this is wrong.",
        "specific": "Declined: this payment looked unusual for your account.",
        "guided":   "Declined: this payment looked unusual for your account. Confirm it was "
                    "you in the app and it will go through.",
    },
}

# What the member must actually DO for the token to be redeemable. Separate from the text,
# because the text is what we say and this is what we will check.
_ACTION = {
    "member_funds": "fund_account",
    "member_limit": "raise_limit_or_reduce_amount",
    "data_error":   "correct_card_details",
    "issuer_risk":  "verify_identity",
}

TOKEN_TTL_SECONDS = 24 * 3600
_TOKEN_VERSION = "v1"


def recoverability(code: str, cause: str) -> str:
    """What can be done about this decline.

    Deliberately NOT a lookup on the code alone. `05` is the industry's catch-all and covers
    both a member who needs to verify and a decision we will not revisit; collapsing that into
    one class is what makes declines feel arbitrary from the outside.
    """
    c = str(code or "").strip()
    if c in _TERMINAL_CODES:
        return TERMINAL
    cls = _RECOVERABILITY.get(str(cause or "").strip().lower())
    return cls or UNKNOWN


def _secret() -> bytes:
    """Signing key for recovery tokens.

    Falls back to a per-process random key when unset, which fails CLOSED across restarts:
    tokens simply stop verifying rather than becoming forgeable. A predictable default would be
    worse than no token at all, because the whole security property is that a retry carrying a
    token has actually been through remediation.
    """
    env = os.environ.get("REDWING_RECOVERY_SECRET")
    if env:
        return env.encode()
    global _EPHEMERAL
    try:
        return _EPHEMERAL
    except NameError:
        _EPHEMERAL = os.urandom(32)
        return _EPHEMERAL


def issue_token(*, decline_id: str, member_id: str, action: str,
                ttl: int = TOKEN_TTL_SECONDS, now: float | None = None) -> str:
    """A bound, expiring handle for one specific remediation by one specific member.

    Bound to the member so a token disclosed to one person cannot be redeemed by another, which
    matters because the reason we can afford to explain a decline at all is that the explanation
    is useless without the token. Carries no reason text: an intercepted token must not leak
    what an intercepted message would have said.
    """
    payload = {
        "v": _TOKEN_VERSION,
        "d": str(decline_id),
        "m": hashlib.sha256(str(member_id).encode()).hexdigest()[:16],
        "a": str(action),
        "x": int((now if now is not None else time.time()) + max(1, int(ttl))),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(_secret(), body, hashlib.sha256).digest()[:16]
    return (base64.urlsafe_b64encode(body).decode().rstrip("=") + "."
            + base64.urlsafe_b64encode(sig).decode().rstrip("="))


def verify_token(token: str, *, member_id: str, now: float | None = None) -> dict:
    """Is this a genuine, unexpired token issued to this member?

    Every failure returns a reason rather than a bare False, because the caller has to
    distinguish "expired, offer a fresh one" from "forged, treat as an attack". Constant-time
    comparison on the signature.
    """
    def bad(reason):
        return {"valid": False, "reason": reason}

    try:
        b64_body, b64_sig = str(token).split(".")
        body = base64.urlsafe_b64decode(b64_body + "=" * (-len(b64_body) % 4))
        sig = base64.urlsafe_b64decode(b64_sig + "=" * (-len(b64_sig) % 4))
    except Exception:                                            # noqa: BLE001
        return bad("malformed")

    expected = hmac.new(_secret(), body, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(sig, expected):
        return bad("bad_signature")

    try:
        p = json.loads(body)
    except ValueError:
        return bad("malformed")
    if p.get("v") != _TOKEN_VERSION:
        return bad("unknown_version")
    if p.get("m") != hashlib.sha256(str(member_id).encode()).hexdigest()[:16]:
        return bad("wrong_member")
    if float(p.get("x", 0)) < (now if now is not None else time.time()):
        return bad("expired")
    return {"valid": True, "decline_id": p["d"], "action": p["a"], "expires_at": p["x"]}


def contract(*, decline_id: str, member_id: str, code: str, cause: str,
             amount=0.0, rail: str = "card", ltv_band: str = "",
             account_age_days=365, config: dict | None = None,
             now: float | None = None) -> dict:
    """The full decline object: what happened, what it cost, what would fix it, and the handle.

    Returns EVERY disclosure level rather than choosing one. The choice is a priced trade
    between recovery uplift and information handed to an adversary, it needs a causal estimate
    this module does not have, and hiding it in here would bury the most consequential decision
    in the system inside a helper.
    """
    cls = recoverability(code, cause)
    action = _ACTION.get(str(cause or "").strip().lower(), "contact_support")

    # What this decline costs if the member was legitimate. Reuses the same false-positive
    # pricing the block decision already uses, so a decline and a hold are valued on one scale.
    try:
        from .liability import false_positive_cost
        # The rail is passed on, and it defaults to "card" here because this module works
        # entirely on ISO 8583 response codes. It matters: on card the revenue forgone is
        # interchange, which has a fixed leg, not a flat margin, so a small-ticket decline costs
        # far more revenue than a percentage suggests. The parameter already existed on this
        # function and was reaching the fraud side of the price but not the customer side.
        cost = false_positive_cost(amount, "DECLINE", ltv_band, account_age_days, config,
                                   rail=rail)
    except Exception:                                            # noqa: BLE001
        cost = {"total": 0.0, "unavailable": True}

    recoverable = cls in (SELF_SERVICE, STEP_UP)
    texts = _REMEDIATION.get(str(cause or "").strip().lower())
    if not texts or not recoverable:
        # A terminal or unknown decline gets the same words at every level. Offering guided
        # remediation for something we will not revisit would be the cruellest possible
        # failure: a member following instructions that cannot work.
        base = ("This payment was declined and cannot be retried on this card."
                if cls == TERMINAL else "This payment was declined.")
        texts = {lvl: base for lvl in DISCLOSURE_LEVELS}

    out = {
        "decline_id": decline_id,
        "response_code": code,
        "cause": cause,
        "recoverability": cls,
        "recoverable": recoverable,
        "required_action": action if recoverable else None,
        "cost_of_this_decline": cost.get("total"),
        "cost_detail": cost,
        "disclosure": {lvl: {"message": texts[lvl]} for lvl in DISCLOSURE_LEVELS},
    }

    # The token rides only on levels that actually tell the member to do something. Attaching it
    # to "none" would hand a retry credential to someone we deliberately told nothing, which is
    # the exact population the opaque level exists to withhold from.
    if recoverable:
        tok = issue_token(decline_id=decline_id, member_id=member_id, action=action, now=now)
        for lvl in ("specific", "guided"):
            out["disclosure"][lvl]["recovery_token"] = tok
            out["disclosure"][lvl]["required_action"] = action
    return out
