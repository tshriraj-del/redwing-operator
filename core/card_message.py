"""
core/card_message.py - the authorization message, normalised, with its gaps named.

WHY THIS EXISTS. `core/ingest_schema.py` is the canonical event contract and it carries no card
fields at all. It knows "card" as a RAIL VALUE and nothing about the message: no BIN, no entry
mode, no AVS, no CVV, no 3DS outcome, no MCC, no token status, no acquirer. Card fields rode
through as untyped passthrough and nothing downstream read them, which is how `/authorize` ended
up scoring authorizations with the push-payment model for its entire life.

WHAT A MISSING FIELD MEANS, AND WHY IT IS THE POINT OF THE MODULE.

An authorization arrives with whatever the acquirer sent. AVS and CVV are absent on a chip
transaction because they do not exist there, and absent on an e-commerce transaction because the
merchant did not collect them. Those are opposite situations wearing the same null, and a scorer
that cannot tell them apart is guessing.

So normalisation here is not tidying. Every field records whether it was PRESENT, ABSENT BY
NATURE (the message form cannot carry it), or ABSENT AND EXPECTED (it should have been there and
was not). The third case is a data-quality defect on the acquiring side and it is the one worth
alerting on, because a merchant whose AVS suddenly stops arriving has changed something.

DEFAULTS ARE NEVER OPTIMISTIC. An unknown AVS result becomes `not_provided`, never `match`.
Filling a gap with the benign value is how a blind spot turns into an approval, and it is the
same failure mode as scoring a card with a model that returns 0.0: the output looks like a clean
result and is actually an absence of information.

Pure stdlib. No ML stack, so this stays testable alongside the rest of core/.
"""

from __future__ import annotations

# The message as an issuer holds it at authorization time. This is the contract card_model.py
# trains against, so the two must not drift; the trainer's own featurise() is imported at serve
# time rather than reimplemented, and this produces the row that function reads.
CATEGORICAL = ("entry_mode", "channel", "card_type", "avs_result", "cvv_result", "three_ds")
NUMERIC = ("amount", "mcc_code", "tokenized")
IDENTIFIERS = ("bin", "merchant_id")

# Card-present forms genuinely cannot carry AVS or CVV: there is no address and no typed CVV in
# a chip conversation. Absence there is correct, not a defect, and must not be reported as one.
# Public (not underscore-prefixed): core/ingest_schema.py documents this set in its self-served
# contract, so callers outside this module have a real reason to read it.
CARD_PRESENT_ENTRY_MODES = ("chip", "contactless", "magstripe", "swipe", "fallback")

# Values that mean "we did not learn anything", chosen so a gap can never read as a pass.
_UNKNOWN = {
    "entry_mode": "unknown",
    "channel": "unknown",
    "card_type": "unknown",
    "avs_result": "not_provided",
    "cvv_result": "not_provided",
    "three_ds": "not_attempted",
}

# Field aliases, because acquirers and processors do not agree on names. Kept explicit rather
# than fuzzy-matched: a silent mismatch on `avs` versus `avs_result` is exactly the kind of
# error that produces a plausible score from an empty message.
_ALIASES = {
    "entry_mode": ("entry_mode", "pos_entry_mode", "entry", "pan_entry_mode"),
    "channel": ("channel", "transaction_channel", "pos_channel"),
    "card_type": ("card_type", "funding_source", "product_type"),
    "avs_result": ("avs_result", "avs", "avs_response", "address_verification_result"),
    "cvv_result": ("cvv_result", "cvv", "cvv2_result", "cvc_result", "csc_result"),
    "three_ds": ("three_ds", "3ds", "threeds", "three_ds_outcome", "eci_outcome"),
    "bin": ("bin", "card_bin", "iin", "pan_prefix"),
    "merchant_id": ("merchant_id", "card_acceptor_id", "merchant", "mid"),
    "mcc_code": ("mcc_code", "mcc", "merchant_category_code"),
    "amount": ("amount", "transaction_amount", "amt"),
    "tokenized": ("tokenized", "is_tokenized", "token_present", "pan_is_token"),
}

PRESENT = "present"
ABSENT_BY_NATURE = "absent_by_nature"
ABSENT_EXPECTED = "absent_expected"


def _first(msg: dict, field: str):
    for key in _ALIASES.get(field, (field,)):
        if key in msg and msg[key] not in (None, ""):
            return msg[key]
    return None


def _entry_mode(msg: dict) -> str:
    v = _first(msg, "entry_mode")
    return str(v).strip().lower() if v is not None else ""


def is_card_present(entry_mode: str) -> bool:
    return str(entry_mode).strip().lower() in CARD_PRESENT_ENTRY_MODES


def normalise(msg: dict) -> dict:
    """One authorization message onto the canonical card field set.

    Returns `{"row": ..., "presence": ..., "missing_expected": [...], "complete": bool}`.
    The row is what a scorer reads; the rest is what an operator needs in order to know how much
    the score is worth.
    """
    msg = msg if isinstance(msg, dict) else {}
    entry = _entry_mode(msg)
    cp = is_card_present(entry)

    row: dict = {}
    presence: dict = {}

    for f in CATEGORICAL:
        v = _first(msg, f)
        if v is not None:
            row[f] = str(v).strip().lower()
            presence[f] = PRESENT
            continue
        row[f] = _UNKNOWN[f]
        # AVS, CVV and 3DS cannot exist on a card-present message. Reporting them as defects
        # would bury the real ones under noise from every chip transaction in the book.
        presence[f] = (ABSENT_BY_NATURE if (cp and f in ("avs_result", "cvv_result", "three_ds"))
                       else ABSENT_EXPECTED)

    for f in IDENTIFIERS:
        v = _first(msg, f)
        row[f] = str(v).strip() if v is not None else ""
        presence[f] = PRESENT if v is not None else ABSENT_EXPECTED

    amt = _first(msg, "amount")
    try:
        row["amount"] = max(0.0, float(amt))
        presence["amount"] = PRESENT
    except (TypeError, ValueError):
        # Amount is the one field with no honest default. Zero is a real amount and would price
        # the decision at nothing, so it is recorded as missing and the caller decides.
        row["amount"] = 0.0
        presence["amount"] = ABSENT_EXPECTED

    mcc = _first(msg, "mcc_code")
    try:
        row["mcc_code"] = int(float(mcc))
        presence["mcc_code"] = PRESENT
    except (TypeError, ValueError):
        row["mcc_code"] = 0
        presence["mcc_code"] = ABSENT_EXPECTED

    tok = _first(msg, "tokenized")
    if tok is None:
        row["tokenized"] = 0
        # A token is a positive assertion. Its absence means "not a token", which is the
        # higher-risk reading, so this is a legitimate default rather than an optimistic one.
        presence["tokenized"] = ABSENT_BY_NATURE
    else:
        row["tokenized"] = 1 if str(tok).strip().lower() in ("1", "true", "yes", "y") else 0
        presence["tokenized"] = PRESENT

    missing = sorted(f for f, s in presence.items() if s == ABSENT_EXPECTED)
    return {
        "row": row,
        "presence": presence,
        "missing_expected": missing,
        "complete": not missing,
        "card_present": cp,
    }


def quality(norm: dict) -> dict:
    """How much is this score worth, given what the message actually carried?

    Not a confidence interval. It is the blunt operational fact that a score computed without
    AVS, CVV and 3DS on an e-commerce authorization is reading four of its top five features as
    "not provided", and the response should say so rather than presenting a number as though it
    were the same number.
    """
    missing = list(norm.get("missing_expected") or [])
    # The fields the card model actually leans on, measured: AVS and CVV results and the 3DS
    # outcome are three of its top five features by importance.
    decisive = [f for f in missing if f in ("avs_result", "cvv_result", "three_ds",
                                            "entry_mode", "amount")]
    if not missing:
        grade, note = "complete", "every expected field was present"
    elif decisive:
        grade, note = "degraded", (
            "missing " + ", ".join(decisive) + "; these are among the model's strongest "
            "features and the score is weaker than its value suggests")
    else:
        grade, note = "partial", "missing " + ", ".join(missing)
    return {"grade": grade, "missing_expected": missing, "decisive_missing": decisive,
            "note": note}
