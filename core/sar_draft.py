"""
core/sar_draft.py - draft a SAR narrative from the case file, and refuse to file one that
says anything the case file does not.

WHY THIS EXISTS. Narrative drafting is described as the single heaviest manual burden in AML
compliance (~4.7M SARs filed in the US in FY2024), and it is the obvious thing to hand to a
language model. Every serious published architecture converges on the same shape: agent-drafted,
human-attested, never a bare generative call. The risk they all name is the same too, a model
inventing a fact that reads perfectly and is not true, in a document filed with a regulator
under a named person's signature.

THE PIECE THAT IS ACTUALLY LOAD-BEARING is not the drafting. It is the check between drafting
and filing, and this module is built around it:

    check_grounding(text, facts) -> every checkable claim in the narrative appears in the case

It is deliberately DRAFTER-AGNOSTIC. The narrative can come from draft_narrative() below, from
an LLM, or from an analyst typing freehand; the gate is identical and runs before a human is
asked to sign anything. That is what makes "never a bare generative call" a property of the
system rather than a promise about how it will be used.

WHAT IT CHECKS, and what it deliberately does not. A hallucination in a SAR is almost never a
stylistic problem, it is an invented NUMBER or an invented IDENTIFIER: an amount that was never
moved, a date the account was not opened, a payee that does not appear anywhere in the case. So
the validator extracts money, percentages, identifiers and material integers, and requires each
to appear in the structured case. It does not grade prose, judge tone, or try to verify
narrative claims like "the customer appeared to be under duress", because a deterministic
checker cannot and pretending otherwise would be worse than not checking.

Filing therefore requires three things, and the endpoint enforces all three:
  1. a narrative that passes the grounding check
  2. a named human attester
  3. the digest of the exact text they attested to, so a draft cannot be edited after sign-off

Pure stdlib, deterministic, unit-testable. No model, no key, no network.
"""

from __future__ import annotations

import hashlib
import re

# Numbers below this are ordinals, stage numbers and counts that appear in ordinary prose
# ("stage 3", "the second payment"). Requiring them to be grounded produces noise, not safety.
MATERIAL_INT = 100

_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_IDENT = re.compile(r"\b((?:recv|user|txn|tx|acct|recipient)[_:][A-Za-z0-9_-]+)\b", re.I)
_INT = re.compile(r"(?<![\w.$])(\d{2,}(?:,\d{3})*(?:\.\d+)?)(?![\w%])")
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _norm_num(s: str) -> str:
    """Canonical form so '$1,200.00', '1200', and 1200.0 all compare equal."""
    try:
        f = float(str(s).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return str(s).strip().lower()
    return f"{f:.4f}".rstrip("0").rstrip(".")


def fact_set(case: dict) -> set:
    """Every value in the case file, flattened and normalised, as the ground truth a narrative
    is allowed to draw on. Walks nested dicts and lists because the case file is deeply nested
    and a fact buried three levels down is still a fact."""
    out: set = set()

    def walk(v):
        if isinstance(v, dict):
            for k, vv in v.items():
                out.add(str(k).strip().lower())
                walk(vv)
        elif isinstance(v, (list, tuple, set)):
            for vv in v:
                walk(vv)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, (int, float)):
            out.add(_norm_num(v))
        else:
            s = str(v).strip()
            if s:
                out.add(s.lower())
                out.add(_norm_num(s))
                for d in _ISO_DATE.findall(s):
                    out.add(d)
    walk(case)
    return {x for x in out if x}


def claims_in(text: str) -> list:
    """The checkable claims a narrative makes: money, percentages, identifiers, dates, and
    integers large enough to be material. Returns (kind, raw, normalised).

    The specific patterns run FIRST and their spans are masked out before the bare-integer
    sweep, because otherwise the sweep re-reads fragments of what they already matched: the
    "200" inside "$4,200" and the "2024" inside "2024-03-02" both looked like unsupported
    standalone numbers, which made the module's own draft fail its own gate. A validator that
    cries wolf on correct text is worse than none, because the first thing anyone does with it
    is turn it off.
    """
    text = text or ""
    found = []
    masked = list(text)

    def take(rx, kind, norm):
        for m in rx.finditer(text):
            raw = m.group(1)
            found.append((kind, raw, norm(raw)))
            for i in range(m.start(), m.end()):
                masked[i] = " "

    take(_MONEY, "money", _norm_num)
    take(_PCT, "percent", _norm_num)
    take(_IDENT, "identifier", lambda r: r.strip().lower())
    take(_ISO_DATE, "date", lambda r: r)

    for m in _INT.finditer("".join(masked)):
        n = _norm_num(m.group(1))
        try:
            if float(n) < MATERIAL_INT:
                continue
        except ValueError:
            continue
        found.append(("number", m.group(1), n))
    return found


def check_grounding(text: str, case: dict) -> dict:
    """Does every checkable claim in `text` appear in `case`?

    This is the gate. An unsupported claim is a REFUSAL, not a warning: a filing that says
    something the case file does not is exactly the failure this module exists to prevent, and
    a warning that can be clicked through is not a control.
    """
    facts = fact_set(case)
    claims = claims_in(text)
    unsupported = []
    for kind, raw, norm in claims:
        if norm in facts:
            continue
        # money and counts often appear in the case as a differently formatted string
        if any(norm == _norm_num(f) for f in facts if f):
            continue
        unsupported.append({"kind": kind, "claim": raw})
    return {
        "grounded": not unsupported,
        "checked": len(claims),
        "unsupported": unsupported,
        "narrative_sha": narrative_sha(text),
    }


def narrative_sha(text: str) -> str:
    """Digest of the exact text. Filing binds the attestation to this, so a draft cannot be
    edited between sign-off and submission."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def draft_narrative(case: dict) -> dict:
    """A deterministic first draft from the case file's own fields.

    Deliberately plain and a little dry. It exists so the grounding gate has something to
    validate with no model present, and so the baseline draft is free; an LLM can produce a
    better-written one and passes through exactly the same check.
    """
    tx = case.get("transaction", {}) or {}
    subj = case.get("subject", {}) or case.get("customer", {}) or {}
    disp = case.get("disposition", {}) or {}
    typ = (case.get("typology") or tx.get("fraud_typology") or "unknown").replace("_", " ")

    amount = tx.get("amount")
    rail = tx.get("payment_rail") or tx.get("rail") or "unknown rail"
    rid = tx.get("recipient_id") or "the beneficiary"
    uid = tx.get("user_id") or subj.get("user_id") or "the customer"
    txid = tx.get("transaction_id") or case.get("transaction_id") or "the transaction"

    money = f"${float(amount):,.0f}" if isinstance(amount, (int, float)) else "an amount"
    lines = [
        f"Subject {uid} initiated a payment of {money} over {rail} to beneficiary {rid} "
        f"(reference {txid}).",
        f"The pattern is consistent with {typ}.",
    ]
    reasons = disp.get("reasons") or case.get("reason_codes") or []
    if reasons:
        lines.append("Contributing signals: " + ", ".join(str(r) for r in reasons[:6]) + ".")
    action = disp.get("action") or disp.get("recommended")
    if action:
        lines.append(f"Recommended disposition at the time of review: {action}.")
    lines.append(
        "This narrative was assembled from structured case data and has not been reviewed. "
        "It requires attestation by a named investigator before filing.")
    text = " ".join(lines)
    return {"narrative": text, "narrative_sha": narrative_sha(text),
            "grounding": check_grounding(text, case), "drafted_by": "deterministic"}
