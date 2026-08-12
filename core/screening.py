"""
core/screening.py - the gate that runs before the model, because it is not a risk opinion.

WHY THIS SITS AHEAD OF EVERYTHING ELSE. Every other layer in this system weighs something
against something: the model weighs evidence, `liability.py` weighs a loss against a wrongly
blocked customer, `decision_policy.py` weighs an action against the institution's appetite.
Sanctions screening weighs nothing. A payment to a designated party cannot be approved at any
fraud score, under any reimbursement posture, past any policy ceiling. There is no trade, so
there is no place for it inside a pricing model.

WHAT WAS THERE BEFORE. Fifteen agency connectors existed and none of them touched a decision.
`/integrations/enrich` was something a human called during an investigation, after the decision
had already been made. And `case_file.py` produced its sanctions verdict with

    sanctions_hit = r.random() < 0.01

sitting a few lines above a field that reported `"sanctions_screened": True`. That is the single
most dangerous line in the repository, because it asserts a control was applied while applying
nothing at all.

THE FAILURE POSTURE IS INVERTED, DELIBERATELY. Everything else here fails SILENT: the novelty
gate, the consortium view and the actor layer all decline to speak rather than take down the
money path, and that is right for an advisory signal. Screening is the opposite. An unavailable
screening list must NOT mean "approve unscreened", because unscreened approval of a designated
party is the failure that ends banking relationships. So this fails CLOSED, and the way it
avoids turning that into an outage is by screening against a LOCAL list rather than a live API:
the SDN list is published, so screening should never have been an availability dependency.

THREE OUTCOMES, NOT TWO. Real screening is fuzzy, because designated parties use aliases and
transliterations, and common names collide constantly. So:

    clear             no match. Proceed to scoring.
    potential_match   blocks THIS attempt and requires a human. It is not an accusation and
                      most potential matches on common names are not real, but releasing one
                      without review is not a decision an automated system may take.
    confirmed_match   blocked, and it carries a reporting obligation.

NEVER RECOVERABLE. A screening block is TERMINAL in the sense `decline_contract.py` means it:
there is no remediation to offer, and a system that told someone how to get a sanctions match
through would be doing something far worse than losing a sale. The contract's terminal class
already refuses to dress those in guided remediation, which is exactly the behaviour needed.

Pure stdlib. Deterministic. No network.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

CLEAR = "clear"
POTENTIAL = "potential_match"
CONFIRMED = "confirmed_match"
UNAVAILABLE = "unavailable"

# Reason codes carried by a screening block. Deliberately distinct from the fraud decline codes:
# an analyst, an auditor and a member support agent all need to be able to tell "we think this
# is fraud" apart from "we are legally prohibited from processing this".
CODE_SANCTIONS = "SCR01"
CODE_WATCHLIST = "SCR02"
CODE_UNAVAILABLE = "SCR99"

# Where the list lives. A FILE, not an API. The SDN list is published and downloadable, so
# making screening depend on a live call would turn a vendor outage into either an approval of
# unscreened traffic or a total stop, and neither is acceptable.
DEFAULT_LIST = Path(os.environ.get(
    "REDWING_SANCTIONS_LIST",
    Path(__file__).resolve().parent.parent / "data" / "sanctions_list.txt"))

_STOPWORDS = {"the", "and", "of", "co", "ltd", "llc", "inc", "sa", "ag", "gmbh", "plc"}


def normalise(name: str) -> str:
    """Canonical form for comparison.

    Transliteration and punctuation are where naive screening fails: a designated party spelled
    with different diacritics, or with the corporate suffix dropped, is the same party and an
    exact string match would sail past it.
    """
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    parts = [p for p in s.split() if p and p not in _STOPWORDS]
    return " ".join(parts)


def _tokens(name: str) -> set:
    return set(normalise(name).split())


class SanctionsList:
    """A loaded screening list, and whether it can be trusted.

    `available` is the property the gate keys on. A list that failed to load is not an empty
    list, and treating it as one would silently approve everything, which is the exact failure
    this module exists to prevent.
    """

    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_LIST)
        self.entries: list = []
        self.available = False
        self.error = ""
        self._load()

    def _load(self):
        try:
            if not self.path.exists():
                self.error = f"sanctions list not found at {self.path}"
                return
            rows = []
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # "canonical name | alias; alias | programme"
                parts = [p.strip() for p in line.split("|")]
                name = parts[0]
                aliases = [a.strip() for a in (parts[1].split(";") if len(parts) > 1 else [])
                           if a.strip()]
                programme = parts[2] if len(parts) > 2 else "SDN"
                rows.append({"name": name, "aliases": aliases, "programme": programme,
                             "tokens": _tokens(name),
                             "alias_tokens": [_tokens(a) for a in aliases]})
            self.entries = rows
            self.available = bool(rows)
            if not rows:
                self.error = "sanctions list is empty"
        except Exception as e:                                   # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.available = False

    def match(self, name: str) -> dict:
        """Screen one name. Returns the strongest match found.

        Token containment rather than equality, because "Acme Trading Company Limited" and
        "Acme Trading" are the same counterparty and a system that only caught the former would
        be theatre. That deliberately produces false positives on common names, which is why a
        match holds for review instead of accusing anybody.
        """
        q = _tokens(name)
        if not q:
            return {"result": CLEAR}
        best = None
        for e in self.entries:
            for cand, label in [(e["tokens"], e["name"])] + \
                               [(t, a) for t, a in zip(e["alias_tokens"], e["aliases"])]:
                if not cand:
                    continue
                overlap = len(q & cand)
                if overlap == 0:
                    continue
                # exact token set  -> confirmed. containment -> potential.
                if q == cand:
                    return {"result": CONFIRMED, "matched": e["name"], "via": label,
                            "programme": e["programme"], "overlap": overlap}
                if cand.issubset(q) or q.issubset(cand):
                    score = overlap / max(len(q), len(cand))
                    if best is None or score > best["score"]:
                        best = {"result": POTENTIAL, "matched": e["name"], "via": label,
                                "programme": e["programme"], "overlap": overlap,
                                "score": round(score, 3)}
        return best or {"result": CLEAR}


_LIST: SanctionsList | None = None


def get_list(path=None, reload: bool = False) -> SanctionsList:
    global _LIST
    if _LIST is None or reload or path is not None:
        _LIST = SanctionsList(path)
    return _LIST


def screen(*, counterparty: str = "", member: str = "", path=None) -> dict:
    """The gate. Runs BEFORE scoring and can only stop, never approve.

    Returns `blocked` plus the reason. A caller that gets `blocked=True` must not proceed to
    price or score the payment: there is nothing to weigh it against.
    """
    lst = get_list(path)
    if not lst.available:
        # FAIL CLOSED. The rest of this system fails silent, and that is right for advisory
        # signals; approving unscreened traffic because a file was missing is not the same
        # class of mistake as scoring without a novelty view.
        return {"screened": False, "blocked": True, "result": UNAVAILABLE,
                "code": CODE_UNAVAILABLE, "terminal": True,
                "reason": (f"screening is unavailable ({lst.error}), so this payment cannot be "
                           f"approved. Unscreened approval is not a degraded mode, it is the "
                           f"failure the control exists to prevent.")}

    for who, label in ((counterparty, "counterparty"), (member, "member")):
        if not who:
            continue
        m = lst.match(who)
        if m["result"] == CONFIRMED:
            return {"screened": True, "blocked": True, "result": CONFIRMED,
                    "code": CODE_SANCTIONS, "terminal": True, "subject": label,
                    "matched": m["matched"], "programme": m.get("programme"),
                    "reporting_obligation": True,
                    "reason": f"{label} matches a designated party ({m['matched']})"}
        if m["result"] == POTENTIAL:
            return {"screened": True, "blocked": True, "result": POTENTIAL,
                    "code": CODE_WATCHLIST, "terminal": True, "subject": label,
                    "matched": m["matched"], "programme": m.get("programme"),
                    "match_score": m.get("score"), "reporting_obligation": False,
                    "requires_human": True,
                    "reason": (f"{label} is a potential match to {m['matched']}. Most potential "
                               f"matches on common names are not real, and releasing one "
                               f"without review is not a decision an automated system may "
                               f"take.")}
    return {"screened": True, "blocked": False, "result": CLEAR, "terminal": False}


def status() -> dict:
    """Is the control actually in force? The question `sanctions_screened: True` used to answer
    without checking anything."""
    lst = get_list()
    return {"available": lst.available, "entries": len(lst.entries),
            "path": str(lst.path), "error": lst.error or None,
            "fails": "closed"}
