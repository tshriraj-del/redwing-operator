"""
core/holdout.py - the randomized monitored-holdout policy (the clean-ground-truth source).

If we block every case the heuristics flag, we never observe what would have happened, so
the training set is censored by our own decisions: the model only ever learns from the
population we let through, and goes blind to exactly the cases the rules already catch. This
is the selection-bias / counterfactual trap that quietly rots fraud ML.

The fix is a small, deliberate holdout: release a capped fraction of would-be-blocked cases,
monitor them closely, and record the true outcome. Those rows are the only UNCONFOUNDED
ground truth we get, because the outcome was not altered by our intervention.

Three safety rails make this responsible rather than reckless:
  1. A hard LIABILITY CEILING. Never gamble a high-value case for data. Above the ceiling the
     case is always enforced; the holdout only ever samples lower-stakes cases.
  2. A small, capped RATE. A single-digit percentage, not a coin flip.
  3. DETERMINISTIC, auditable sampling. A hash of (salt, subject_ref) decides, so the same
     case always resolves the same way, the decision is reproducible, and it cannot be gamed
     by retrying.

The policy only ever RELEASES (turns a block into a monitored allow). It never turns an
allow into a block, and it never touches victim-protection or safeguarding decisions.

Pure Python, deterministic, unit-testable.
"""

from __future__ import annotations

import hashlib

# The enforcement actions a holdout may convert into a monitored release. Protective and
# allow-like actions are deliberately excluded (never hold back protection to gather data).
_HOLDOUTABLE = ("BLOCK", "DECLINE", "HOLD", "STEP_UP")

DEFAULT_HOLDOUT = {
    "rate": 0.02,             # release at most ~2% of eligible would-be-blocks
    "max_liability": 2000.0,  # never release a case exposing more than this many dollars
    "salt": "redwing-holdout-v1",
}


def _bucket(subject_ref: str, salt: str) -> float:
    """Deterministic uniform value in [0,1) from the subject id. Stable and auditable: the
    same case always lands in the same bucket, so a fraudster cannot retry into a release."""
    h = hashlib.sha256(f"{salt}:{subject_ref}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def holdout_decision(subject_ref: str, proposed_action: str,
                     expected_liability: float = 0.0, config: dict | None = None) -> dict:
    """Decide whether a would-be-blocked case is instead diverted into the monitored holdout.

    Returns a dict with:
      release        True if the case should be released and observed instead of blocked
      enforced_action  the action to actually take (unchanged, unless released -> 'ALLOW')
      holdout        True if this decision is part of the holdout (its outcome is clean truth)
      reason         a short human string explaining the call
    """
    cfg = {**DEFAULT_HOLDOUT, **(config or {})}
    action = str(proposed_action or "").upper()
    liab = float(expected_liability or 0.0)

    # Only enforcement actions are eligible; allow-like and protective actions pass through.
    if action not in _HOLDOUTABLE:
        return {"release": False, "enforced_action": action, "holdout": False,
                "reason": "action not eligible for holdout"}

    # Safety rail 1: the liability ceiling. High-stakes cases are always enforced.
    if liab > cfg["max_liability"]:
        return {"release": False, "enforced_action": action, "holdout": False,
                "reason": f"liability ${liab:.0f} over ceiling ${cfg['max_liability']:.0f}; always enforce"}

    # Safety rails 2 + 3: capped, deterministic sampling.
    b = _bucket(str(subject_ref or ""), cfg["salt"])
    if b < cfg["rate"]:
        return {"release": True, "enforced_action": "ALLOW", "holdout": True,
                "reason": f"holdout release ({cfg['rate']*100:.0f}% monitored sample) for clean ground truth"}

    return {"release": False, "enforced_action": action, "holdout": False,
            "reason": "not sampled into holdout; enforce as proposed"}


def holdout_rationale(decision: dict) -> dict:
    """The metadata to stash on a logged decision so training can identify holdout rows and
    reject-inference can separate observed outcomes (released) from censored ones (blocked)."""
    return {
        "holdout": bool(decision.get("holdout")),
        "released": bool(decision.get("release")),
        "reason": decision.get("reason", ""),
    }
