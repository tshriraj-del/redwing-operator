"""
core/decision_policy.py - turn a priced score into an ACTION, under a versioned policy.

WHAT THIS REPLACES. The live path ended eight possible actions in one boolean:

    proposed = "HOLD" if event["is_alert"] else "ALLOW"

Everything above that line is signal: the model, the pattern matcher, the consortium view, the
novelty gate, and `liability.price_decision()` which already works out which action the money
supports. Below it there was no policy at all, so a $12 card payment and a $40,000 Zelle push
to a three-day-old payee were resolved the same way.

WHAT A POLICY IS, AND IS NOT. It is NOT a second opinion on risk. Adding one here would be
laundering a hunch into a decision the evidence does not support. A policy expresses the
institution's GUARDRAILS around a priced decision:

    floor    the least the institution will do at this risk on this rail, whatever the
             economics say. "We do not silently allow a high-scoring push to a brand-new payee
             at 2am, even when the amount is small enough that blocking looks like a bad trade."
    ceiling  the most it will do without a human. "We hold on this rail; we do not auto-decline,
             because a wrong decline here costs a member their rent."

So the price chooses, and the policy bounds. Between them the action is derived rather than
tuned, and both halves are inspectable.

WHY IT IS VERSIONED, which is not tidiness. `decisions.policy_version` has existed in the schema
since the substrate was built and has never once been written. In the US the compliance target
is not a fixed standard you build to and forget: the CFPB has retreated, state Attorneys General
are setting terms through litigation rather than rulemaking, and the authorized-push question is
being decided case by case. Policy will change often and under pressure, and when an outcome
moves you have to be able to say which policy was live when the decision was made. A hash of the
table, written on every decision, is what makes that answerable.

DE-ESCALATION IS ALLOWED AND ALWAYS RECORDED. Everything else in this codebase composes
escalate-only, and that rule is about EVIDENCE: a clean network view must never talk down a
signal the local book found. A ceiling is a different thing, a statement about what the
institution will do rather than about what it believes, and real policies genuinely contain
them. So a ceiling may soften an action, and when it does the decision carries
`policy_deescalated: true` and the rule that did it. Forbidding it would push the behaviour into
someone's config; recording it keeps it in the audit trail.

Pure stdlib, deterministic, no I/O unless a policy file is passed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Severity ladder. Every comparison in this module is an index into this list, so "harder" and
# "softer" have exactly one definition. MONITOR sits above ALLOW because it lets the money go
# but marks the account for observation, which is a real intermediate an institution uses.
LADDER = ["ALLOW", "MONITOR", "STEP_UP", "HOLD", "BLOCK", "DECLINE"]
_RANK = {a: i for i, a in enumerate(LADDER)}

# Score bands. Named rather than numeric at the call site so a policy row reads like a policy
# rather than like a threshold nobody can defend.
BANDS = (("low", 0.0), ("elevated", 0.35), ("high", 0.65), ("severe", 0.90))

# The default table. Deliberately small: this is a starting posture an institution replaces,
# not a claim about what any real bank should do. Rows are matched most-specific-first.
#
# The rails are split the way US liability splits them. Push rails are irrevocable and the money
# is gone at settlement, so the floor rises early. Card is chargeback-protected and network
# rules constrain what an issuer may do, so the ceiling stays low: an auto-decline on a card
# costs a member at a till and buys little that a hold does not.
DEFAULT_POLICY = {
    "name": "redwing-default-us",
    "notes": "Starting posture for a US book. Replace with the institution's own matrix.",
    "rules": [
        # --- irrevocable push rails -------------------------------------------------
        {"rail": "zelle|fednow|rtp|wire|crypto", "direction": "outbound", "band": "severe",
         "floor": "HOLD", "ceiling": "BLOCK",
         "why": "irrevocable and severe: stop it, but a human confirms the block"},
        {"rail": "zelle|fednow|rtp|wire|crypto", "direction": "outbound", "band": "high",
         "floor": "STEP_UP", "ceiling": "HOLD",
         "why": "irrevocable: never silently allow at this risk, never auto-decline either"},
        {"rail": "zelle|fednow|rtp|wire|crypto", "direction": "outbound", "band": "elevated",
         "tier": "new_account", "floor": "STEP_UP", "ceiling": "HOLD",
         "why": "a new account pushing on an irrevocable rail is the mule-onboarding shape"},

        # --- ACH ---------------------------------------------------------------------
        {"rail": "ach", "direction": "outbound", "band": "severe",
         "floor": "HOLD", "ceiling": "BLOCK",
         "why": "ACH has a return window, so a hold is usually enough to recover"},

        # --- card --------------------------------------------------------------------
        {"rail": "card", "band": "severe", "floor": "STEP_UP", "ceiling": "BLOCK",
         "why": "card loss is largely chargeback-protected; step up before declining"},
        {"rail": "card", "band": "high", "floor": "ALLOW", "ceiling": "STEP_UP",
         "why": "a wrong card decline is felt at the till and buys little over a step-up"},
    ],
    "default": {"floor": "ALLOW", "ceiling": "BLOCK",
                "why": "no specific rule: the price decides, unconstrained"},
}


def band_for(score) -> str:
    """The named band a score falls in."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "low"
    name = "low"
    for label, lo in BANDS:
        if s >= lo:
            name = label
    return name


def policy_version(policy: dict) -> str:
    """A stable hash of the table's CONTENT.

    Content, not a hand-maintained version string, because a hand-maintained one drifts the
    moment somebody edits a row and forgets to bump it, and a decision would then be attributed
    to a policy that never made it. `sort_keys` makes the hash independent of key order so
    reformatting the file does not read as a policy change.
    """
    body = json.dumps({k: v for k, v in (policy or {}).items() if k != "version"},
                      sort_keys=True, separators=(",", ":"))
    return f"{(policy or {}).get('name', 'policy')}@{hashlib.sha256(body.encode()).hexdigest()[:12]}"


def load_policy(path=None) -> dict:
    """The default table, or one from disk. A malformed file falls back to the default and says
    so rather than leaving the live path with no policy at all."""
    if not path:
        return DEFAULT_POLICY
    try:
        loaded = json.loads(Path(path).read_text())
        if not isinstance(loaded.get("rules"), list):
            raise ValueError("policy has no rules list")
        return loaded
    except Exception as e:                                        # noqa: BLE001
        return {**DEFAULT_POLICY,
                "load_error": f"{type(e).__name__}: {e}; fell back to the built-in default"}


def _matches(rule: dict, *, rail: str, direction: str, band: str, tier: str) -> int:
    """Specificity of a rule for this context, or -1 if it does not apply.

    Higher is more specific, so the most specific matching rule wins. A rule that names three
    dimensions beats one that names two, which is the behaviour anyone reading a decision matrix
    expects and the opposite of first-match-wins.
    """
    spec = 0
    for field, value in (("rail", rail), ("direction", direction),
                         ("band", band), ("tier", tier)):
        want = rule.get(field)
        if want in (None, "", "*"):
            continue
        options = [w.strip().lower() for w in str(want).split("|")]
        if str(value).strip().lower() not in options:
            return -1
        spec += 1
    return spec


def decide(priced_action: str, score, *, rail: str = "", direction: str = "outbound",
           tier: str = "", policy: dict | None = None) -> dict:
    """Bound a priced action by policy. Returns the action plus everything that produced it.

    `priced_action` is what `liability.price_decision()` recommended, i.e. what the money
    supports. This never second-guesses that; it only applies the institution's floor and
    ceiling. If the two never disagree, the policy is doing nothing and should say so, which is
    why `bounded_by` is reported even when it is None.
    """
    pol = policy or DEFAULT_POLICY
    band = band_for(score)
    proposed = str(priced_action or "ALLOW").upper()
    if proposed not in _RANK:
        proposed = "ALLOW"

    best, best_spec = pol.get("default", {}), -1
    for rule in pol.get("rules", []):
        spec = _matches(rule, rail=rail, direction=direction, band=band, tier=tier or "")
        if spec > best_spec:
            best, best_spec = rule, spec

    floor = str(best.get("floor", "ALLOW")).upper()
    ceiling = str(best.get("ceiling", "DECLINE")).upper()
    f_rank = _RANK.get(floor, 0)
    c_rank = _RANK.get(ceiling, len(LADDER) - 1)
    if c_rank < f_rank:
        # An inverted rule is a policy authoring error. Honour the floor, because the floor is
        # the safety-side bound, and report it rather than silently picking one.
        c_rank, inverted = f_rank, True
    else:
        inverted = False

    p_rank = _RANK[proposed]
    final_rank = min(max(p_rank, f_rank), c_rank)
    final = LADDER[final_rank]

    return {
        "action": final,
        "priced_action": proposed,
        "band": band,
        "floor": floor,
        "ceiling": ceiling,
        "bounded_by": ("floor" if final_rank > p_rank else
                       "ceiling" if final_rank < p_rank else None),
        "policy_escalated": final_rank > p_rank,
        # Recorded, never silent. A ceiling softening an action is a legitimate institutional
        # choice and also the one an auditor will ask about first.
        "policy_deescalated": final_rank < p_rank,
        "rule": {k: best.get(k) for k in ("rail", "direction", "band", "tier", "why")
                 if best.get(k)},
        "policy_version": policy_version(pol),
        "inverted_rule": inverted,
    }
