"""
core/mule_behaviour.py - the witting-ness tells a system can observe about itself.

WHY THIS EXISTS. core/mule_network.py places an account holder on the witting-ness spectrum
from twelve tells, and the graduation gate wants to compare that heuristic against
human-adjudicated gold before anyone trains a model on it. Measured, the gate could never fire:
`paired_with_heuristic` was 0 for every target and could not rise, because the heuristic never
ran. It never ran because its twelve tells and the session telemetry the scorer collects have
EXACTLY ZERO overlap:

    telemetry gives:  coaching_copresence, pii_pasted, script_reading, hesitation_entropy, ...
    the tells want:   believes_legitimate_job, kept_small_cut, continues_after_warning, ...

Those are different kinds of evidence. Telemetry is what happened in one session. The tells are
dispositional, discovered across an arc, and most of them come from an investigator talking to
a person. Feeding one to the other returns "undetermined" every time, which is what the first
attempt at wiring this measured.

WHAT THIS DERIVES, and the bias it deliberately corrects. Only a few tells are observable
without an investigator, and picking the observable ones naively is a trap: the two easiest to
compute, `many_victim_sources` and `continues_after_warning`, BOTH point at witting or herder.
A heuristic that can only ever see evidence of guilt will find guilt in everyone, and those
labels would then seed a classifier. That is not a modelling inconvenience, it is how a system
launders its own assumptions into training data.

So the warning evidence is derived as a PAIR from one source. The same decision history that
shows somebody carried on after being warned also shows somebody stopped, and `stops_on_warning`
is the strongest exculpatory tell in the table (unwitting 0.7). Deriving one without the other
would be choosing to see only half of what the data says.

Everything interior stays out: believes_legitimate_job, kept_small_cut, keeps_consistent_cut,
launder_language, recruits_others. A model should not be guessing what somebody believed, and
those belong to the adjudication panel where a human records what they actually found.

Pure stdlib. Returns tells in the 0..1 strength form classify_mule expects.
"""

from __future__ import annotations

# Distinct senders at which "receives from many unrelated parties" is fully evidenced. Below it
# the tell fires proportionally; a payee with two senders is not a collector.
FANIN_FULL = 12.0
FANIN_FLOOR = 4        # under this, no tell at all - ordinary payees have a few senders

# Actions the ACCOUNT HOLDER actually experienced as friction. ESCALATE is deliberately absent:
# it routes a case to an analyst and the customer never sees it, so treating it as a warning
# means treating ordinary internal review as something the person defied.
#
# Measured before that exclusion: 396 of 400 accounts read as "witting", because nearly every
# account has an ESCALATE somewhere in its history followed by more activity. That is the
# guilt-bias this module's docstring warns about, arrived at by a different route.
_WARNING_ACTIONS = {"BLOCK", "DECLINE", "HOLD", "STEP_UP"}

# How many separate warnings before "carried on regardless" is the better reading. One warning
# followed by ordinary activity describes almost every customer: somebody step-upped on a
# holiday payment who buys groceries the following week has not defied anything. A REPEATED
# warning is the evidence, so the tell needs a second one.
_REPEAT_WARNINGS = 2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def fanin_tell(sender_labels) -> dict:
    """`many_victim_sources` from the distinct senders paying this account.

    Counts DISTINCT senders, not transactions: one victim groomed over twenty payments is one
    source, and treating it as twenty would turn a pig-butchering victim's own ramp into
    evidence that their payee is a collector."""
    distinct = len({uid for uid, _ in (sender_labels or [])})
    if distinct < FANIN_FLOOR:
        return {}
    return {"many_victim_sources": _clamp01(distinct / FANIN_FULL)}


def warning_tells(decisions) -> dict:
    """`continues_after_warning` OR `stops_on_warning`, from the account's own decision history.

    Derived as a pair on purpose: see the module docstring. Whichever way the evidence points,
    it is the same query, and taking only the incriminating half would be a choice to see half
    the data.
    """
    rows = [d for d in (decisions or []) if getattr(d, "ts", None)]
    if not rows:
        return {}
    rows.sort(key=lambda d: d.ts)
    warnings = [d for d in rows if str(getattr(d, "action", "")).upper() in _WARNING_ACTIONS]
    if not warnings:
        return {}                      # never warned: this evidence does not exist either way

    # The evidence is REPEATED friction, not "was warned once and kept banking". Continuing to
    # transact after a step-up is what everybody does; being warned again, and again, is the
    # thing that distinguishes somebody who did not take the hint.
    if len(warnings) >= _REPEAT_WARNINGS:
        return {"continues_after_warning": _clamp01(len(warnings) / 5.0)}

    # Warned exactly once, and never warned again, though they went on using the account.
    # Held deliberately weak: we cannot tell "took the hint" from "we blocked them and they had
    # no opportunity to repeat it", and only the first reading is exculpatory.
    return {"stops_on_warning": 0.4}


def behavioural_tells(store, entity_id: str, recipient_entity_id: str = "") -> dict:
    """The observable subset of the witting-ness tells, for one account holder.

    Returns {} when nothing is evidenced, which is the correct answer far more often than not.
    classify_mule turns that into "undetermined" rather than a guess.
    """
    tells: dict = {}
    try:
        if recipient_entity_id:
            tells.update(fanin_tell(store.recipient_sender_labels(recipient_entity_id)))
    except Exception:
        pass
    try:
        if entity_id:
            tells.update(warning_tells(store.decisions_for_entity(entity_id)))
    except Exception:
        pass
    return tells


# Tells that can only ever point AT somebody. A read built from these alone is not a finding,
# it is the shape of the evidence we happened to be able to compute.
_INCRIMINATING_ONLY = {"many_victim_sources", "continues_after_warning",
                       "keeps_consistent_cut", "launder_language", "recruits_others",
                       "controls_multiple_accounts", "kept_small_cut", "ignored_red_flags"}


def observable_role(store, entity_id: str, recipient_entity_id: str = "") -> dict:
    """The witting-ness read from observable evidence, WITH A REFUSAL BUILT IN.

    A heuristic that can only see incriminating evidence must not return a verdict. That is not
    caution, it is the difference between a prediction and a foregone conclusion, and this
    module walked into it twice before the rule was written down:

      - deriving only the two easiest tells put 99% of accounts at "witting"
      - fixing that left 68%, because the decisions table turned out to be 66% HOLD, so
        "was warned repeatedly" described almost everybody with a history

    Both times the number looked like a finding about mules and was a fact about our own data.
    So unless at least one exculpatory tell is AVAILABLE to fire, this returns undetermined and
    says why. `many_victim_sources` on its own is a reason to look at an account, not a reason
    to record what its holder intended.
    """
    from .mule_network import classify_mule
    tells = behavioural_tells(store, entity_id, recipient_entity_id)
    if tells and not (set(tells) - _INCRIMINATING_ONLY):
        return {"role": "undetermined",
                "role_label": "Intent not established: only incriminating evidence is derivable",
                "confidence": 0.0, "drivers": [], "scores": {}, "is_victim_adjacent": False,
                "tells_used": sorted(tells), "evidence_basis": "one_directional_refused",
                "why": "every derivable tell points one way, so a verdict would describe the "
                       "evidence we can compute rather than the person"}
    read = classify_mule(tells)
    read["tells_used"] = sorted(tells)
    read["evidence_basis"] = "observable_only"
    return read
