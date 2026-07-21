"""
core/adjudication.py - the contract for turning analyst judgement into gold labels.

The graduation gate waits on four targets. Phase 2 filled exactly one of them, `outcome.is_fraud`,
because an outcome falls out of the ledger and can be replayed at volume. The other three are
INTENT targets, and intent does not fall out of a ledger. Motive, witting role, and scam stage
are things only a human adjudicator can establish, so the actor layer could never graduate: its
heuristics had no human baseline to be measured against, and there was no way for an analyst to
supply one.

Everything needed already existed and was never connected:

  the vocabularies    MOTIVES in core/motive.py, MULE_ROLES in core/mule_network.py, and the
                      arc stages in core/scam_arc.py
  the write path      close_loop() already accepts an `intent` dict and writes each key as
                      analyst ground truth at confidence 0.9
  the reader          graduation.evaluate_target() already compares heuristic against gold

This module is the missing contract between them: one place that says what an analyst may be
asked, sourced from the modules themselves so a vocabulary cannot drift out of sync with the
heuristic it will be compared against.

A NOTE ON THE ONE THING THAT WOULD RUIN THIS. The gate measures heuristic-versus-human
agreement. If the interface pre-selects the machine's guess and the analyst clicks through, the
agreement it measures is manufactured, and the resulting "the rule agrees with humans 94% of
the time" is an artefact of the UI rather than a finding about the rule. So the contract below
carries the heuristic's guess explicitly labelled as a suggestion, and every field is returned
with no default selected. The analyst has to choose. That costs a click and buys the only thing
that makes the measurement worth having.
"""

from __future__ import annotations

from .motive import MOTIVES
from .mule_network import MULE_ROLES
from .scam_arc import SCAM_ARC as _ARC

# Which label_key each vocabulary answers, matching graduation.readiness_report's targets.
INTENT_TARGETS = ("motive", "witting_role", "scam_stage")


def _arc_options() -> list:
    """Scam-arc stages, in narrative order, from the arc definition itself."""
    out = []
    for s in _ARC:
        out.append({
            "value": s["key"],
            "label": s["label"],
            "note": ("money is moving at this stage" if s.get("money_moving")
                     else "no money has moved yet"),
        })
    return out


def _opts(mapping: dict) -> list:
    return [{"value": k, "label": v, "note": ""} for k, v in mapping.items()]


def schema() -> dict:
    """What an analyst can be asked to adjudicate, and why each answer matters.

    Sourced from the modules that produce the competing heuristic, so the options an analyst
    picks from are exactly the classes the heuristic can emit. If they diverged, agreement
    would be unmeasurable for reasons that had nothing to do with the heuristic's quality."""
    return {
        "label_space": "intent",
        "source": "analyst",
        "confidence": 0.9,
        "why": ("Outcome labels arrive from the ledger. Intent labels do not exist anywhere "
                "until a human establishes them, which is why the actor layer cannot graduate "
                "without this step."),
        "no_default_selected": True,
        "no_default_reason": ("The graduation gate measures heuristic-versus-human agreement. "
                              "Pre-selecting the heuristic's guess would manufacture that "
                              "agreement and make the measurement meaningless."),
        "fields": [
            {
                "key": "motive",
                "question": "Why did this person move the money?",
                "matters": ("Drives whether the response protects or penalises. A coerced "
                            "victim and an organised launderer can produce identical "
                            "transactions and require opposite interventions."),
                "options": _opts(MOTIVES),
            },
            {
                "key": "witting_role",
                "question": "How knowing was their participation?",
                "matters": ("The difference between a person who was tricked and one who was "
                            "paid. Getting it wrong penalises a victim or protects a "
                            "professional."),
                "options": _opts(MULE_ROLES),
            },
            {
                "key": "scam_stage",
                "question": "Where in the scam arc was this payment?",
                "matters": ("Interventions are stage-appropriate: educate early, add friction "
                            "in the middle, hard-stop with safeguarding at extraction."),
                "options": _arc_options(),
            },
        ],
    }


def valid_values(field: str) -> set:
    for f in schema()["fields"]:
        if f["key"] == field:
            return {o["value"] for o in f["options"]}
    return set()


def validate(intent: dict) -> tuple:
    """Split a submitted adjudication into (accepted, rejected).

    Rejects unknown keys and out-of-vocabulary values rather than storing them, because a gold
    label the gate cannot compare against is worse than no label: it inflates the gold count
    while contributing nothing, which is exactly how a readiness metric starts lying."""
    accepted, rejected = {}, []
    for k, v in (intent or {}).items():
        key = str(k).strip()
        if key not in INTENT_TARGETS:
            rejected.append({"field": key, "reason": "not an adjudicable intent target"})
            continue
        val = str(v).strip()
        if not val:
            continue                                    # skipped field, not an error
        if val not in valid_values(key):
            rejected.append({"field": key, "value": val,
                             "reason": f"not in the {key} vocabulary the heuristic emits"})
            continue
        accepted[key] = val
    return accepted, rejected
