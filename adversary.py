"""
adversary.py - the cheap-vs-costly adversary simulator. This is REDWING's canonical
adversarial surface: it red-teams the live model directly, in-process.

Takes a seed fraud the model currently CATCHES, applies NAMED 2026 evasion moves each
tagged by adversary COST, re-scores every variant against the LIVE model, and measures
DETECTION DECAY.

The thesis it tests (and proves or refutes per-case, rather than asserting): most of a
fraud model's detection rests on signals an adversary controls for FREE - the amount
shape, the rail, the hour, round-number avoidance. Those are DEPRECIATING signals. A
detector that survives only because the adversary was sloppy is not a detector. The
moves that SHOULD be required to evade are COSTLY and provenance-backed: a clean
established mule, an aged account, throttled velocity - signals that APPRECIATE because
faking them costs real resources. If cheap moves alone defeat the model, that is the
finding. If the model only falls to costly moves, it is resilient (the economic win).

Operates on the feature vector and re-scores via the operator's own scaler+model, so it
measures the model exactly as it serves. Score is P(fraud)*100.
"""

from __future__ import annotations

# ── Strategy registry ─────────────────────────────────────────────────────────
# Each move sets the features it would change to their evaded (low-risk) values.
STRATEGIES = [
    # CHEAP - the adversary controls these for free, per transaction.
    {"id": "match_amount_baseline", "label": "Shape amount to the victim's baseline",
     "cost": "cheap",
     "rationale": "Send an amount near the account's own average. Defeats amount_zscore, amount_vs_max, is_new_maximum - the single heaviest feature cluster.",
     "sets": {"amount_zscore": 0.0, "amount_vs_max": 0.3, "is_new_maximum": 0.0}},
    {"id": "use_preferred_rail", "label": "Use the victim's usual rail",
     "cost": "cheap",
     "rationale": "Move on the account's normal rail instead of a high-risk one. Defeats preferred_rail_deviation, rail_risk.",
     "sets": {"preferred_rail_deviation": 0.0, "rail_risk": 0.15, "is_crypto": 0.0, "is_instant_rail": 0.0}},
    {"id": "normal_hour", "label": "Transact during normal hours",
     "cost": "cheap",
     "rationale": "Fire the transaction in a low-risk hour band. Defeats hour_risk.",
     "sets": {"hour_risk": 0.1}},
    {"id": "avoid_round_amounts", "label": "Avoid round / threshold amounts",
     "cost": "cheap",
     "rationale": "Use $1,847.33 not $2,000, and stay clear of reporting thresholds. Defeats is_round_amount, amount_just_below_threshold.",
     "sets": {"is_round_amount": 0.0, "amount_just_below_threshold": 0.0}},

    # COSTLY - these require real resources the adversary must invest in.
    {"id": "throttle_velocity", "label": "Throttle velocity (slow and low)",
     "cost": "costly",
     "rationale": "Spread activity over days instead of bursting. Costs the adversary TIME and parallelism. Defeats the velocity cluster.",
     "sets": {"velocity_1h": 0.0, "velocity_4h": 0.0, "velocity_24h": 0.0,
              "velocity_7d": 0.05, "velocity_30d": 0.05, "inter_tx_time_short": 0.0,
              "new_recipient_streak": 0.0}},
    {"id": "established_recipient", "label": "Route through an established payee",
     "cost": "costly",
     "rationale": "Cash out to a recipient with real history and no fraud reputation. Costs a CLEAN mule the adversary had to recruit and age. Defeats recipient_familiarity, recipient_global_fraud_rate, is_new_recipient.",
     "sets": {"recipient_familiarity": 1.0, "recipient_global_fraud_rate": 0.0, "is_new_recipient": 0.0}},
    {"id": "aged_account", "label": "Use an aged account",
     "cost": "costly",
     "rationale": "Operate from an account with months of tenure. Costs the adversary PATIENCE (or a high-value ATO target). Defeats account_age_days.",
     "sets": {"account_age_days": 2200.0}},
    {"id": "recognised_device", "label": "Use a recognised device",
     "cost": "costly",
     "rationale": "Act from a device the account has used before. Costs real device compromise, not a throwaway emulator. Defeats device_familiarity.",
     "sets": {"device_familiarity": 1.0}},
]

COST_ORDER = {"cheap": 0, "costly": 1}
ACTION_THRESHOLD = 0.45   # below this the model would no longer flag (agent flag floor)


def strategies() -> list:
    """The evasion registry - id, label, cost, rationale."""
    return [{k: s[k] for k in ("id", "label", "cost", "rationale")} for s in STRATEGIES]


def _apply(features: dict, sets: dict) -> dict:
    f = dict(features)
    f.update(sets)
    return f


def simulate(features: dict, score_fn, threshold: float = ACTION_THRESHOLD) -> dict:
    """Run the full adversary sweep on one seed.
    score_fn(feature_dict) -> P(fraud) in [0,1]."""
    base = float(score_fn(features)) * 100.0
    by_id = {s["id"]: s for s in STRATEGIES}

    # 1. Per-strategy ABLATION - each move applied alone, measured against the seed.
    ablation = []
    for s in STRATEGIES:
        sc = float(score_fn(_apply(features, s["sets"]))) * 100.0
        ablation.append({
            "id": s["id"], "label": s["label"], "cost": s["cost"],
            "score": round(sc, 1), "drop": round(base - sc, 1),
            "rationale": s["rationale"],
        })

    # 2. CHEAPEST-FIRST cumulative decay - stack moves in ascending cost.
    ordered = sorted(STRATEGIES, key=lambda s: (COST_ORDER[s["cost"]], s["id"]))
    curve = [{"step": 0, "move": "seed (caught)", "cost": "-", "score": round(base, 1),
              "actioned": base >= threshold * 100}]
    acc = dict(features)
    crossed_at = None
    cheap_only_score = None
    for i, s in enumerate(ordered, 1):
        acc = _apply(acc, s["sets"])
        sc = float(score_fn(acc)) * 100.0
        actioned = sc >= threshold * 100
        curve.append({"step": i, "move": s["label"], "cost": s["cost"],
                      "score": round(sc, 1), "actioned": actioned})
        if crossed_at is None and not actioned:
            crossed_at = {"step": i, "move": s["label"], "cost": s["cost"]}
        if s["cost"] == "cheap":
            cheap_only_score = sc

    # 3. Verdict.
    cheap_only_score = cheap_only_score if cheap_only_score is not None else base
    lost_to_cheap = base - cheap_only_score
    share_cheap = (lost_to_cheap / base) if base > 0 else 0.0
    defeated_by_cheap = cheap_only_score < threshold * 100
    if defeated_by_cheap:
        verdict = "FRAGILE"
        headline = (f"Free moves alone drop detection {base:.0f} -> {cheap_only_score:.0f} "
                    f"({share_cheap*100:.0f}% of the score), below the action threshold. "
                    f"The model was catching a sloppy adversary, not the fraud.")
    elif crossed_at and crossed_at["cost"] == "costly":
        verdict = "RESILIENT"
        headline = (f"Cheap moves erode {share_cheap*100:.0f}% of the score but do NOT defeat "
                    f"detection; only a COSTLY, provenance-backed move ({crossed_at['move']}) "
                    f"crosses the threshold. Evasion costs real resources - the economic win.")
    else:
        verdict = "PARTIAL"
        headline = (f"Cheap moves remove {share_cheap*100:.0f}% of the score; the fraud survives "
                    f"near the threshold. Detection leans on cheap signals but isn't fully free to evade.")

    return {
        "baseline_score": round(base, 1),
        "action_threshold": round(threshold * 100, 1),
        "ablation": sorted(ablation, key=lambda a: -a["drop"]),
        "decay_curve": curve,
        "verdict": verdict,
        "cheap_only_score": round(cheap_only_score, 1),
        "share_lost_to_cheap": round(share_cheap, 3),
        "crossed_at": crossed_at,
        "headline": headline,
    }
