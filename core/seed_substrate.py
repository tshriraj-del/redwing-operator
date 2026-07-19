"""
core/seed_substrate.py - a SYNTHETIC labeled cohort for exercising the training pipeline.

The training and graduation machinery (core/train.py, core/graduation.py) cannot be
demonstrated until the substrate holds labelled data, and no real adjudicated labels exist yet.
This seeder populates the substrate with a clearly-synthetic cohort: point-in-time decisions,
each carrying the module's heuristic self-label AND a later adjudicated gold label.

The scenario is deliberately constructed to be the exact situation graduation exists to detect:
the gold (true) motive depends on TWO signals, but the hand-rule looks at only one, so the rule
is systematically wrong on the subpopulation where only the second signal fires. A model that
sees both features can recover those cases and beat the rule. That is the whole point: the
heuristic bootstraps the labels, and the model then learns the interaction the rule missed.

NOTHING HERE IS REAL DATA. These are fabricated examples to prove the mechanism end to end. Do
not report any accuracy computed on this cohort as a fraud metric.
"""

from __future__ import annotations

import random

from .loop import close_loop, record_decision


def seed_labeled_cohort(store, n: int = 200, seed: int = 7) -> dict:
    """Seed `n` synthetic, fully-labelled cases. Deterministic for a given seed."""
    rng = random.Random(seed)
    for i in range(n):
        tid = f"seed_{seed}_{i}"
        survival_spend = rng.randint(0, 1)
        benefit_timing = rng.randint(0, 1)
        professional = 1 if rng.random() < 0.3 else 0

        # GOLD (adjudicated) truth depends on BOTH survival_spend and benefit_timing.
        if professional:
            gold = "income_source"
        elif survival_spend or benefit_timing:
            gold = "survival"
        else:
            gold = "opportunistic"

        # The HEURISTIC rule ignores benefit_timing, so it mislabels the benefit-timing-only
        # survival cases as opportunistic. This is the gap a trained model should close.
        if professional:
            heur = "income_source"
        elif survival_spend:
            heur = "survival"
        else:
            heur = "opportunistic"

        record_decision(
            store, tid, module="motive",
            features={"survival_spend": survival_spend, "benefit_timing": benefit_timing,
                      "professional": professional},
            heuristic_labels=[{"space": "intent", "key": "motive",
                               "value": heur, "confidence": 0.3}],
            decision_id=f"dec:{tid}",
        )
        # analyst adjudicates the gold motive (and a coarse outcome) for this synthetic case
        close_loop(store, tid, f"r{i % 20}", "adjudicate_synthetic",
                   is_fraud=(gold != "opportunistic"), intent={"motive": gold})

    return {"seeded": n, "synthetic": True,
            "note": "fabricated cohort for pipeline demonstration; not real fraud data"}
