"""
core/active_learning.py - which case should the analyst adjudicate NEXT?

Adjudication is the scarcest resource in the platform. A human can label a handful of cases a
day, the graduation gate needs 50 gold labels and 30 heuristic/gold pairs per target before it
will say anything but "not enough", and there are four targets. Asking the analyst to work the
queue in arrival order wastes most of that effort on cases that teach the system nothing.

This module ranks the unlabelled cases by how much labelling them would actually move the gate.

THE CONSTRAINT THAT ACTUALLY BINDS. The gate needs gold labels AND pairs, and a pair only forms
where a heuristic prediction already exists. A gold label on a case the heuristic never scored
raises gold_labels and leaves paired_with_heuristic untouched, so it cannot contribute to the
agreement measurement the verdict rests on. Candidates are therefore drawn only from cases the
heuristic HAS spoken on: labelling one of those moves both counters at once.

THREE PHASES, because what is useful changes as the substrate fills:

  cold_start   Too few gold labels to train anything, so there is no model whose uncertainty we
               could measure. Ranking by "uncertainty" here would be ranking by noise. Instead
               stratify across the heuristic's classes: you cannot measure agreement on a class
               you have zero examples of, so breadth beats depth until every class is present.
  uncertainty  A model can be trained. Rank by its entropy on each candidate: the cases it is
               least sure about are the ones a human answer most changes.
  satisfied    The target has already cleared the gate. Stop spending human attention on it and
               say so, rather than continuing to rank cases nobody needs.

THE BIAS THIS WOULD OTHERWISE INTRODUCE, and the reason for `explore_frac`. Pure uncertainty
sampling only ever asks about cases near the decision boundary, so the labelled set stops
resembling the population and every metric computed on it quietly becomes a metric about the
boundary. That is the same censoring failure as reject inference, which this platform already
refuses to accept elsewhere (core/holdout.py). So a fixed fraction of every queue is drawn
representatively instead of by uncertainty, chosen by a stable hash so the split is auditable
and a case cannot be retried into a different bucket. It costs some efficiency and buys a
labelled set you can still trust to describe the population.

Pure Python stdlib.
"""

from __future__ import annotations

import hashlib
import math

from .graduation import GOLD_SOURCES, MIN_GOLD, MIN_PAIRED, evaluate_target
from .train import _lr_proba, _lr_train, _prepare, _vec

# Below this many gold labels a trained model's uncertainty is not worth believing, so the
# queue stays in cold start and prioritises class coverage instead.
MIN_GOLD_TO_MODEL = 25

# Share of the queue drawn representatively rather than by uncertainty. Not a tuning knob so
# much as a stance: it is the price of keeping the labelled set describable.
DEFAULT_EXPLORE_FRAC = 0.2

_EXPLORE_SALT = "redwing-active-learning-v1"


def _bucket(subject_ref: str) -> float:
    """Stable uniform value in [0,1) per subject, so explore/exploit assignment is deterministic
    and auditable rather than shuffling on every request."""
    h = hashlib.sha256(f"{_EXPLORE_SALT}:{subject_ref}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _entropy(proba: dict) -> float:
    """Shannon entropy of a class distribution, normalised to [0,1] by log(n_classes) so the
    number is comparable across targets with different vocabulary sizes (motive has 7 classes,
    witting_role has 4)."""
    ps = [p for p in proba.values() if p > 0]
    if len(ps) < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return round(h / math.log(len(proba)), 4)


def candidates(store, label_space: str, label_key: str) -> list:
    """Unlabelled cases the heuristic has already scored, with their feature snapshot.

    Drawn from training_rows(sources=["heuristic"]) because that join already guarantees both a
    heuristic prediction and a decision row with features. Anything already carrying a gold
    label is removed: re-asking a question a human has answered is the one thing that is purely
    wasted effort."""
    gold = {r["subject_ref"] for r in
            store.labels_for_target(label_space, label_key, sources=list(GOLD_SOURCES))
            if r["subject_ref"]}
    seen, out = set(), []
    for r in store.training_rows(label_space, label_key, sources=["heuristic"], limit=1_000_000):
        sr = r.get("subject_ref")
        if not sr or sr in gold or sr in seen or not r.get("features"):
            continue
        seen.add(sr)
        out.append({"subject_ref": sr, "features": r["features"],
                    "heuristic": r["label"], "decision_id": r.get("decision_id", "")})
    return out


def _phase(store, label_space: str, label_key: str) -> tuple:
    """Which phase this target is in, and the gate reading that decided it."""
    gate = evaluate_target(store, label_space, label_key)
    if gate["gold_labels"] >= MIN_GOLD and gate["paired_with_heuristic"] >= MIN_PAIRED:
        return "satisfied", gate
    if gate["gold_labels"] < MIN_GOLD_TO_MODEL:
        return "cold_start", gate
    return "uncertainty", gate


def rank(store, label_space: str, label_key: str, limit: int = 20,
         explore_frac: float = DEFAULT_EXPLORE_FRAC) -> dict:
    """Rank the unlabelled candidates for one target, highest value first.

    Every returned case carries the reason it was chosen, because an analyst asked to answer a
    question deserves to know why this case and not another, and because an unexplained queue is
    one nobody trusts enough to work."""
    phase, gate = _phase(store, label_space, label_key)
    pool = candidates(store, label_space, label_key)
    target = f"{label_space}.{label_key}"

    base = {
        "target": target, "phase": phase,
        "gold_labels": gate["gold_labels"],
        "paired_with_heuristic": gate["paired_with_heuristic"],
        "needs_gold": max(0, MIN_GOLD - gate["gold_labels"]),
        "needs_paired": max(0, MIN_PAIRED - gate["paired_with_heuristic"]),
        "candidates_available": len(pool),
    }

    if phase == "satisfied":
        return {**base, "queue": [],
                "reading": (f"{target} has cleared the gate ({gate['gold_labels']} gold, "
                            f"{gate['paired_with_heuristic']} paired). Spend the analyst's "
                            f"attention on a target that still needs it.")}
    if not pool:
        return {**base, "queue": [],
                "reading": (f"No unlabelled cases carrying a heuristic prediction for {target}. "
                            f"A gold label without one cannot form a pair, so there is nothing "
                            f"here that would move the gate.")}

    # explore slice: representative, not boundary-selected
    explore = [c for c in pool if _bucket(c["subject_ref"]) < explore_frac]
    exploit = [c for c in pool if _bucket(c["subject_ref"]) >= explore_frac]

    if phase == "cold_start":
        # Breadth over depth: round-robin the heuristic's classes so every class gets examples.
        by_class: dict = {}
        for c in exploit:
            by_class.setdefault(c["heuristic"], []).append(c)
        for v in by_class.values():
            v.sort(key=lambda c: c["subject_ref"])
        ordered, classes = [], sorted(by_class)
        while any(by_class[k] for k in classes):
            for k in classes:
                if by_class[k]:
                    c = by_class[k].pop(0)
                    ordered.append({**c, "score": None, "uncertainty": None,
                                    "why": f"class coverage: only {len(by_class[k])+1} unlabelled "
                                           f"'{c['heuristic']}' cases remain to draw from"})
        reading = (f"Cold start: {gate['gold_labels']} gold labels is too few to trust a model's "
                   f"uncertainty, so the queue spreads across the heuristic's classes. You cannot "
                   f"measure agreement on a class with no examples.")
    else:
        # Train on what gold exists, then rank the unlabelled by the model's entropy.
        rows = [r for r in store.training_rows(label_space, label_key,
                                               sources=list(GOLD_SOURCES), limit=1_000_000)
                if r.get("features")]
        rows = list({r["subject_ref"]: r for r in rows}.values())
        keys, medians = _prepare(rows)
        model = _lr_train([_vec(r["features"], keys, medians) for r in rows],
                          [r["label"] for r in rows], sorted({r["label"] for r in rows}))
        scored = []
        for c in exploit:
            u = _entropy(_lr_proba(model, _vec(c["features"], keys, medians)))
            scored.append({**c, "score": u, "uncertainty": u,
                           "why": f"model is uncertain here (entropy {u}); a human answer "
                                  f"changes more than on a case it already understands"})
        scored.sort(key=lambda c: -c["score"])
        ordered = scored
        reading = (f"{gate['gold_labels']} gold labels is enough to train, so the queue is ranked "
                   f"by model uncertainty. {int(explore_frac*100)}% is held back for "
                   f"representative sampling so the labelled set does not collapse onto the "
                   f"decision boundary.")

    # interleave the explore slice so it is genuinely worked, not stranded at the bottom
    every = max(2, int(1 / explore_frac)) if explore_frac > 0 else 0
    queue, ei = [], 0
    for i, c in enumerate(ordered):
        if every and i and i % every == 0 and ei < len(explore):
            e = explore[ei]; ei += 1
            queue.append({**e, "score": None, "uncertainty": None, "selection": "explore",
                          "why": "representative sample, chosen independently of the model so "
                                 "the labelled set keeps describing the population"})
        queue.append({**c, "selection": "exploit"})
        if len(queue) >= limit:
            break

    return {**base, "explore_frac": explore_frac,
            "queue": [{k: v for k, v in q.items() if k != "features"} for q in queue[:limit]],
            "reading": reading}


def next_questions(store, targets=None, limit: int = 10) -> dict:
    """Across all intent targets, where should the next hour of adjudication go?

    Picks the target furthest from clearing the gate rather than round-robining, because a
    target left short of MIN_PAIRED contributes nothing at all: partial progress on four targets
    is worth less than one target actually crossing the line."""
    if targets is None:
        targets = [("intent", "motive"), ("intent", "witting_role"), ("intent", "scam_stage")]

    reports = [rank(store, sp, ky, limit=limit) for sp, ky in targets]
    open_targets = [r for r in reports if r["phase"] != "satisfied" and r["queue"]]
    # furthest from the line = most pairs still needed
    open_targets.sort(key=lambda r: -r["needs_paired"])
    focus = open_targets[0] if open_targets else None

    return {
        "targets": [{k: v for k, v in r.items() if k != "queue"} for r in reports],
        "focus": focus["target"] if focus else None,
        "queue": focus["queue"] if focus else [],
        "reading": (f"Focus on {focus['target']}: it needs {focus['needs_paired']} more "
                    f"heuristic/gold pairs, the furthest of any target from clearing the gate."
                    if focus else
                    "Every target is either satisfied or has no candidates carrying a heuristic "
                    "prediction. Nothing to ask right now."),
    }
