"""
core/train.py - graduate a heuristic into a trained model (stdlib, no ML deps).

The graduation gate (core/graduation.py) says WHEN training is worthwhile: enough gold labels
and enough residual disagreement between the rule and the humans. This module does the training
itself and answers the decisive question: does a model trained on the point-in-time features
actually BEAT the heuristic on held-out gold labels? A rule should only be replaced once the
answer is yes.

Deliberately dependency-free: a compact categorical Naive-Bayes classifier with Laplace
smoothing, features median-binned from the decision snapshots. It runs under any Python, stays
inside the concept's self-contained substrate, and (unlike a single-signal rule) uses every
feature, so it can recover a subpopulation a hand-rule systematically misses. Reads
(features, label) rows from store.training_rows and the heuristic baseline from label history.

HONESTY: the numbers this returns are only as real as the labels in the store. On a synthetic
seeded cohort (core/seed_substrate.py) they are SYNTHETIC and demonstrate the mechanism only; a
real graduation needs real adjudicated gold labels.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from .graduation import GOLD_SOURCES, _heuristic_by_subject


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _prepare(rows) -> tuple:
    """Feature keys seen in TRAIN rows, and a per-key median for numeric binning (train-only,
    so nothing leaks from the test split). Only genuinely continuous numerics (more than a few
    distinct values) are median-binned; low-cardinality numerics like 0/1 flags are kept
    categorical, since median-splitting a binary column collapses it to a single useless bin."""
    keys = sorted({k for r in rows for k in r["features"].keys()})
    medians = {}
    for k in keys:
        vals = sorted(float(r["features"][k]) for r in rows
                      if k in r["features"] and _is_num(r["features"][k]))
        medians[k] = vals[len(vals) // 2] if (vals and len(set(vals)) > 4) else None
    return keys, medians


def _vec(features: dict, keys, medians) -> dict:
    """Discretise one feature dict: numeric -> hi/lo about the train median, else categorical."""
    out = {}
    for k in keys:
        val = features.get(k)
        if medians.get(k) is not None and _is_num(val):
            out[k] = "hi" if float(val) >= medians[k] else "lo"
        else:
            out[k] = "na" if val is None else str(val)
    return out


def _nb_train(X, y) -> dict:
    classes = sorted(set(y))
    class_count = {c: 0 for c in classes}
    feat_count = {c: defaultdict(lambda: defaultdict(int)) for c in classes}
    feat_vals = defaultdict(set)
    for xi, yi in zip(X, y):
        class_count[yi] += 1
        for k, val in xi.items():
            feat_count[yi][k][val] += 1
            feat_vals[k].add(val)
    return {"classes": classes, "class_count": class_count, "feat_count": feat_count,
            "feat_vals": {k: sorted(v) for k, v in feat_vals.items()}, "n": len(y)}


def _nb_predict(model: dict, xi: dict):
    classes, n = model["classes"], model["n"]
    best, best_lp = classes[0], -1e18
    for c in classes:
        lp = math.log((model["class_count"][c] + 1) / (n + len(classes)))     # class prior, smoothed
        total = model["class_count"][c]
        for k, val in xi.items():
            card = len(model["feat_vals"].get(k, [val])) or 1
            cnt = model["feat_count"][c][k].get(val, 0)
            lp += math.log((cnt + 1) / (total + card))                         # likelihood, Laplace
        if lp > best_lp:
            best_lp, best = lp, c
    return best


def _split(subject_ref: str, test_frac: float, seed: str) -> bool:
    """Deterministic train/test assignment: True = test. Hash-based so the split is stable and
    reproducible across runs and cannot drift between train and evaluate."""
    h = int(hashlib.sha256(f"{seed}:{subject_ref}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < test_frac


def train_target(store, label_space: str, label_key: str, gold_sources=GOLD_SOURCES,
                 test_frac: float = 0.3, min_rows: int = 40, seed: str = "redwing-train",
                 observed_only: bool = False) -> dict:
    """Train a model on gold labels for one target and compare it to the heuristic on a held-out
    test split. Returns model vs heuristic accuracy and whether the model beats the rule. Set
    `observed_only=True` for OUTCOME targets to train on uncensored (allowed) decisions only,
    so the model is not fit on labels the analyst inferred for cases we blocked."""
    rows = [r for r in store.training_rows(label_space, label_key,
                                           sources=list(gold_sources),
                                           observed_only=observed_only, limit=1_000_000)
            if r["features"]]
    rows = list({r["subject_ref"]: r for r in rows}.values())    # one row per subject
    if len(rows) < min_rows:
        return {"target": f"{label_space}.{label_key}", "trained": False,
                "reason": f"{len(rows)} usable gold rows with features (< {min_rows}); "
                          f"not enough to train and hold out"}

    heur = _heuristic_by_subject(store, [r["subject_ref"] for r in rows], label_space, label_key)
    train = [r for r in rows if not _split(r["subject_ref"], test_frac, seed)]
    test = [r for r in rows if _split(r["subject_ref"], test_frac, seed)]
    if len(train) < 10 or len(test) < 10:
        return {"target": f"{label_space}.{label_key}", "trained": False,
                "reason": f"split too small (train {len(train)}, test {len(test)})"}

    keys, medians = _prepare(train)
    model = _nb_train([_vec(r["features"], keys, medians) for r in train],
                      [r["label"] for r in train])

    m_correct = h_correct = h_evaluable = 0
    for r in test:
        gold = r["label"]
        if _nb_predict(model, _vec(r["features"], keys, medians)) == gold:
            m_correct += 1
        hl = heur.get(r["subject_ref"])
        if hl is not None:
            h_evaluable += 1
            h_correct += int(hl == gold)

    m_acc = round(m_correct / len(test), 3)
    h_acc = round(h_correct / h_evaluable, 3) if h_evaluable else None
    beats = h_acc is not None and m_acc > h_acc
    verdict = ("model_beats_rule: graduate this target to the trained model" if beats
               else "model_does_not_beat_rule: keep the heuristic for now" if h_acc is not None
               else "no heuristic baseline on the test split to compare against")

    return {
        "target": f"{label_space}.{label_key}",
        "trained": True,
        "n_train": len(train),
        "n_test": len(test),
        "classes": model["classes"],
        "model_accuracy": m_acc,
        "heuristic_accuracy": h_acc,
        "beats_heuristic": beats,
        "verdict": verdict,
    }
