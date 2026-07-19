"""
core/graduation.py - is a heuristic module ready to become a trained model? (readiness gate).

The actor layer runs on expert-set deterministic rules. The plan is to graduate each module
to a trained model once the labeling substrate has accumulated enough human-adjudicated
ground truth. The discipline that keeps that honest: do NOT replace a rule until a learned
model would actually beat it, and until the human labels are trustworthy and plentiful.

This module measures exactly that, without needing the ML stack. For a target (label_space,
label_key) it pairs each subject's HEURISTIC self-label (what the rule said, source=heuristic,
recovered from the label history) against the GOLD label (what a human/confirmed source later
adjudicated) and computes:

  - coverage: how many gold labels exist, and how many pair with a heuristic prediction,
  - accuracy: how often the rule already agrees with the gold label,
  - Cohen's kappa: agreement corrected for chance (the honest metric on skewed classes),
  - a confusion table over the observed classes,
  - a VERDICT: not_enough_gold / low_agreement / ready_to_train, with the reason.

The verdict is deliberately conservative. "ready_to_train" does not mean the model is built;
it means there is now enough trustworthy, disagreement-bearing data that training one is
worthwhile. A rule that already agrees with humans near-perfectly does not need replacing yet
(there is nothing for a model to learn beyond it); the interesting targets are the ones with
enough gold AND enough residual disagreement for a model to improve on.

Pure Python stdlib. Reads a store exposing label_history / labeling_stats (core/store.py).
"""

from __future__ import annotations

# Sources trusted as ground truth (not the module's own guess).
GOLD_SOURCES = ("analyst", "confirmed_loss", "chargeback", "victim_report", "law_enforcement")

# Readiness thresholds. Conservative on purpose.
MIN_GOLD = 50           # below this, any metric is noise
MIN_PAIRED = 30         # need heuristic+gold pairs to measure agreement at all
KAPPA_FLOOR = 0.4       # below this the heuristic and humans barely agree: fix the rule first
KAPPA_CEILING = 0.9     # at/above this the rule already matches humans: little for a model to add


def _subjects_with_gold(store, label_space: str, label_key: str, gold_sources) -> dict:
    """subject_ref -> gold label_value, taking the most recent CURRENT gold label per subject."""
    rows = store.training_rows(label_space, label_key, sources=list(gold_sources), limit=1_000_000)
    out = {}
    for r in rows:
        if r["subject_ref"]:
            out[r["subject_ref"]] = r["label"]
    return out


def _heuristic_by_subject(store, subjects, label_space: str, label_key: str) -> dict:
    """subject_ref -> the heuristic self-label the module recorded, recovered from history
    (the analyst label supersedes it, so it lives in label_history, not current_labels)."""
    out = {}
    for subj in subjects:
        for lbl in store.label_history(subject_ref=subj):
            if (lbl.label_space == label_space and lbl.label_key == label_key
                    and lbl.source == "heuristic"):
                out[subj] = lbl.label_value
                break
    return out


def _cohen_kappa(pairs) -> float:
    """Cohen's kappa over (heuristic, gold) pairs. Chance-corrected agreement, so it does not
    flatter a rule that just always predicts the majority class."""
    n = len(pairs)
    if n == 0:
        return 0.0
    classes = sorted({c for pair in pairs for c in pair})
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    mat = [[0] * k for _ in range(k)]
    for h, g in pairs:
        mat[idx[h]][idx[g]] += 1
    po = sum(mat[i][i] for i in range(k)) / n
    row = [sum(mat[i]) for i in range(k)]
    col = [sum(mat[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    if pe >= 1.0:
        return 1.0                      # degenerate single-class agreement
    return round((po - pe) / (1 - pe), 3)


def _confusion(pairs) -> dict:
    """Nested dict heuristic_value -> {gold_value: count}."""
    out: dict = {}
    for h, g in pairs:
        out.setdefault(h, {}).setdefault(g, 0)
        out[h][g] += 1
    return out


def evaluate_target(store, label_space: str, label_key: str,
                    gold_sources=GOLD_SOURCES) -> dict:
    """Graduation readiness for one (label_space, label_key) target."""
    gold = _subjects_with_gold(store, label_space, label_key, gold_sources)
    heur = _heuristic_by_subject(store, gold.keys(), label_space, label_key)

    pairs = [(heur[s], gold[s]) for s in gold if s in heur]
    n_gold = len(gold)
    n_paired = len(pairs)
    accuracy = round(sum(1 for h, g in pairs if h == g) / n_paired, 3) if n_paired else 0.0
    kappa = _cohen_kappa(pairs)

    if n_gold < MIN_GOLD or n_paired < MIN_PAIRED:
        verdict, reason = "not_enough_gold", (
            f"{n_gold} gold labels, {n_paired} paired with a heuristic prediction; "
            f"need >= {MIN_GOLD} gold and >= {MIN_PAIRED} paired to measure agreement")
    elif kappa < KAPPA_FLOOR:
        verdict, reason = "low_agreement", (
            f"kappa {kappa} below floor {KAPPA_FLOOR}: the rule and human adjudicators barely "
            f"agree, so fix or re-specify the heuristic before training on its bootstrap")
    elif kappa >= KAPPA_CEILING:
        verdict, reason = "rule_already_strong", (
            f"kappa {kappa} at/above {KAPPA_CEILING}: the rule already matches humans closely, "
            f"so a trained model has little to add yet; revisit as harder cases accumulate")
    else:
        verdict, reason = "ready_to_train", (
            f"kappa {kappa} in the useful band with {n_paired} paired examples: enough "
            f"trustworthy data and residual disagreement for a model to improve on the rule")

    return {
        "target": f"{label_space}.{label_key}",
        "gold_labels": n_gold,
        "paired_with_heuristic": n_paired,
        "heuristic_accuracy_vs_gold": accuracy,
        "cohen_kappa": kappa,
        "confusion": _confusion(pairs),
        "verdict": verdict,
        "reason": reason,
    }


def readiness_report(store, targets=None) -> dict:
    """Graduation readiness across the actor layer's trainable targets, plus substrate health.
    `targets` is a list of (label_space, label_key); a sensible default covers the modules
    whose intent labels a human adjudicates."""
    if targets is None:
        targets = [
            ("outcome", "is_fraud"),
            ("intent", "motive"),
            ("intent", "witting_role"),
            ("intent", "scam_stage"),
        ]
    return {
        "substrate": store.labeling_stats(),
        "targets": [evaluate_target(store, sp, ky) for sp, ky in targets],
    }
