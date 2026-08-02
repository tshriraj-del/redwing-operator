"""
core/train.py - graduate a heuristic into a trained model (stdlib, no ML deps).

The graduation gate (core/graduation.py) says WHEN training is worthwhile: enough gold labels
and enough residual disagreement between the rule and the humans. This module does the training
itself and answers the decisive question: does a model trained on the point-in-time features
actually BEAT the heuristic on held-out gold labels? A rule should only be replaced once the
answer is yes.

Two classifiers, both dependency-free, both over the same median-binned features:

  naive_bayes  a compact categorical NB with Laplace smoothing. Simple and fast, but it assumes
               the features are conditionally independent given the class. REDWING's features
               are not: the velocity family (velocity_1h / 4h / 24h / 7d / 30d) moves together
               almost by construction, so a single burst is counted as several independent
               pieces of evidence and NB is driven to overconfident probabilities. Its accuracy
               can look fine while its calibration is bad, which is dangerous for a model whose
               whole job is a graduation verdict.

  logreg       multinomial logistic regression, gradient descent, L2-regularised. It learns the
               feature weights JOINTLY, so it splits the credit across correlated features
               instead of multiplying it. On this feature set that makes it better calibrated,
               which is the property that matters here. Still pure Python, still no ML deps.

The default is logreg for exactly that reason. `train_target(..., model="naive_bayes")` keeps
the old classifier, and `compare_calibration()` measures the difference rather than asserting
it: on a cohort with duplicated (correlated) signals NB's log-loss blows up while its accuracy
barely moves, which is the overconfidence made visible.

HONESTY: the numbers this returns are only as real as the labels in the store. On a synthetic
seeded cohort (core/seed_substrate.py) they are SYNTHETIC and demonstrate the mechanism only; a
real graduation needs real adjudicated gold labels.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from .graduation import GOLD_SOURCES, _heuristic_by_subject
from .label_maturity import maturity_floor, partition


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


# -- Logistic regression (multinomial, gradient descent, pure Python) ----------
#
# The discretised feature dicts (key -> "hi"/"lo"/category) are one-hot encoded into a sparse
# index. Multinomial LR then learns one weight per (feature, class) plus a bias, jointly, so
# two correlated features share the credit for their common signal instead of each claiming it
# in full the way NB does. That joint fit is the whole reason it stays calibrated where NB does
# not. No numpy: the substrate is small (hundreds of rows) and staying stdlib is the point.

def _feature_index(vecs) -> dict:
    """Map each (key, value) seen in training to a column. Values unseen at train time are
    simply absent at predict time, which is the correct behaviour: an unknown category carries
    no learned weight rather than a guessed one."""
    idx = {}
    for v in vecs:
        for k, val in v.items():
            key = (k, val)
            if key not in idx:
                idx[key] = len(idx)
    return idx


def _sparse(vec: dict, index: dict) -> list:
    """Column indices this example activates (one-hot; absent keys contribute nothing)."""
    out = []
    for k, val in vec.items():
        j = index.get((k, val))
        if j is not None:
            out.append(j)
    return out


def _softmax(scores) -> list:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _lr_train(vecs, y, classes, *, epochs: int = 300, lr: float = 0.5,
              l2: float = 1.0) -> dict:
    """Multinomial logistic regression by full-batch gradient descent with L2.

    L2 matters here beyond the usual overfitting story: on perfectly correlated features it
    spreads the weight evenly across the duplicates instead of letting one absorb it all, which
    is exactly the joint behaviour NB lacks."""
    index = _feature_index(vecs)
    n_feat, n_cls = len(index), len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    rows = [(_sparse(v, index), cidx[t]) for v, t in zip(vecs, y)]

    # weights[class][feature], bias[class]
    W = [[0.0] * n_feat for _ in range(n_cls)]
    b = [0.0] * n_cls
    n = len(rows) or 1

    for _ in range(epochs):
        gW = [[0.0] * n_feat for _ in range(n_cls)]
        gb = [0.0] * n_cls
        for cols, ti in rows:
            scores = [b[c] + sum(W[c][j] for j in cols) for c in range(n_cls)]
            probs = _softmax(scores)
            for c in range(n_cls):
                err = probs[c] - (1.0 if c == ti else 0.0)
                gb[c] += err
                for j in cols:
                    gW[c][j] += err
        for c in range(n_cls):
            b[c] -= lr * gb[c] / n
            for j in range(n_feat):
                # L2 gradient (bias is not regularised, by convention)
                W[c][j] -= lr * (gW[c][j] / n + l2 * W[c][j] / n)

    return {"classes": classes, "index": index, "W": W, "b": b, "cidx": cidx}


def _lr_proba(model: dict, vec: dict) -> dict:
    cols = _sparse(vec, model["index"])
    scores = [model["b"][i] + sum(model["W"][i][j] for j in cols)
              for i in range(len(model["classes"]))]
    probs = _softmax(scores)
    return {c: probs[i] for i, c in enumerate(model["classes"])}


def _lr_predict(model: dict, vec: dict):
    proba = _lr_proba(model, vec)
    return max(proba, key=proba.get)


def _nb_proba(model: dict, xi: dict) -> dict:
    """NB class posteriors, normalised. Used to measure calibration, not just the argmax."""
    classes, n = model["classes"], model["n"]
    logp = {}
    for c in classes:
        lp = math.log((model["class_count"][c] + 1) / (n + len(classes)))
        total = model["class_count"][c]
        for k, val in xi.items():
            card = len(model["feat_vals"].get(k, [val])) or 1
            cnt = model["feat_count"][c][k].get(val, 0)
            lp += math.log((cnt + 1) / (total + card))
        logp[c] = lp
    m = max(logp.values())
    exps = {c: math.exp(lp - m) for c, lp in logp.items()}
    z = sum(exps.values()) or 1.0
    return {c: e / z for c, e in exps.items()}


def _log_loss(proba_fn, vecs, y, classes) -> float:
    """Mean cross-entropy of the predicted class distribution against the truth. This is the
    number that exposes overconfidence: a model that is confidently wrong is punished far
    harder than one that is uncertainly wrong, which accuracy cannot see."""
    eps, total = 1e-15, 0.0
    for v, t in zip(vecs, y):
        p = proba_fn(v).get(t, 0.0)
        total += -math.log(min(1.0, max(eps, p)))
    return round(total / (len(y) or 1), 4)


def _split(subject_ref: str, test_frac: float, seed: str) -> bool:
    """Deterministic train/test assignment: True = test. Hash-based so the split is stable and
    reproducible across runs and cannot drift between train and evaluate."""
    h = int(hashlib.sha256(f"{seed}:{subject_ref}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < test_frac


def _prf(pairs, positive: str) -> dict:
    """Precision / recall / F1 for one class. `pairs` is [(predicted, gold), ...]."""
    tp = sum(1 for p, g in pairs if p == positive and g == positive)
    fp = sum(1 for p, g in pairs if p == positive and g != positive)
    fn = sum(1 for p, g in pairs if p != positive and g == positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def train_target(store, label_space: str, label_key: str, gold_sources=GOLD_SOURCES,
                 test_frac: float = 0.3, min_rows: int = 40, seed: str = "redwing-train",
                 observed_only: bool = False, positive_label: str = "",
                 model: str = "logreg", mature_only: bool = False, as_of=None) -> dict:
    """Train a model on gold labels for one target and compare it to the heuristic on a held-out
    test split. Returns model vs heuristic accuracy and whether the model beats the rule. Set
    `observed_only=True` for OUTCOME targets to train on uncensored (allowed) decisions only,
    so the model is not fit on labels the analyst inferred for cases we blocked.

    `mature_only=True` additionally drops decisions too recent for their labels to have arrived.
    These two flags close DIFFERENT holes and neither substitutes for the other: observed_only
    is about cases we blocked, so the outcome was never observable at all; mature_only is about
    cases we allowed, where the outcome exists and simply has not reached us yet. Train without
    it and the recent window contributes rows whose fraud is still in the post, every one of
    them labelled negative, and the model learns that recent traffic is safe.

    It REFUSES rather than degrades. If the maturity curve cannot be derived, this returns
    trained=False with the reason instead of quietly training on everything, because a filter
    that silently does nothing while the caller believes it ran is worse than no filter.

    `model` is "logreg" (default) or "naive_bayes". Logreg fits feature weights jointly and so
    stays calibrated on REDWING's correlated features (the velocity family); NB assumes
    independence and grows overconfident on them. Both are pure Python.

    `positive_label` matters when the target is rare. Accuracy is the right comparison for a
    roughly balanced target like intent.motive, and a useless one for outcome.is_fraud at a
    0.65% base rate, where predicting "never fraud" scores 99.35%. Pass the positive class
    (e.g. "True") and the verdict is decided on F1 for that class instead, with precision and
    recall reported for both model and rule. Left empty, behaviour is unchanged."""
    rows = [r for r in store.training_rows(label_space, label_key,
                                           sources=list(gold_sources),
                                           observed_only=observed_only, limit=1_000_000)
            if r["features"]]
    rows = list({r["subject_ref"]: r for r in rows}.values())    # one row per subject

    maturity = None
    if mature_only:
        mf = maturity_floor(store, label_space, label_key, as_of=as_of)
        if not mf["known"]:
            return {"target": f"{label_space}.{label_key}", "trained": False,
                    "mature_only": True, "maturity_known": False,
                    "reason": f"mature_only was requested and the maturity floor is not "
                              f"derivable, so the filter cannot be applied and training would "
                              f"silently include immature rows: {mf['reason']}"}
        part = partition(rows, mf["floor"])
        maturity = {"floor": mf["floor"], "days_to_coverage": mf["days_to_coverage"],
                    "coverage": mf["coverage"], "excluded_immature": len(part["immature"])}
        rows = part["mature"]

    if len(rows) < min_rows:
        # Name the maturity filter when it is what emptied the set, so nobody reads this as
        # "we have no labels" and goes looking for analysts. It is "we have labels about a
        # window too recent to trust", and the fix is time, not effort.
        why = (f" ({maturity['excluded_immature']} more were dropped as immature, decided "
               f"after {maturity['floor']})" if maturity and maturity["excluded_immature"]
               else "")
        return {"target": f"{label_space}.{label_key}", "trained": False,
                "maturity": maturity,
                "reason": f"{len(rows)} usable gold rows with features (< {min_rows})"
                          f"{why}; not enough to train and hold out"}

    heur = _heuristic_by_subject(store, [r["subject_ref"] for r in rows], label_space, label_key)
    train = [r for r in rows if not _split(r["subject_ref"], test_frac, seed)]
    test = [r for r in rows if _split(r["subject_ref"], test_frac, seed)]
    if len(train) < 10 or len(test) < 10:
        return {"target": f"{label_space}.{label_key}", "trained": False,
                "reason": f"split too small (train {len(train)}, test {len(test)})"}

    keys, medians = _prepare(train)
    tr_vecs = [_vec(r["features"], keys, medians) for r in train]
    tr_y = [r["label"] for r in train]
    classes = sorted(set(tr_y))

    if model == "naive_bayes":
        nb = _nb_train(tr_vecs, tr_y)
        predict = lambda v: _nb_predict(nb, v)
        proba = lambda v: _nb_proba(nb, v)
    else:
        model = "logreg"
        lr = _lr_train(tr_vecs, tr_y, classes)
        predict = lambda v: _lr_predict(lr, v)
        proba = lambda v: _lr_proba(lr, v)

    te_vecs = [_vec(r["features"], keys, medians) for r in test]
    te_y = [r["label"] for r in test]
    log_loss = _log_loss(proba, te_vecs, te_y, classes)

    m_correct = h_correct = h_evaluable = 0
    m_pairs, h_pairs = [], []
    for r, v in zip(test, te_vecs):
        gold = r["label"]
        pred = predict(v)
        m_pairs.append((pred, gold))
        if pred == gold:
            m_correct += 1
        hl = heur.get(r["subject_ref"])
        if hl is not None:
            h_evaluable += 1
            h_correct += int(hl == gold)
            h_pairs.append((hl, gold))

    m_acc = round(m_correct / len(test), 3)
    h_acc = round(h_correct / h_evaluable, 3) if h_evaluable else None

    out = {
        "target": f"{label_space}.{label_key}",
        "trained": True,
        "classifier": model,               # "logreg" | "naive_bayes"
        "n_train": len(train),
        "n_test": len(test),
        "classes": classes,
        "model_accuracy": m_acc,
        "heuristic_accuracy": h_acc,
        "model_log_loss": log_loss,        # calibration: lower is better, exposes overconfidence
        "maturity": maturity,              # None when mature_only was not requested
    }

    if positive_label:
        # Rare-target mode: judge on F1 for the positive class, because accuracy here is
        # dominated by the negative class and would call a do-nothing model excellent.
        mp, hp = _prf(m_pairs, positive_label), _prf(h_pairs, positive_label)
        base = sum(1 for _, g in m_pairs if g == positive_label) / max(len(m_pairs), 1)
        beats = bool(h_pairs) and mp["f1"] > hp["f1"]
        out.update({
            "positive_label": positive_label,
            "test_base_rate": round(base, 5),
            "model": mp,
            "heuristic": hp,
            "decided_on": "f1",
            "beats_heuristic": beats,
            "verdict": (f"model_beats_rule on f1 ({mp['f1']} vs {hp['f1']}): graduate" if beats
                        else f"model_does_not_beat_rule on f1 ({mp['f1']} vs {hp['f1']}): keep the rule"
                        if h_pairs else "no heuristic baseline on the test split"),
        })
        return out

    beats = h_acc is not None and m_acc > h_acc
    out.update({
        "decided_on": "accuracy",
        "beats_heuristic": beats,
        "verdict": ("model_beats_rule: graduate this target to the trained model" if beats
                    else "model_does_not_beat_rule: keep the heuristic for now" if h_acc is not None
                    else "no heuristic baseline on the test split to compare against"),
    })
    return out


def compare_calibration(store, label_space: str, label_key: str, **kw) -> dict:
    """Train both classifiers on the same target and split, and report the difference the way
    it actually shows up: not in accuracy, which can be near-identical, but in calibration.

    On REDWING's correlated features NB's log-loss runs well above logreg's while its accuracy
    barely moves. That gap is the overconfidence, quantified. Returned as measurement, not
    assertion; if the gap is small on a given cohort, that is what it will say."""
    nb = train_target(store, label_space, label_key, model="naive_bayes", **kw)
    lr = train_target(store, label_space, label_key, model="logreg", **kw)
    if not (nb.get("trained") and lr.get("trained")):
        return {"trained": False, "reason": nb.get("reason") or lr.get("reason")}

    nb_ll, lr_ll = nb["model_log_loss"], lr["model_log_loss"]
    return {
        "target": f"{label_space}.{label_key}",
        "naive_bayes": {"accuracy": nb["model_accuracy"], "log_loss": nb_ll},
        "logreg": {"accuracy": lr["model_accuracy"], "log_loss": lr_ll},
        "log_loss_improvement": round(nb_ll - lr_ll, 4),
        "reading": (
            f"logreg is better calibrated: log-loss {lr_ll} vs {nb_ll} "
            f"(accuracy {lr['model_accuracy']} vs {nb['model_accuracy']}). NB's independence "
            f"assumption made it overconfident on the correlated features."
            if lr_ll < nb_ll - 0.02 else
            f"no meaningful calibration gap on this cohort: log-loss {lr_ll} vs {nb_ll}. "
            f"The features here may not be correlated enough for NB's assumption to bite."
        ),
    }
