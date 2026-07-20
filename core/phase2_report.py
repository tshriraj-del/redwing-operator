"""
core/phase2_report.py - WS7 + WS8: fire the graduation gate and run the counterfactual.

WS7 asks the question Phase 1 built the machinery for and never got to answer: on real labeled
data, does a model trained on point-in-time features beat the hand-written rule?

WS8 asks the question that justifies paying for a holdout: how much worse would that model be
if the system had blocked everything its rule flagged and never bought back a counterfactual?

The two arms are identical replays over the same transactions with the same fixed rule. The
only difference is the holdout policy, so any gap between them is attributable to it. One
variable, one measured effect.

MEASURED RESULT (300K transactions per arm, 2% release rate, recorded here because the
hypothesis was half wrong and that is worth keeping):

  - WS7 held. The model beat the rule on f1, 0.1795 vs 0.0147, on gold labels it never saw.
  - WS8's stated hypothesis did NOT hold. The holdout moved model f1 by +0.0017, which is
    less than the 0.0029 that a single extra true positive is worth on this test split. At a
    defensible release rate only 17 cases were released, 6 of them fraud, against ~209K
    training rows. Seventeen rows cannot move a model, and reporting +0.0017 as an effect
    would be exactly the number-dressing this platform exists to refuse.
  - The real finding was the other column. In the control arm the rule scores tp=0, fp=0,
    precision 0, recall 0 - not a weak rule, an UNMEASURABLE one. Every case it flagged was
    blocked, so no flagged case ever revealed an outcome. A system without a holdout cannot
    evaluate its own control at all. That, not training volume, is what the holdout buys.

`compare()` therefore checks every delta against an explicit noise floor before calling it an
effect, and reports rule evaluability as a first-class result rather than a footnote.
"""

from __future__ import annotations

from .graduation import readiness_report
from .train import train_target

TARGET = ("outcome", "is_fraud")
POSITIVE = "True"


def _arm(store, label: str) -> dict:
    """Readiness + trained comparison for one substrate."""
    space, key = TARGET
    stats = store.labeling_stats()
    ready = readiness_report(store, targets=[TARGET])
    # observed_only: train on uncensored decisions, which is the entire point. A held
    # transaction never revealed its outcome, so it carries no label to learn from anyway.
    trained = train_target(store, space, key, observed_only=True, positive_label=POSITIVE)
    return {"arm": label, "substrate": stats, "readiness": ready, "trained": trained}


def _f1(tp, fp, fn) -> float:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-12)


def _one_tp_worth(m: dict) -> float:
    """F1 movement caused by a single extra true positive on this test split.

    The noise floor. Any delta smaller than this is one lucky transaction, not an effect, and
    calling it an effect would be exactly the kind of number-dressing this platform exists to
    avoid."""
    tp, fp, fn = m.get("tp", 0), m.get("fp", 0), m.get("fn", 0)
    if tp <= 0:
        return 0.0
    return round(_f1(tp, fp, fn) - _f1(tp - 1, fp, fn + 1), 4)


def compare(holdout_store, control_store) -> dict:
    """Run both arms and diff them, against an explicit noise floor."""
    a = _arm(holdout_store, "with_holdout")
    b = _arm(control_store, "no_holdout")

    ma = a["trained"].get("model") or {}
    mb = b["trained"].get("model") or {}
    ha = a["trained"].get("heuristic") or {}
    hb = b["trained"].get("heuristic") or {}

    delta = (None if not ma or not mb else round(ma["f1"] - mb["f1"], 4))
    floor = _one_tp_worth(ma)
    significant = delta is not None and abs(delta) > floor

    # The rule's own evaluability is the other half of the question, and the more important
    # half: a control arm that blocks everything it flags never observes an outcome for any
    # case it flagged, so its rule has no measurable precision or recall at all.
    rule_evaluable = {
        "with_holdout": bool(ha.get("tp", 0) + ha.get("fp", 0)),
        "no_holdout": bool(hb.get("tp", 0) + hb.get("fp", 0)),
    }

    if delta is None:
        reading = "one or both arms did not train; no comparison"
    elif significant:
        reading = (f"holdout moved model f1 by {delta:+.4f}, above the {floor:.4f} value of a "
                   f"single test-set transaction: a real effect at this volume")
    else:
        reading = (f"holdout moved model f1 by {delta:+.4f}, BELOW the {floor:.4f} value of a "
                   f"single test-set transaction. That is noise, not an effect. At a "
                   f"defensible release rate the holdout does not measurably improve the "
                   f"model; its value here is that it makes the RULE evaluable at all")

    return {
        "target": f"{TARGET[0]}.{TARGET[1]}",
        "with_holdout": a,
        "no_holdout": b,
        "holdout_f1_delta": delta,
        "holdout_recall_delta": (None if not ma or not mb
                                 else round(ma["recall"] - mb["recall"], 4)),
        "noise_floor_one_tp": floor,
        "delta_is_significant": significant,
        "rule_evaluable": rule_evaluable,
        "reading": reading,
    }


def render(cmp: dict) -> str:
    """Plain-text summary. Deliberately shows the counts, not just the verdict."""
    L = []
    L.append(f"TARGET  {cmp['target']}")
    for key in ("with_holdout", "no_holdout"):
        a = cmp[key]
        s, t, r = a["substrate"], a["trained"], a["readiness"]
        L.append("")
        L.append(f"[{a['arm']}]")
        L.append(f"  decisions {s.get('decisions_total'):,}   observed {s.get('decisions_observed'):,}"
                 f"   censored {s.get('decisions_censored'):,}   labels {s.get('labels_current'):,}")
        tg = (r.get("targets") or [{}])[0]
        L.append(f"  gate: gold={tg.get('gold_labels')} paired={tg.get('paired_with_heuristic')} "
                 f"-> {tg.get('verdict', tg.get('recommendation', 'n/a'))}")
        if t.get("trained"):
            m, h = t["model"], t["heuristic"]
            L.append(f"  train {t['n_train']:,} / test {t['n_test']:,}   base rate {t['test_base_rate']}")
            L.append(f"  rule   p={h['precision']} r={h['recall']} f1={h['f1']}  (tp {h['tp']} fp {h['fp']} fn {h['fn']})")
            L.append(f"  model  p={m['precision']} r={m['recall']} f1={m['f1']}  (tp {m['tp']} fp {m['fp']} fn {m['fn']})")
            L.append(f"  verdict: {t['verdict']}")
        else:
            L.append(f"  not trained: {t.get('reason')}")
    L.append("")
    L.append(f"RULE EVALUABLE?  with_holdout={cmp['rule_evaluable']['with_holdout']}   "
             f"no_holdout={cmp['rule_evaluable']['no_holdout']}")
    L.append(f"HOLDOUT EFFECT   f1 delta {cmp['holdout_f1_delta']}   "
             f"noise floor (1 tp) {cmp['noise_floor_one_tp']}   "
             f"significant={cmp['delta_is_significant']}")
    L.append(f"  {cmp['reading']}")
    return "\n".join(L)
