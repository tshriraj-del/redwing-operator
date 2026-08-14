"""
core/model_performance.py - did the model get WORSE, and how would we know.

WHY. `drift_monitor.py` computes PSI over score and feature distributions. That is genuine and
it is label-free, which means it can only ever say THE INPUT MOVED. It cannot say the model got
worse, because nothing in it ever meets an outcome. Until the outcome ledger there was nothing
it could have met.

So REDWING could detect a shifted population and could not detect a decayed model. That is the
first thing a model-risk reviewer asks about, and the gap was found by withdrawing a claim: this
module exists because an earlier note asserted that label maturity had broken drift monitoring,
and checking showed drift monitoring never used labels at all.

THE THREE-WAY ATTRIBUTION IS THE WHOLE POINT. When this month looks worse than last, there are
three explanations and they demand opposite responses:

    the model degraded      retrain, and the sooner the better
    the population shifted  the model may be fine; the mix it is being asked about changed
    the labels are late     nothing happened at all, you are reading an empty window

Conflating the third with the first is how a team retrains on noise. Conflating the third with
IMPROVEMENT is how a team misses real decay for a quarter, because unreported fraud looks
exactly like an absence of fraud. `diagnose()` separates them, using PSI for the second and
core/label_maturity for the third, and it will decline to report a verdict rather than guess.

CENSORING, AND WHY "RECALL" IS ALMOST ALWAYS A LIE HERE. Outcomes exist only where we ALLOWED
the payment. A blocked case never reveals what it would have been, so every metric computed from
production data describes the allowed population and nothing else. Reporting that as the model's
recall overstates it, silently and permanently, because the frauds the model caught and blocked
are exactly the ones missing from the denominator.

This module never calls it recall. It calls it `recall_on_allowed` and reports the censored
share beside it. Where a holdout exists, `estimate_censored()` uses the released sample (cases
the policy wanted to block, released at random, at real cost) to estimate the blocked
population's fraud rate, which is the correction the holdout was built and paid for.

Pure stdlib. Reads decisions, labels and the maturity curve; writes nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from match_engine import is_alert

from .label_maturity import GOLD_SOURCES, _parse, maturity_floor
from .store import FRAUD_TRUE

# Below this many labelled rows a window's metrics are noise and are not reported as metrics.
MIN_LABELLED = 30
# Below this many holdout releases the censored-population estimate is not attempted: a fraud
# rate from a handful of releases has an error bar wider than the quantity it estimates.
MIN_RELEASES = 20
# A drop of at least this much in precision or recall before "degraded" is even considered.
# Smaller moves on a few hundred labels are sampling noise wearing a trend's clothing.
MATERIAL_DROP = 0.05

CENSORING_ACTIONS = ("BLOCK", "HOLD", "DECLINE")


def _iso(d: datetime) -> str:
    return d.isoformat().replace("+00:00", "Z")


def _rows(store, start: str, end: str) -> list:
    """Decisions in a window, with their current outcome label when one exists.

    LEFT JOIN, deliberately. training_rows() inner-joins because training needs a label, but
    measurement needs to know how many decisions have NO outcome yet: that count is label
    coverage, and it is the difference between "the model missed these" and "nobody has told us
    about these".
    """
    q = ("SELECT d.decision_id, d.subject_ref, d.ts, d.action, d.score, d.expected_liability, "
         "d.features, d.rationale, l.label_value, l.source, l.ts AS label_ts "
         "FROM decisions d LEFT JOIN labels l "
         "  ON l.decision_id = d.decision_id AND l.label_space='outcome' "
         "  AND l.label_key='is_fraud' AND l.superseded_by='' "
         f"  AND l.source IN ({','.join('?' * len(GOLD_SOURCES))}) "
         "WHERE d.ts >= ? AND d.ts < ? AND d.score IS NOT NULL")
    args = list(GOLD_SOURCES) + [start, end]
    out = []
    for r in store._conn.execute(q, args).fetchall():
        try:
            rat = json.loads(r["rationale"] or "{}")
        except (ValueError, TypeError):
            rat = {}
        out.append({
            "decision_id": r["decision_id"], "subject_ref": r["subject_ref"], "ts": r["ts"],
            "action": (r["action"] or "").upper(), "score": float(r["score"] or 0.0),
            "liability": float(r["expected_liability"] or 0.0),
            "outcome": r["label_value"], "source": r["source"],
            "released": bool(rat.get("released")),
        })
    return out


def _rows_with_features(store, start: str, end: str) -> list:
    """Same window as _rows(), but carrying the point-in-time feature snapshot. Kept separate
    because the metric path never needs features and parsing them for every row would make the
    common call slower for nothing."""
    out = []
    for r in store._conn.execute(
        "SELECT ts, score, features FROM decisions WHERE ts >= ? AND ts < ? AND score IS NOT NULL",
        (start, end),
    ).fetchall():
        try:
            feats = json.loads(r["features"] or "{}")
        except (ValueError, TypeError):
            feats = {}
        out.append({"ts": r["ts"], "score": float(r["score"] or 0.0), "features": feats})
    return out


def _prf(rows) -> dict:
    """Confusion of the MODEL'S call against the outcome, over rows that have one."""
    tp = fp = fn = tn = 0
    caught = missed = 0.0
    for r in rows:
        if r["outcome"] is None:
            continue
        fraud = r["outcome"] == FRAUD_TRUE
        flag = is_alert(r["score"])
        if flag and fraud:
            tp += 1
            caught += r["liability"]
        elif flag and not fraud:
            fp += 1
        elif fraud:
            fn += 1
            missed += r["liability"]
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = round(tp / (tp + fp), 4) if (tp + fp) else None
    rec = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1 = (round(2 * prec * rec / (prec + rec), 4)
          if (prec and rec and (prec + rec)) else None)
    return {"n_labelled": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_on_allowed": prec, "recall_on_allowed": rec, "f1_on_allowed": f1,
            "base_rate": round((tp + fn) / n, 5) if n else None,
            "liability_caught": round(caught, 2), "liability_missed": round(missed, 2)}


def estimate_censored(rows) -> dict:
    """What the blocked population probably contained, from the holdout releases.

    A released decision is one the policy wanted to block and which was allowed anyway, chosen
    at random and at real expected cost. That sample is the only unbiased view of the blocked
    population anyone has, which is precisely what a holdout is for and why it is worth paying
    for. Its observed fraud rate estimates the blocked population's, and multiplying gives the
    frauds the block wall is plausibly stopping.

    Refuses below MIN_RELEASES, because a rate from a handful of releases carries an error bar
    wider than the number it is estimating, and a confidently wrong correction is worse than an
    acknowledged gap.
    """
    blocked = [r for r in rows if r["action"] in CENSORING_ACTIONS and not r["released"]]
    released = [r for r in rows if r["released"] and r["outcome"] is not None]
    out = {"n_blocked": len(blocked), "n_released_with_outcome": len(released),
           "censored_share": round(len(blocked) / len(rows), 4) if rows else 0.0}
    if len(released) < MIN_RELEASES:
        out.update({"estimable": False,
                    "reason": f"{len(released)} holdout releases with an outcome, need "
                              f">= {MIN_RELEASES}; the blocked population stays unmeasured "
                              f"rather than estimated from a handful"})
        return out
    rate = sum(1 for r in released if r["outcome"] == FRAUD_TRUE) / len(released)
    out.update({
        "estimable": True,
        "fraud_rate_in_released": round(rate, 4),
        "estimated_frauds_in_blocked": round(len(blocked) * rate, 1),
        "why": ("released cases are would-be-blocks allowed at random, so their fraud rate is "
                "an unbiased estimate of the blocked population's"),
    })
    return out


def window(store, start: str, end: str, *, mature_only: bool = True, as_of=None) -> dict:
    """Realised performance for decisions made in [start, end).

    Cohorts are keyed on the DECISION date, never the label date. A window is a set of payments
    the model judged; when we found out is a different question, and answering it here would
    quietly mix a slow month for the disputes team into the model's score.
    """
    rows = _rows(store, start, end)
    out = {"window": {"start": start, "end": end}, "n_decisions": len(rows)}

    mat = maturity_floor(store, "outcome", "is_fraud", as_of=as_of)
    out["maturity"] = {"known": mat["known"], "floor": mat.get("floor"),
                       "reason": mat["reason"] if not mat["known"] else None}

    if mature_only and mat["known"]:
        floor = _parse(mat["floor"])
        keep = [r for r in rows if (_parse(r["ts"]) or datetime.max.replace(
            tzinfo=timezone.utc)) <= floor]
        out["maturity"]["excluded_immature"] = len(rows) - len(keep)
        rows = keep
    elif mature_only:
        # The curve is unavailable, so nothing can be certified mature. Measure anyway and say
        # loudly what the number is worth, rather than either refusing outright or pretending
        # the filter ran. Refusing outright would leave a system with no performance view at
        # all for its first quarter, which is worse than a clearly-caveated one.
        out["maturity"]["excluded_immature"] = 0
        out["maturity"]["caveat"] = (
            "maturity is not derivable, so these figures may be measured over a window whose "
            "outcomes have not finished arriving. Late outcomes are overwhelmingly FRAUD, so "
            "the error is one-directional: precision and recall here are optimistic.")

    labelled = [r for r in rows if r["outcome"] is not None]
    out["n_labelled"] = len(labelled)
    out["label_coverage"] = round(len(labelled) / len(rows), 4) if rows else 0.0
    out["censoring"] = estimate_censored(rows)

    if len(labelled) < MIN_LABELLED:
        out.update({"measurable": False,
                    "reason": f"{len(labelled)} outcome-labelled decisions in this window, "
                              f"need >= {MIN_LABELLED}"})
        return out

    out["measurable"] = True
    out["metrics"] = _prf(rows)
    return out


def trend(store, *, windows: int = 6, window_days: int = 30, as_of=None,
          mature_only: bool = True) -> dict:
    """Consecutive cohorts, oldest first, so a direction is visible rather than a single point."""
    now = _parse(as_of) or datetime.now(timezone.utc)
    out = []
    for i in range(windows, 0, -1):
        end = now - timedelta(days=window_days * (i - 1))
        start = end - timedelta(days=window_days)
        out.append(window(store, _iso(start), _iso(end),
                          mature_only=mature_only, as_of=now))
    return {"window_days": window_days, "windows": out}


def _psi_between(store, a: dict, b: dict) -> dict:
    """PSI between two windows on the INPUT and, separately, on the output.

    The distinction is the one that makes the attribution work, and the first version of this
    module got it wrong: it tested for population shift using PSI on the SCORE, and a genuinely
    decayed model was then filed as a shifted population. Of course it was. The score is the
    model's own output, so it moves when the traffic changes AND when the model changes, which
    makes it useless for telling those apart. Tested against a fixture where accuracy fell from
    0.95 to 0.55 on identical inputs, score PSI read 0.29 and the verdict came back
    "population_shift".

    The input is what a population shift is actually about, so the feature distribution is the
    right instrument: features moved means the traffic changed; features stable while
    performance fell means the model did. Score PSI is still reported, because it is what
    drift_monitor has always shown and dropping it would make two views of the system disagree
    for no reason, but it no longer decides anything.

    Reuses the drift monitor's estimator rather than writing a second one that could drift from
    it.
    """
    out = {"score_psi": None, "feature_psi": None, "feature_psi_by_key": {},
           "min_per_side": None}
    try:
        from drift_monitor import MIN_PER_SIDE, _compute_psi
    except Exception:                                            # noqa: BLE001
        return out
    out["min_per_side"] = MIN_PER_SIDE
    ra = _rows_with_features(store, a["start"], a["end"])
    rb = _rows_with_features(store, b["start"], b["end"])
    if len(ra) < MIN_PER_SIDE or len(rb) < MIN_PER_SIDE:
        # The sample floor is the drift monitor's, imported rather than restated. Two copies of
        # a threshold is how one window gets called shifted by this module and steady by the
        # monitor, and a reviewer then has to work out which view to believe.
        return out

    def psi(xs, ys):
        p = _compute_psi(xs, ys)
        return round(float(p), 4) if p is not None else None

    out["score_psi"] = psi([r["score"] for r in ra], [r["score"] for r in rb])

    keys = ({k for r in ra for k, v in r["features"].items() if isinstance(v, (int, float))}
            & {k for r in rb for k, v in r["features"].items() if isinstance(v, (int, float))})
    for k in sorted(keys):
        xs = [float(r["features"][k]) for r in ra if k in r["features"]]
        ys = [float(r["features"][k]) for r in rb if k in r["features"]]
        p = psi(xs, ys)
        if p is not None:
            out["feature_psi_by_key"][k] = p
    if out["feature_psi_by_key"]:
        out["feature_psi"] = max(out["feature_psi_by_key"].values())
    return out


# The differential, in the order diagnose() tests it. These are not severity levels, they are
# competing EXPLANATIONS for the same observation, and they demand opposite responses. The order
# is the order in which one can be excluded: cheapest and most common first, so the expensive
# conclusion (the model decayed, go retrain) is only reached once the cheap ones are gone.
REASON_LADDER = (
    ("unmeasurable",
     "Have enough outcomes arrived to measure this window at all?"),
    ("no_baseline",
     "Is there an earlier measurable window to compare against?"),
    ("stable",
     "Did anything actually fall, by more than sampling noise?"),
    ("population_shift",
     "Did the INPUT distribution move, so the model is being asked a different question?"),
    ("degraded",
     "Input steady and performance down: what is left is the model."),
)

RULED_IN = "ruled_in"
RULED_OUT = "ruled_out"
NOT_REACHED = "not_reached"


def _differential(steps: dict) -> list:
    """The ladder as an ordered list, with every rung the procedure never reached marked as such.

    WHY THIS IS ASSEMBLED HERE AND NOT IN THE CONSOLE. diagnose() already walks this ladder and
    then throws away every rung it passed, leaving the caller a verdict with no way to show what
    was excluded to reach it. Rebuilding the ladder in the frontend would be a second copy of the
    decision procedure, free to disagree with this one after the next edit to either. So the
    procedure reports its own work.

    `not_reached` is a THIRD status and the distinction from `ruled_out` is the whole value. An
    untested explanation has not been excluded, it is simply unknown. A view that greys both out
    the same way tells a reviewer three reasons were considered when the procedure stopped after
    one, which is the specific false assurance this module exists to refuse.
    """
    out = []
    for reason, question in REASON_LADDER:
        s = steps.get(reason)
        out.append({
            "reason": reason,
            "question": question,
            "status": s[0] if s else NOT_REACHED,
            "evidence": s[1] if s else "not tested: the procedure stopped before this rung",
        })
    return out


def _attribute(store, prev: dict, recent: dict, base: dict, steps: dict) -> dict:
    """Rungs 3 to 5, once a measurable window and a baseline both exist.

    Split out of diagnose() only for length. The ladder is one procedure; this is its tail.
    """
    psi = _psi_between(store, prev["window"], recent["window"])
    base["score_psi_vs_previous"] = psi["score_psi"]
    base["feature_psi_vs_previous"] = psi["feature_psi"]
    base["feature_psi_by_key"] = psi["feature_psi_by_key"]

    drops = {}
    for k in ("precision_on_allowed", "recall_on_allowed"):
        a, b = prev["metrics"].get(k), recent["metrics"].get(k)
        if a is not None and b is not None:
            drops[k] = round(b - a, 4)
    worst = min(drops.values()) if drops else 0.0
    base["deltas_vs_previous"] = drops
    moves = ", ".join(f"{k.split('_on_')[0]} {v:+}" for k, v in drops.items()) or "no comparable metric"

    if worst > -MATERIAL_DROP:
        steps["stable"] = (RULED_IN,
                           f"{moves} against the previous window, none past the {MATERIAL_DROP} "
                           f"materiality bar on {recent['metrics']['n_labelled']} labelled rows")
        return {**base, "verdict": "stable", "differential": _differential(steps),
                "reason": (f"no metric fell by more than {MATERIAL_DROP} against the previous "
                           f"window; moves this size on {recent['metrics']['n_labelled']} "
                           f"labelled rows are sampling noise")}

    steps["stable"] = (RULED_OUT, f"{moves}, past the {MATERIAL_DROP} materiality bar")

    fpsi = psi["feature_psi"]
    if fpsi is not None and fpsi >= 0.20:
        worst_key = max(psi["feature_psi_by_key"], key=psi["feature_psi_by_key"].get)
        steps["population_shift"] = (RULED_IN,
                                     f"feature PSI {fpsi} on {worst_key!r}, past the 0.20 "
                                     f"drift threshold; the traffic moved")
        return {**base, "verdict": "population_shift", "differential": _differential(steps),
                "reason": (f"performance fell (worst delta {worst}) AND the INPUT distribution "
                           f"moved materially (feature PSI {fpsi} on {worst_key!r}). The model "
                           f"may be unchanged and simply being asked a different question. "
                           f"Investigate the mix before retraining, because retraining on a "
                           f"shifted population bakes the shift in")}

    # fpsi is None means the PSI never ran, which is NOT the same as the inputs holding still.
    # The verdict falls through to the model either way, and saying so is the point: a
    # `not_reached` here tells the reader this conclusion rests on an exclusion nobody performed.
    if fpsi is None:
        steps["population_shift"] = (
            NOT_REACHED,
            "no feature PSI could be computed (under 50 rows carrying numeric features on one "
            "side of the comparison), so the inputs are UNTESTED rather than steady")
        held = "the input distribution was never tested"
    else:
        steps["population_shift"] = (
            RULED_OUT,
            f"worst feature PSI {fpsi}, under the 0.20 threshold; the inputs held still")
        held = f"a steady input distribution (feature PSI {fpsi})"

    # `degraded` is the verdict that authorises a retrain, so it requires BOTH competing
    # explanations to have actually been excluded, not merely to have been skipped. An
    # uncomputable PSI and an underivable maturity curve are the same kind of gap: something the
    # conclusion rests on that nobody measured. Either one downgrades this to unconfirmed.
    unconfirmed = []
    if fpsi is None:
        unconfirmed.append(
            "the input distribution was never tested, so a population shift is not excluded")
    if not recent["maturity"]["known"]:
        unconfirmed.append(
            "maturity is not derivable, and a window short of its late-arriving frauds shows a "
            "falling recall for a reason that has nothing to do with the model")

    if unconfirmed:
        steps["degraded"] = (RULED_IN,
                             f"worst delta {worst} with {held}. NOT confirmed: "
                             + "; ".join(unconfirmed))
        return {**base, "verdict": "degraded_unconfirmed", "differential": _differential(steps),
                "reason": (f"performance fell (worst delta {worst}), which points at the model. "
                           f"It is NOT confirmed, because " + "; and ".join(unconfirmed)
                           + ". Treat as a signal to watch, not a finding")}

    steps["degraded"] = (RULED_IN,
                         f"worst delta {worst} on a MATURE window with {held}. Late labels are "
                         f"excluded and so is the population")
    return {**base, "verdict": "degraded", "differential": _differential(steps),
            "reason": (f"performance fell (worst delta {worst}) on a MATURE window whose INPUT "
                       f"distribution is stable (feature PSI {fpsi}). Population shift and label "
                       f"lag are both excluded, which leaves the model. This is the one that "
                       f"justifies a retrain")}


def diagnose(store, *, window_days: int = 30, as_of=None) -> dict:
    """Model degraded, population shifted, or labels still arriving? THE question this exists for.

    Returns one verdict, the evidence for it, and under `differential` the full ladder of
    competing explanations with each one marked ruled in, ruled out, or never reached.
    `unmeasurable` is a first-class answer and the most common one on a young substrate: a window
    with too few outcomes is not a bad month, it is an empty one, and calling it a regression
    would send somebody to retrain against noise.
    """
    t = trend(store, windows=4, window_days=window_days, as_of=as_of)
    ws = t["windows"]
    recent = ws[-1]
    priors = [w for w in ws[:-1] if w.get("measurable")]
    steps: dict = {}

    base = {"recent": recent["window"], "window_days": window_days,
            "maturity_known": recent["maturity"]["known"],
            "n_decisions": recent["n_decisions"], "n_labelled": recent["n_labelled"],
            "censoring": recent.get("censoring", {})}

    if not recent.get("measurable"):
        why = recent.get("reason", "the recent window carries too few outcomes").rstrip(". ")
        steps["unmeasurable"] = (RULED_IN, why)
        return {**base, "verdict": "unmeasurable", "differential": _differential(steps),
                "reason": (f"{why}. "
                           f"This is a statement about the label supply, not about the model. "
                           f"Do not retrain on it."),
                "label_coverage": recent["label_coverage"]}

    steps["unmeasurable"] = (RULED_OUT,
                             f"{recent['n_labelled']} of {recent['n_decisions']} decisions carry "
                             f"an outcome, at or above the {MIN_LABELLED} needed to measure")

    if not priors:
        steps["no_baseline"] = (RULED_IN,
                                "no earlier window clears the label floor, so the recent figures "
                                "are a single point and a point has no direction")
        return {**base, "verdict": "no_baseline", "differential": _differential(steps),
                "reason": ("no earlier window has enough outcomes to compare against, so the "
                           "recent figures stand alone and a direction cannot be read from one "
                           "point"),
                "metrics": recent["metrics"]}

    prev = priors[-1]
    steps["no_baseline"] = (RULED_OUT,
                            f"comparing against {prev['window']['start'][:10]} to "
                            f"{prev['window']['end'][:10]}, {prev['n_labelled']} labelled rows")
    return _attribute(store, prev, recent, base, steps)


def main():
    from .store import Store
    s = Store()
    d = diagnose(s)
    print(f"verdict: {d['verdict']}\n  {d['reason']}\n")
    for w in trend(s)["windows"]:
        m = w.get("metrics") or {}
        print(f"{w['window']['start'][:10]} -> {w['window']['end'][:10]}  "
              f"decisions {w['n_decisions']:>5}  labelled {w['n_labelled']:>4}  "
              f"coverage {w['label_coverage']:.2%}  "
              + (f"P {m['precision_on_allowed']} R {m['recall_on_allowed']}"
                 if w.get("measurable") else "not measurable"))


if __name__ == "__main__":
    main()
