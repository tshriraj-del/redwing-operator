"""
core/label_maturity.py - how complete is a window's label set, and what is safe to train on.

WHY. Fraud labels arrive late, and the lateness is not random. Card fraud reports in days.
The scams that do the damage on an irrevocable rail, grooming, pig butchering, invoice
redirection, surface in weeks or months, because the victim does not know yet. So "every label
I have today" is a sample in which the recent window is systematically missing its positives.

Train on it and the model learns that recent traffic is safe. Measure drift on it and drift
reports IMPROVEMENT exactly while things deteriorate, because this month's fraud has not been
reported yet. Open the graduation gate on it and you certify a model against a cohort whose
fraud is still in the post. Every one of those failures is silent: nothing errors, the label set
simply looks finished.

The store was built for this and then never used it. `labels.effective_ts` is documented "when
the labeled fact became true (for latency)", `labels.source` anticipates confirmed_loss /
chargeback / victim_report, `store.training_rows()` returns decided_ts and labeled_ts on every
row. Nothing read any of it. This module is what reads it.

WHAT IT MEASURES

    arrival lag = (effective_ts or labeled_ts) - decided_ts

`effective_ts` is when the fact became true and is preferred whenever it is populated, because
it is immune to when the row happened to be written. `labeled_ts` is the fallback, and the
fallback is the one that can lie.

From the lags it derives F(d), the share of a cohort's eventual labels in hand by day d, and
inverts it: `maturity_floor(coverage=0.9)` is the most recent decision date whose labels are 90%
complete. Decisions after that floor are IMMATURE and must not be trained on, measured on, or
graduated against.

THE THREE THINGS THIS MODULE REFUSES TO DO

1. It will not put a machine's own call in the denominator. A heuristic self-label is written at
   score time and so has an arrival lag of zero BY CONSTRUCTION; it is a prediction, not a
   report from the world. Include those and the curve collapses toward zero and declares
   everything mature, which is the precise failure this module exists to prevent. So the curve
   is derived from GOLD sources only. This is a filter of principle, not a tuned threshold, and
   it is the one doing most of the work: on this instance it takes outcome.is_fraud from 380
   labels to 5.

2. It will not guess which bulk arrivals are real. A daily chargeback file and a retro-write
   look identical in the data (see SINGLE_ARRIVAL_MIN_LABELS), so this module reports them and
   declines to exclude either. Provenance settles it, and `effective_ts` is the provenance.

3. `mature_only` will not silently pass everything through when the curve is unavailable. A
   filter that quietly does nothing when it cannot compute is worse than no filter, because the
   caller believes maturity was enforced. It refuses instead, and says why.

THE HORIZON IS AN ASSUMPTION, NOT A FINDING. F(d) can only be estimated from cohorts old enough
to be complete, and "old enough" is defined by the quantity being estimated. That circularity is
real and unbreakable, so it is declared rather than hidden: `horizon_days` is the age at which a
cohort is treated as settled, it defaults to 90, and the curve reports whether its own tail is
pressed against that horizon (`horizon_truncated`), which is the observable symptom of having
set it too short.

Pure stdlib, deterministic, no ML stack.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# How old a cohort must be before its labels are treated as complete. AN ASSUMPTION.
# 90 days is the conventional performance window for push-payment fraud, long enough to
# capture victim-initiated reports and beneficiary-bank recalls. Override per portfolio.
DEFAULT_HORIZON_DAYS = 90

# What share of a cohort's eventual labels must be in hand before it counts as mature.
DEFAULT_COVERAGE = 0.9

# Below this, an empirical lag distribution is noise and no curve is returned.
MIN_LABELS_FOR_CURVE = 30

# A "single-arrival bucket" is a labelled day whose rows all share one labeled_ts, so each
# row's lag is fixed by its decision date alone and the lag range equals the decision span.
# These are REPORTED, NOT EXCLUDED, and the reason is worth the paragraph.
#
# The first version excluded them, on the theory that they are scripts. Then consider what a
# real bank's label supply actually looks like: the chargeback file lands every morning, and
# every row in Tuesday's file carries Tuesday. Decisions inside it span the last four months.
# That is a single-arrival bucket with a perfect identity, and its lag is entirely REAL: the
# chargeback genuinely did arrive on Tuesday. Excluding it would throw away the single most
# valuable label source in the building and leave the curve derived from whatever ad-hoc
# labelling remained.
#
# The batch SHAPE cannot distinguish that file from a backfill that stamped today onto facts
# known months ago. Both look identical in the data, because the difference is not in the
# distribution, it is in the pipeline: whether labeled_ts is when the information arrived or
# just when a row got written. No estimator can recover that from the timestamps.
#
# So this module stops trying. Integrity comes from provenance instead, and rests on two
# things that ARE knowable: gold sources only (a machine's own call is never a report from the
# world), and effective_ts preferred wherever populated (which makes write time irrelevant).
# The bucket report is left in as an operator signal: if one of these turns out to be
# retro-written, the curve overstates lag, and the fix is to populate effective_ts on that
# pipeline rather than to add a cleverer detector here.
SINGLE_ARRIVAL_MIN_LABELS = 20
# How near-perfect the identity must be to call it one arrival. Slack for clock skew and for a
# job straddling midnight.
SINGLE_ARRIVAL_TOL = 0.02

GOLD_SOURCES = ("analyst", "confirmed_loss", "chargeback", "victim_report", "law_enforcement")


def _parse(ts) -> datetime | None:
    """ISO-8601 to an aware datetime, or None. Tolerant because these strings arrive from
    connectors, CSVs and other institutions, and a malformed one must cost a row, not a run."""
    if not ts:
        return None
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _day(d: datetime) -> str:
    return d.date().isoformat()


def lag_rows(store, label_space: str, label_key: str, sources=None) -> list:
    """Every label for a target that can carry a lag.

    Uses training_rows() because a lag needs BOTH timestamps, and only a label joined to its
    decision has both. A label on a case this instance never scored has no decided_ts and so no
    measurable lag: it is dropped here and counted separately by the caller, because a substrate
    full of unjoinable labels is itself the finding.

    `lag_days` prefers effective_ts over labeled_ts. When the store knows WHEN THE FACT BECAME
    TRUE, that is the real arrival lag, and it does not care when the row was written; a year of
    chargebacks imported this morning still carries its true lags. labeled_ts is the fallback,
    and the fallback is what batch_writes() exists to police.
    """
    out = []
    for r in store.training_rows(label_space, label_key, sources=sources, limit=1_000_000):
        dec, lab = _parse(r.get("decided_ts")), _parse(r.get("labeled_ts"))
        if not dec or not lab:
            continue
        eff = _parse(r.get("effective_ts"))
        known = eff or lab
        out.append({
            "subject_ref": r.get("subject_ref", ""),
            "source": r.get("source", ""),
            "decided": dec,
            "labeled": lab,
            "effective": eff,
            "lag_days": (known - dec).total_seconds() / 86400.0,
            "lag_basis": "effective_ts" if eff else "labeled_ts",
            # how long the fact sat in the world before it reached us, when both are known
            "reporting_lag_days": ((lab - eff).total_seconds() / 86400.0) if eff else None,
        })
    return out


def single_arrival_buckets(rows) -> list:
    """Labelled days whose rows all arrived at one instant. REPORTED, NOT EXCLUDED.

    Detected by the identity at SINGLE_ARRIVAL_TOL: one labeled_ts across the bucket means each
    row's lag is determined entirely by its decision date, so the spread of lags equals the
    spread of decision dates.

    This is the normal shape of a daily chargeback or recall file, where the lag is real, AND
    the shape of a backfill, where it is not. See the note at SINGLE_ARRIVAL_MIN_LABELS for why
    nothing here can tell those apart and why the answer is provenance rather than a detector.

    Rows carrying effective_ts are skipped: their lag no longer depends on when the row was
    written, so the bucket says nothing about them.
    """
    buckets: dict = {}
    for r in rows:
        if r.get("lag_basis") == "effective_ts":
            continue
        buckets.setdefault(_day(r["labeled"]), []).append(r)

    found = []
    for day, rs in sorted(buckets.items()):
        if len(rs) < SINGLE_ARRIVAL_MIN_LABELS:
            continue
        decided = [r["decided"] for r in rs]
        span = (max(decided) - min(decided)).total_seconds() / 86400.0
        lags = [r["lag_days"] for r in rs]
        lag_range = max(lags) - min(lags)
        if span <= 0:
            continue                      # all decided at one instant: no identity to test
        if abs(lag_range - span) > SINGLE_ARRIVAL_TOL * span:
            continue                      # lag varies independently of decision date
        found.append({
            "labeled_day": day,
            "labels": len(rs),
            "decision_span_days": round(span, 2),
            "lag_range_days": round(lag_range, 2),
            "sources": sorted({r["source"] for r in rs}),
            "note": ("all of these arrived at one instant, so their lag is fixed by their "
                     "decision date. Real if this is a daily outcome file; fictitious if it "
                     "is a retro-write of facts known earlier. Populate effective_ts on this "
                     "pipeline to settle it."),
        })
    return found


def _ecdf_at(lags, d: float) -> float:
    return sum(1 for x in lags if x <= d) / len(lags)


def _percentile(sorted_lags, p: float) -> float:
    """Nearest-rank percentile. Exact on small samples, which is what this always has."""
    if not sorted_lags:
        return 0.0
    k = max(1, min(len(sorted_lags), int(round(p * len(sorted_lags) + 0.5))))
    return sorted_lags[k - 1]


def lag_curve(store, label_space: str, label_key: str, *, sources=GOLD_SOURCES,
              horizon_days: int = DEFAULT_HORIZON_DAYS,
              coverage: float = DEFAULT_COVERAGE, as_of=None) -> dict:
    """F(d): the share of a cohort's eventual labels in hand by day d, or a refusal.

    `sources` defaults to GOLD only, and that default is load-bearing. A heuristic label is the
    machine's own call written at score time; its lag is zero by construction because nothing
    had to arrive. Letting those into the sample would pull the curve to zero and certify every
    cohort as mature the moment it was scored, which is precisely the false assurance this
    module exists to deny. Widen it only to answer a different question than maturity.

    Estimated only over SETTLED cohorts, meaning decisions older than `horizon_days`, because a
    recent cohort is still receiving labels and including it would bias the curve toward short
    lags: the long-lag labels it is missing are precisely the ones being measured.
    """
    now = _parse(as_of) or _now()
    settle_before = now - timedelta(days=horizon_days)

    all_rows = lag_rows(store, label_space, label_key, sources=list(sources) if sources else None)
    settled = [r for r in all_rows if r["decided"] <= settle_before]
    buckets = single_arrival_buckets(settled)

    base = {
        "target": f"{label_space}.{label_key}",
        "as_of": now.isoformat().replace("+00:00", "Z"),
        "horizon_days": horizon_days,
        "coverage_target": coverage,
        "sources": list(sources) if sources else ["*"],
        "labels_with_both_timestamps": len(all_rows),
        "settled_cohort_labels": len(settled),
        "lag_from_effective_ts": sum(1 for r in settled if r["lag_basis"] == "effective_ts"),
        "single_arrival_buckets": buckets,
    }

    if len(settled) < MIN_LABELS_FOR_CURVE:
        base.update({
            "derivable": False,
            "reason": (
                f"{len(settled)} labels from {base['sources']} in settled cohorts (decisions "
                f"older than {horizon_days}d), need >= {MIN_LABELS_FOR_CURVE}. No curve is "
                f"returned rather than one fitted to too little; an invented maturity curve "
                f"would license training on exactly the immature window it exists to "
                f"withhold."),
        })
        return base

    lags = sorted(max(0.0, r["lag_days"]) for r in settled)
    p50, p90, p95 = (_percentile(lags, 0.50), _percentile(lags, 0.90), _percentile(lags, 0.95))

    # smallest observed lag at which coverage is met
    d_cov = next((d for d in lags if _ecdf_at(lags, d) >= coverage), lags[-1])

    base.update({
        "derivable": True,
        "n": len(lags),
        "lag_days": {"p50": round(p50, 2), "p90": round(p90, 2), "p95": round(p95, 2),
                     "max": round(lags[-1], 2), "mean": round(sum(lags) / len(lags), 2)},
        "curve": [{"day": d, "coverage": round(_ecdf_at(lags, d), 3)}
                  for d in (1, 3, 7, 14, 30, 60, 90, 120, 180) if d <= horizon_days],
        "days_to_coverage": round(d_cov, 2),
        # The observable symptom of a horizon set too short: the tail is pressed against it,
        # so labels beyond it exist and were never seen, and the curve is truncated rather
        # than complete. Reported, not silently corrected, because the fix is a judgement
        # about the portfolio's reporting behaviour and not something to infer from the data.
        "horizon_truncated": bool(p95 >= 0.9 * horizon_days),
        "reporting_lag_measurable": sum(1 for r in settled if r["reporting_lag_days"] is not None),
    })
    if base["horizon_truncated"]:
        base["horizon_warning"] = (
            f"p95 lag ({round(p95, 1)}d) is within 10% of the {horizon_days}d horizon, so the "
            f"curve is probably truncated: labels arriving later than the horizon were never "
            f"observed and coverage is overstated. Raise horizon_days and re-derive.")
    return base


def maturity_floor(store, label_space: str, label_key: str, **kw) -> dict:
    """The most recent decision date whose labels are `coverage` complete.

    Decisions after this date are immature. This is the number every other caller wants, so
    it carries the curve's refusal through unchanged rather than substituting a default: a
    floor that silently means "no floor" is how an unenforced filter passes for an enforced
    one.
    """
    curve = lag_curve(store, label_space, label_key, **kw)
    if not curve.get("derivable"):
        return {"known": False, "floor": None, "reason": curve["reason"], "curve": curve}
    now = _parse(curve["as_of"]) or _now()
    floor = now - timedelta(days=float(curve["days_to_coverage"]))
    return {
        "known": True,
        "floor": floor.isoformat().replace("+00:00", "Z"),
        "days_to_coverage": curve["days_to_coverage"],
        "coverage": curve["coverage_target"],
        "reason": (f"{curve['coverage_target']:.0%} of a cohort's labels are in hand by "
                   f"{curve['days_to_coverage']}d, so decisions after {_day(floor)} are "
                   f"still accumulating labels"),
        "curve": curve,
    }


def partition(rows, floor_iso: str) -> dict:
    """Split training rows into mature and immature by their DECISION date.

    Keyed on decided_ts, not labeled_ts, which is the whole point. A label written today about
    a decision made today is not mature; it is one early report from a cohort whose other
    labels have not arrived. Maturity is a property of the cohort, never of the label.
    """
    floor = _parse(floor_iso)
    if not floor:
        return {"mature": list(rows), "immature": [], "applied": False}
    mature, immature = [], []
    for r in rows:
        dec = _parse(r.get("decided_ts"))
        (mature if (dec and dec <= floor) else immature).append(r)
    return {"mature": mature, "immature": immature, "applied": True}


def maturity_report(store, targets=None, **kw) -> dict:
    """Label maturity across the trainable targets, plus what it means for each."""
    if targets is None:
        targets = [("outcome", "is_fraud"), ("intent", "motive"),
                   ("intent", "witting_role"), ("intent", "scam_stage")]
    out = []
    for sp, ky in targets:
        f = maturity_floor(store, sp, ky, **kw)
        entry = {"target": f"{sp}.{ky}", "known": f["known"], "floor": f.get("floor"),
                 "reason": f["reason"]}
        c = f["curve"]
        entry["labels_with_both_timestamps"] = c["labels_with_both_timestamps"]
        entry["settled_cohort_labels"] = c["settled_cohort_labels"]
        entry["lag_from_effective_ts"] = c["lag_from_effective_ts"]
        entry["single_arrival_buckets"] = c["single_arrival_buckets"]
        if f["known"]:
            entry["lag_days"] = c["lag_days"]
            entry["days_to_coverage"] = c["days_to_coverage"]
            gold = store.training_rows(sp, ky, sources=list(GOLD_SOURCES), limit=1_000_000)
            part = partition(gold, f["floor"])
            entry["gold_mature"] = len(part["mature"])
            entry["gold_immature"] = len(part["immature"])
        out.append(entry)
    return {"targets": out, "horizon_days": kw.get("horizon_days", DEFAULT_HORIZON_DAYS),
            "coverage_target": kw.get("coverage", DEFAULT_COVERAGE)}


def main():
    from .store import Store
    rep = maturity_report(Store())
    print(f"horizon {rep['horizon_days']}d, coverage target {rep['coverage_target']:.0%}\n")
    for t in rep["targets"]:
        print(f"{t['target']}")
        print(f"  gold labels with both timestamps : {t['labels_with_both_timestamps']}")
        print(f"  of those, in settled cohorts     : {t['settled_cohort_labels']}")
        print(f"  lag taken from effective_ts      : {t['lag_from_effective_ts']}")
        for b in t["single_arrival_buckets"]:
            print(f"  single arrival: {b['labels']} labels on {b['labeled_day']} "
                  f"covering {b['decision_span_days']}d of decisions {b['sources']}")
        if t["known"]:
            print(f"  lag p50/p90/p95 : {t['lag_days']['p50']} / {t['lag_days']['p90']} / "
                  f"{t['lag_days']['p95']} days")
            print(f"  maturity floor  : {t['floor']}")
            print(f"  gold mature/immature : {t['gold_mature']} / {t['gold_immature']}")
        else:
            print(f"  NOT DERIVABLE: {t['reason']}")
        print()


if __name__ == "__main__":
    main()
