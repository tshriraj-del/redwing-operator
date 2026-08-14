"""
core/outcome_ledger.py - the outcomes the business already produces, turned into gold labels.

WHY. Of the five sources `graduation.GOLD_SOURCES` trusts as ground truth, exactly one had a
live production path: `analyst`, meaning a human clicking. `confirmed_loss` existed only inside
a replay harness. `chargeback`, `victim_report` and `law_enforcement` appeared in a constant
tuple and a comment and nowhere else. So the gold supply was one person's afternoon, against a
gate wanting 50 gold and 30 pairs per target, times four targets.

That is the wrong resource to optimise. A real fraud shop's labels are a BYPRODUCT of
operations: the customer disputes, the beneficiary bank sends a recall, the loss is confirmed
and reconciled, the scheme adjudicates a chargeback. Those arrive in volume, on their own,
carrying real dates. This module is the path in.

The connectors already existed (file, DB, webhook, checkpoints, HMAC), so almost none of this
is transport. The work is in three semantics nothing handled:

PRECEDENCE, which now lives in store.add_label() rather than here, because a rule enforced by
convention is a rule enforced by whoever remembers it. This repo already ran that experiment:
backfill_outcome_labels.py wrote machine calls over two of five analyst gold labels. The fix was
a skip-guard inside that one script, which protects nothing from the next writer, and a nightly
outcome feed is precisely the next writer. A weaker source now arrives already superseded: kept
as evidence, never winning.

REVERSAL. A chargeback gets represented and reversed. A confirmed fraud is re-adjudicated as
first-party abuse. The outcome flips, and the flip is not an error to be smoothed over: it is
the strongest single statement the business ever makes about a case. `reversals()` finds them,
and anything trained before one needs to know it happened.

DISAGREEMENT AS SIGNAL. The analyst cleared it; a chargeback arrives three weeks later saying
fraud. That row is a labelled FALSE NEGATIVE, discovered by the world rather than by us, and it
is the most valuable single record in the substrate. Before this it was buried in label_history
where nothing read it. `disagreements()` surfaces it.

WHAT IS REAL HERE AND WHAT IS NOT. REDWING has no live chargeback feed, because REDWING's ledger
is synthetic. The ingest path, the precedence resolution, the reversal handling and the
disagreement extraction are real code doing real work on whatever arrives. The arrival itself is
simulated by `seed_outcome_file()`, which is labelled as a simulation everywhere it is used and
must never be mistaken for a bank's data.

Pure stdlib. An outcome is written twice on purpose: as an EVENT on the backbone (the record,
with its amount, reason code and reference) and as a LABEL (the assertion the gate reads). The
event is what happened; the label is what we now believe.
"""

from __future__ import annotations

import json

from .store import FRAUD_FALSE, FRAUD_TRUE, LABEL_SOURCES, eid, precedence_of

# Outcome vocabulary as it arrives from the world, mapped to the stored representation.
FRAUD_WORDS = {"fraud", "confirmed_fraud", "1", "true", "yes", "loss"}
LEGIT_WORDS = {"legit", "legitimate", "not_fraud", "0", "false", "no", "clean", "reversed"}

# Sources that are outcome REPORTS rather than opinions. `analyst` is deliberately absent: an
# adjudication arrives through /feedback and close_loop(), and giving it a second door here
# would let the same judgement be counted twice under two annotators.
OUTCOME_SOURCES = ("confirmed_loss", "chargeback", "victim_report", "law_enforcement")


def normalise_outcome(value) -> str | None:
    """'fraud' / 'legit' / 1 / '0' / True to the stored "1" or "0", or None if unreadable.

    Returns None rather than guessing. An outcome file with a column this cannot read is a
    mapping bug in the integration, and defaulting it to "not fraud" would quietly bury real
    losses in the negative class, which is the same one-directional error label maturity exists
    to prevent.
    """
    if isinstance(value, bool):
        return FRAUD_TRUE if value else FRAUD_FALSE
    v = str(value if value is not None else "").strip().lower()
    if v in FRAUD_WORDS:
        return FRAUD_TRUE
    if v in LEGIT_WORDS:
        return FRAUD_FALSE
    return None


# What an outcome report is worth when the source does not say. Every source used to get this
# unconditionally; it is now only the fallback.
DEFAULT_CONFIDENCE = 0.95


def _clamp_confidence(v) -> float:
    """A confidence outside [0,1] is a bug in the caller, not a strong opinion."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    return min(1.0, max(0.0, f))


def validate(rec: dict) -> tuple:
    """(clean, error). Same shape as ingest_schema's validator, deliberately."""
    if not isinstance(rec, dict):
        return None, "record is not an object"
    subj = str(rec.get("subject_ref") or rec.get("transaction_id") or "").strip()
    if not subj:
        return None, "subject_ref (or transaction_id) is required"
    src = str(rec.get("source") or "").strip().lower()
    if src not in LABEL_SOURCES:
        return None, f"source {src!r} is not one of {list(LABEL_SOURCES)}"
    if src == "heuristic":
        return None, "heuristic is the machine's own call, not an outcome report"
    out = normalise_outcome(rec.get("outcome", rec.get("is_fraud")))
    if out is None:
        return None, (f"outcome {rec.get('outcome')!r} is unreadable; expected one of "
                      f"{sorted(FRAUD_WORDS | LEGIT_WORDS)}")
    return {
        "subject_ref": subj,
        "outcome": out,
        "source": src,
        # WHEN the fact became true. Without it the only lag measurable is how long until
        # somebody ran the import, which is a fact about scheduling. See core/label_maturity.
        "effective_ts": str(rec.get("effective_ts") or "").strip(),
        "reference": str(rec.get("reference") or "").strip(),
        "recipient_id": str(rec.get("recipient_id") or "").strip(),
        "amount": rec.get("amount"),
        "reason_code": str(rec.get("reason_code") or "").strip(),
        # A source that knows how sure it is may say so. This used to be hardcoded to 0.95 at
        # write time for every source and every record, which flattened real differences: on the
        # card rail a WITHDRAWN dispute and a contested win the merchant took on evidence are
        # both "legit" and are emphatically not equally strong, and training weighted them the
        # same. DEFAULT_CONFIDENCE preserves the old behaviour for every caller that says
        # nothing, so this is additive.
        "confidence": _clamp_confidence(rec.get("confidence")),
        "notes": str(rec.get("notes") or "").strip(),
    }, ""


def _event_id(rec: dict) -> str:
    """Stable id so re-reading yesterday's file writes nothing new. Prefers the source's own
    reference, because that is the only identifier the source guarantees is stable."""
    if rec["reference"]:
        return f"outcome:{rec['source']}:{rec['reference']}"
    return (f"outcome:{rec['source']}:{rec['subject_ref']}:{rec['outcome']}"
            f":{rec['effective_ts']}")


def record_outcome(store, rec: dict, *, override_reason: str = "") -> dict:
    """Record one outcome. Returns what actually happened to it, in the caller's language.

    `resolution` is one of:
      accepted    it is now the standing outcome for this subject
      duplicate   the same source already said the same thing at the same time; nothing written
      outranked   a stronger source stands; kept as evidence, did not win
      override    a weaker source deliberately overturned a stronger one, with a reason
    """
    clean, err = validate(rec)
    if err:
        return {"ok": False, "error": err}

    subj = clean["subject_ref"]
    standing = None
    for lbl in store.current_labels(subject_ref=subj):
        if lbl.label_space == "outcome" and lbl.label_key == "is_fraud":
            standing = lbl
            break

    if (standing and standing.source == clean["source"]
            and standing.label_value == clean["outcome"]
            and (standing.effective_ts or "") == clean["effective_ts"]):
        return {"ok": True, "resolution": "duplicate", "subject_ref": subj,
                "why": "same source, same outcome, same effective date"}

    rid = clean["recipient_id"]
    ent = eid("recipient", rid) if rid else ""

    # The record: what arrived, with everything the label has no room for.
    store.append_event(
        "outcome", entities=([ent] if ent else []), event_id=_event_id(clean),
        payload={k: clean[k] for k in
                 ("subject_ref", "source", "reference", "reason_code", "amount", "effective_ts")},
        derived={"is_fraud": int(clean["outcome"] == FRAUD_TRUE)},
    )

    # The assertion: what we now believe. Precedence is enforced inside add_label.
    outranked = bool(standing and not override_reason
                     and precedence_of(clean["source"]) < precedence_of(standing.source))
    label_id = store.add_label(
        "outcome", "is_fraud", clean["outcome"], source=clean["source"],
        confidence=clean["confidence"],
        subject_ref=subj, entity_id=ent, effective_ts=clean["effective_ts"],
        annotator=f"outcome_ledger:{clean['source']}",
        notes=json.dumps({k: clean[k] for k in
                          ("reference", "reason_code", "amount", "notes") if clean[k]},
                         sort_keys=True),
        override_reason=override_reason,
    )

    res = "outranked" if outranked else ("override" if (override_reason and standing)
                                         else "accepted")
    out = {"ok": True, "resolution": res, "subject_ref": subj, "label_id": label_id,
           "source": clean["source"], "outcome": clean["outcome"]}
    if standing:
        out["previous"] = {"source": standing.source, "outcome": standing.label_value}
        if standing.label_value != clean["outcome"]:
            out["disagrees_with_previous"] = True
            # Same source flipping its own earlier call is a reversal, not a dispute between
            # two parties. Worth naming, because the two mean very different things to a model
            # trained in between them.
            out["reversal"] = standing.source == clean["source"]
    return out


def ingest_outcomes(store, records, *, override_reason: str = "") -> dict:
    """Batch entry point for a file, a webhook body or a connector poll.

    `contradicted_standing` is NOT the same number as `disagreements()` returns, and the two
    were briefly both called "disagreements", which produced a run reporting 10 of one and 0 of
    the other in the same breath. The distinction is real and worth keeping straight:

      contradicted_standing   the incoming outcome differs from whatever stood before, and what
                              usually stood before is the MACHINE'S OWN CALL. So this is mostly
                              a count of model errors, which the graduation gate already
                              measures properly as kappa. Useful as an ingest-time signal, not
                              as evidence about a case.

      disagreements()         two sources that BOTH count as ground truth said different things.
                              That is the rare and valuable one: nobody's opinion is available
                              to dismiss, so the case has to be looked at.
    """
    counts = {"accepted": 0, "duplicate": 0, "outranked": 0, "override": 0, "rejected": 0}
    errors, flips = [], []
    for rec in records or []:
        r = record_outcome(store, rec, override_reason=override_reason)
        if not r.get("ok"):
            counts["rejected"] += 1
            if len(errors) < 20:
                errors.append({"record": rec, "error": r["error"]})
            continue
        counts[r["resolution"]] += 1
        if r.get("disagrees_with_previous"):
            flips.append(r)
    gold_disputes = [f for f in flips
                     if f.get("previous", {}).get("source") in OUTCOME_SOURCES + ("analyst",)]
    return {"counts": counts, "errors": errors, "total": sum(counts.values()),
            "contradicted_standing": flips,
            "gold_vs_gold_disputes": len(gold_disputes)}


# ---------------------------------------------------------------- reading the ledger back

def _outcome_history(store, subject_ref: str) -> list:
    """Every outcome label for a subject, oldest first, superseded ones included."""
    return [l for l in store.label_history(subject_ref=subject_ref)
            if l.label_space == "outcome" and l.label_key == "is_fraud"]


def disagreements(store, *, gold_only: bool = True, limit: int = 500) -> list:
    """Subjects where two sources that both count as ground truth said different things.

    THE point of this module, and the reason an outcome feed is worth more than its volume.
    When the analyst cleared a payment and a chargeback later says fraud, that is a labelled
    FALSE NEGATIVE found by the world rather than by us: a case the system let through, with
    the point-in-time features still attached, which is the most valuable single row a fraud
    model can be shown.

    `gold_only` excludes the machine's own call by default, because heuristic-versus-gold
    disagreement is what the graduation gate already measures as kappa. What is new here is
    gold-versus-gold, which nothing measured at all.
    """
    from .graduation import GOLD_SOURCES
    out = []
    seen = set()
    for row in store.labels_for_target("outcome", "is_fraud", limit=limit * 4):
        subj = row["subject_ref"]
        if not subj or subj in seen:
            continue
        seen.add(subj)
        hist = _outcome_history(store, subj)
        if gold_only:
            hist = [l for l in hist if l.source in GOLD_SOURCES]
        vals = {l.label_value for l in hist}
        if len(hist) < 2 or len(vals) < 2:
            continue
        first, last = hist[0], hist[-1]
        out.append({
            "subject_ref": subj,
            "first": {"source": first.source, "outcome": first.label_value, "ts": first.ts},
            "current": {"source": last.source, "outcome": last.label_value, "ts": last.ts},
            "kind": ("missed_fraud" if last.label_value == FRAUD_TRUE else "false_alarm"),
            "reversal": first.source == last.source,
            "why": ("the world later contradicted our standing call; the point-in-time features "
                    "are still attached to the decision, so this trains"),
        })
        if len(out) >= limit:
            break
    return out


def reversals(store, limit: int = 500) -> list:
    """Disagreements where the SAME source changed its own mind: a chargeback represented and
    won back, a confirmed fraud re-adjudicated as first-party abuse. Separated from
    cross-source disputes because a model trained between the two versions was trained on a
    label that no longer exists, and nothing else in the system would notice."""
    return [d for d in disagreements(store, limit=limit) if d["reversal"]]


def simulate_outcome_feed(store, *, n: int = 200, seed: int = 7, as_of=None) -> list:
    """SIMULATED outcome records over decisions already in the store. NOT A DATA SOURCE.

    REDWING's ledger is synthetic, so there is no chargeback file to read. Everything else in
    this module is real code doing real work on whatever arrives; this function is the part
    that pretends, and it is kept in one clearly named place so nothing downstream can mistake
    its output for a bank's data. Every record it emits carries source-specific lag drawn to
    resemble published reporting behaviour, which is the property that makes the maturity curve
    derivable at all:

        chargeback      slow and scheme-bound, weeks out
        confirmed_loss  slower, it waits on reconciliation
        victim_report   bimodal: fast when the victim notices at once, very slow when the scam
                        is a grooming arc and they do not know yet for months

    It reads the decisions' own scores to decide which cases became fraud, so the feed is
    correlated with the model rather than random, which is what makes the resulting
    disagreements resemble real missed fraud instead of noise.
    """
    import random
    from datetime import datetime, timedelta, timezone

    rng = random.Random(seed)
    now = as_of or datetime.now(timezone.utc)
    if isinstance(now, str):
        from .label_maturity import _parse
        now = _parse(now) or datetime.now(timezone.utc)

    rows = store._conn.execute(
        "SELECT subject_ref, entity_id, score, ts FROM decisions "
        "WHERE subject_ref<>'' AND score IS NOT NULL ORDER BY ts DESC LIMIT ?", (int(n * 3),)
    ).fetchall()

    lag_by_source = {
        "chargeback": lambda: rng.triangular(10, 45, 25),
        "confirmed_loss": lambda: rng.triangular(20, 90, 45),
        "victim_report": lambda: (rng.triangular(1, 10, 3) if rng.random() < 0.6
                                  else rng.triangular(30, 180, 70)),
    }
    out = []
    for r in rows[:n]:
        score = float(r["score"] or 0.0)
        is_fraud = rng.random() < min(0.95, max(0.01, score))
        src = rng.choices(list(lag_by_source), weights=[0.5, 0.2, 0.3])[0]
        if not is_fraud and src != "chargeback":
            continue                      # a clean payment rarely generates a report at all
        decided = r["ts"]
        from .label_maturity import _parse
        d = _parse(decided) or now
        eff = d + timedelta(days=lag_by_source[src]())
        if eff > now:
            continue                      # has not happened yet as of the simulated clock
        out.append({
            "subject_ref": r["subject_ref"],
            "outcome": "fraud" if is_fraud else "legit",
            "source": src,
            "effective_ts": eff.isoformat().replace("+00:00", "Z"),
            "reference": f"sim_{src[:2]}_{len(out):05d}",
            "reason_code": ("10.4" if src == "chargeback" else ""),
            "simulated": True,
        })
    return out


def ledger_stats(store) -> dict:
    """What the outcome supply actually looks like, by source."""
    c = store._conn
    by_source = {r["source"]: r["n"] for r in c.execute(
        "SELECT source, COUNT(*) n FROM labels WHERE label_space='outcome' "
        "AND label_key='is_fraud' AND superseded_by='' GROUP BY source").fetchall()}
    outranked = int(c.execute(
        "SELECT COUNT(*) n FROM labels WHERE label_space='outcome' AND label_key='is_fraud' "
        "AND notes LIKE '%outranked:%'").fetchone()["n"])
    with_eff = int(c.execute(
        "SELECT COUNT(*) n FROM labels WHERE label_space='outcome' AND label_key='is_fraud' "
        "AND superseded_by='' AND effective_ts<>''").fetchone()["n"])
    dis = disagreements(store)
    return {
        "current_outcomes_by_source": by_source,
        "outranked_writes": outranked,
        "with_effective_ts": with_eff,
        "disagreements": len(dis),
        "reversals": sum(1 for d in dis if d["reversal"]),
        "missed_fraud": sum(1 for d in dis if d["kind"] == "missed_fraud"),
    }
