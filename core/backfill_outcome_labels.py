"""
core/backfill_outcome_labels.py - give historical decisions the machine call they never recorded.

WHY. Adjudication only pays off if a human label lands on a subject the machine already spoke
about: graduation.py measures AGREEMENT, and a gold label on a case with no prediction beside it
raises gold_labels and leaves paired_with_heuristic untouched. Measured before this, ZERO
subjects in the substrate carried both, so no amount of labelling could ever fill the gate.

main.py now records the model's call on every new decision. This is the other half: the
decisions already in the store predate that, so the analyst labels sitting on them still cannot
pair. The stored decision carries its score, so the call can be recovered rather than re-run.

THE HONEST WRINKLE, and why the annotator differs. record_decision() stores `cascade_score`,
this institution's own book BEFORE the network view; the live path records
is_alert(network_score), AFTER it. They are two different quantities and backfilling one under
the other's name would quietly mix them in a column somebody later computes kappa over. So a
backfilled row is annotated `model_local_score_call` and a live one `model_score_call`, and
anything comparing them can tell which it is holding.

Idempotent: a subject that already has a heuristic outcome label is skipped, so re-running adds
nothing. Reports what it would do under --dry-run, which is the default.

    python3 -m core.backfill_outcome_labels            # dry run, prints the plan
    python3 -m core.backfill_outcome_labels --apply    # writes
"""

from __future__ import annotations

import argparse

from match_engine import is_alert

from .store import Store

ANNOTATOR = "model_local_score_call"


def plan(store: Store) -> dict:
    """What the backfill would do, without doing it."""
    c = store._conn
    # Skip a subject that carries ANY outcome label already, not just a heuristic one.
    #
    # add_label SUPERSEDES the current label for a target, so writing a machine call onto a
    # subject an analyst has already judged marks the human label superseded. The first version
    # of this did exactly that to two of the five gold labels in the store, which is the worst
    # possible outcome for a script whose entire purpose is making human judgment usable.
    #
    # Those two are simply not recoverable as PAIRS from here: pairing wants the machine call to
    # predate the human one, and a backfill runs after both. Going forward the ordering is right
    # by construction, because main.py records the call at scoring time, long before anyone
    # adjudicates. Preserving the gold matters more than manufacturing two pairs.
    have = {r["subject_ref"] for r in c.execute(
        "SELECT DISTINCT subject_ref FROM labels "
        "WHERE label_space='outcome' AND label_key='is_fraud'"
    ).fetchall()}
    rows = c.execute(
        "SELECT decision_id, subject_ref, entity_id, score FROM decisions "
        "WHERE score IS NOT NULL AND subject_ref<>''"
    ).fetchall()

    todo, skipped, calls = [], 0, {0: 0, 1: 0}
    seen = set()
    for r in rows:
        sub = r["subject_ref"]
        if sub in have or sub in seen:
            skipped += 1
            continue
        seen.add(sub)
        call = 1 if is_alert(float(r["score"])) else 0
        calls[call] += 1
        todo.append({"decision_id": r["decision_id"], "subject_ref": sub,
                     "entity_id": r["entity_id"], "score": float(r["score"]), "call": call})

    # how many of these would immediately become pairable with an existing gold label
    gold = {r["subject_ref"] for r in c.execute(
        "SELECT DISTINCT subject_ref FROM labels "
        "WHERE label_space='outcome' AND label_key='is_fraud' AND source<>'heuristic'"
    ).fetchall()}
    return {"to_write": todo, "skipped": skipped, "calls": calls,
            "would_pair": sum(1 for t in todo if t["subject_ref"] in gold),
            "gold_subjects": len(gold)}


def apply(store: Store, p: dict) -> int:
    n = 0
    for t in p["to_write"]:
        store.add_label("outcome", "is_fraud", t["call"], source="heuristic",
                        confidence=round(t["score"], 4), decision_id=t["decision_id"],
                        subject_ref=t["subject_ref"], entity_id=t["entity_id"] or "",
                        annotator=ANNOTATOR)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    a = ap.parse_args()
    s = Store()
    p = plan(s)
    print(f"decisions eligible : {len(p['to_write']):,}")
    print(f"  already labelled : {p['skipped']:,} (skipped, this is idempotent)")
    print(f"  model call = 1   : {p['calls'][1]:,}")
    print(f"  model call = 0   : {p['calls'][0]:,}")
    print(f"gold outcome labels in store : {p['gold_subjects']}")
    print(f"  of which would PAIR after this : {p['would_pair']}")
    if not a.apply:
        print("\ndry run, nothing written. re-run with --apply")
        return
    print(f"\nwrote {apply(s, p):,} labels as annotator={ANNOTATOR}")


if __name__ == "__main__":
    main()
