"""
experiments/matched_pairs.py - the matched-pair set for the investigator-agent trial.

WHAT CHANGED, AND WHY, because the first version of this file was built on a broken premise.

  V1 sourced transactions.csv and matched on model score. MEASURED 2026-08-15, that data is the
  model's own TRAINING set: fraud scores p50 = 0.995, legitimate p50 = 0.000. There is no
  uncertain band to pair within, and above 0.3 there were 4 hard negatives in 200,000 rows. A
  matched-pair design needs same-score-opposite-label cases and in-sample scores do not produce
  any.

  V2 sources challenge_ledger.csv, which challenge_set.py writes and the training path never
  reads (tests/test_challenge_set.py asserts no leakage). On those 264,761 rows the model scores
  novel fraud at p50 = 0.002. It does not generalise to unseen typologies at all.

THE ROUTING KEY IS THE NOVELTY GATE, NOT A SCORE BAND. That follows from the number above: any
score-band router hides the model's blind spot, which is the only region where an investigator
adds anything. Measured on the same 264,761 rows, the novelty gate escalates 38.3% of novel fraud
against 3.8% of legitimate traffic, a 10.1x lift, giving a queue that is ~9.8% fraud. That queue
is the population this file samples from, and 9.8% is the precision an investigator arm has to
beat.

WHY BOTH SIDES OF A PAIR ARE GATE-ESCALATED. Pairing an escalated fraud against a quiet
legitimate row tests the gate, which is already measured. Pairing two cases the SYSTEM ALREADY
CONSIDERS EQUALLY SUSPICIOUS means the only thing left to separate them is evidence the score and
the gate never saw. That is the hypothesis.

WHAT THIS FILE CANNOT TEST. The device and sequence gates read entities, and the challenge
ledger's device-id namespace has ZERO overlap with the device graph (7,432 ids vs 29,739, no
intersection), so entity-based controls are unmeasurable here. Any arm depending on graph
cross-reference is limited to user/recipient linkage present in the ledger itself.

Usage:
    python experiments/matched_pairs.py --pairs 118
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_ALLOW_OPEN", "i-understand-this-is-open")
os.environ.setdefault("REDWING_RECOVERY_SECRET", "matched-pairs-experiment")

CHALLENGE = os.path.expanduser("~/pulseml_models/challenge_ledger.csv")

# PRE-REGISTERED. Fixed before any arm is run, echoed into the manifest so a later reader can
# tell whether they were tuned after seeing results.
SEED = 20260815
AMOUNT_DECILES = 10
REQUIRE_GATE_ESCALATION = True    # both sides of a pair must have been escalated
BASELINE_PRECISION = 0.098        # measured: the escalated queue is 9.8% fraud

MCC_GROUPS = {
    "crypto": {"crypto", "digital_goods"},
    "cash_like": {"money_transfer", "gambling", "prepaid"},
    "retail": {"retail", "grocery", "food", "clothing"},
    "travel": {"travel", "airline", "hotel", "car_rental"},
    "services": {"services", "utilities", "subscription", "healthcare"},
}


def _mcc_group(c) -> str:
    c = str(c or "").strip().lower()
    for name, members in MCC_GROUPS.items():
        if c in members:
            return name
    return "other"


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "t")


def score_and_gate(limit=None):
    """Score every row and record whether the novelty gate escalated it.

    The gate view travels with the case because `escalated` is the routing key, and because an
    arm that is handed only escalated cases must be able to show WHY each one arrived.
    """
    import pandas as pd
    import main

    t0 = time.perf_counter()
    df = pd.read_csv(CHALLENGE, low_memory=False, nrows=limit)
    rows = df.to_dict("records")
    print(f"  loaded {len(rows):,} challenge rows in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        feats = main.compute_features(r)
        s = float(main.ml_score_row(feats))
        raised, view = main.apply_novelty_gate(s, feats)
        r["_score"] = s
        r["_escalated"] = bool(view.get("escalated"))
        r["_anomaly"] = float(view.get("anomaly") or 0.0)
        r["_score_after"] = float(raised)
        if i and i % 50_000 == 0:
            print(f"  processed {i:,}...", flush=True)
    dt = time.perf_counter() - t0
    print(f"  scored+gated {len(rows):,} in {dt:.1f}s ({len(rows)/dt:,.0f}/sec)", flush=True)
    return rows


def _decile_bucket(rows):
    amounts = sorted(float(r.get("amount") or 0.0) for r in rows)
    n = len(amounts)
    edges = [amounts[min(n - 1, int(n * k / AMOUNT_DECILES))] for k in range(1, AMOUNT_DECILES)]

    def bucket(a):
        a = float(a or 0.0)
        for i, e in enumerate(edges):
            if a <= e:
                return i
        return AMOUNT_DECILES - 1
    return bucket, edges


def build_pairs(rows, want_pairs):
    pool = [r for r in rows if r["_escalated"]] if REQUIRE_GATE_ESCALATION else rows
    bucket, edges = _decile_bucket(rows)
    rng = random.Random(SEED)

    frauds = [r for r in pool if _truthy(r.get("is_fraud"))]
    legits = [r for r in pool if not _truthy(r.get("is_fraud"))]
    print(f"  escalated queue: {len(pool):,}  ({len(frauds):,} fraud / {len(legits):,} legit "
          f"= {100*len(frauds)/max(len(pool),1):.1f}% precision)")

    def stratum(r):
        return (str(r.get("payment_rail") or ""), _mcc_group(r.get("merchant_category")),
                bucket(r.get("amount")))

    index = defaultdict(list)
    for r in legits:
        index[stratum(r)].append(r)

    rng.shuffle(frauds)
    pairs, used = [], set()
    for f in frauds:
        if len(pairs) >= want_pairs:
            break
        cands = [c for c in index.get(stratum(f), []) if c["transaction_id"] not in used]
        if not cands:
            continue
        partner = rng.choice(cands)
        used.add(f["transaction_id"]); used.add(partner["transaction_id"])
        pairs.append((f, partner))

    print(f"  built {len(pairs)} matched pairs "
          f"(both sides gate-escalated, matched on rail + mcc group + amount decile)")
    return pairs, edges, len(pool), len(frauds)


def _case(r, pair_id, side):
    return {
        "pair_id": pair_id,
        "case_id": hashlib.sha256(f"{pair_id}:{r['transaction_id']}".encode()).hexdigest()[:16],
        "visible": {
            "transaction_id": r.get("transaction_id"), "user_id": r.get("user_id"),
            "device_id": r.get("device_id"), "recipient_id": r.get("recipient_id"),
            "amount": r.get("amount"), "timestamp": r.get("timestamp"),
            "payment_rail": r.get("payment_rail"),
            "merchant_category": r.get("merchant_category"), "mcc_code": r.get("mcc_code"),
            "hour": r.get("hour"), "day_of_week": r.get("day_of_week"),
            "model_score": round(r["_score"], 4),
            "novelty_anomaly": round(r["_anomaly"], 4),
            "escalated_to": round(r["_score_after"], 4),
            "why_here": "novelty gate escalated this case; the model scored it near zero",
        },
        "sealed": {
            "is_fraud": bool(_truthy(r.get("is_fraud"))),
            "actually_fraud": bool(_truthy(r.get("actually_fraud"))),
            "fraud_typology": str(r.get("fraud_typology") or ""),
            "scam_stage": str(r.get("scam_stage") or ""),
            "side": side,
        },
    }


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=118,
                    help="118/arm detects a 15-point effect at 80%% power, alpha 0.05")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "pairs.jsonl"))
    a = ap.parse_args()

    print("scoring + gating the challenge ledger (out-of-sample by construction)...")
    rows = score_and_gate(a.limit)
    print("building matched pairs from the escalated queue...")
    pairs, edges, pool_n, pool_f = build_pairs(rows, a.pairs)
    if not pairs:
        print("NO PAIRS BUILT.")
        return 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for i, (f, n) in enumerate(pairs):
            fh.write(json.dumps(_case(f, i, "fraud")) + "\n")
            fh.write(json.dumps(_case(n, i, "legit")) + "\n")

    typ = defaultdict(int)
    for f, _ in pairs:
        typ[str(f.get("fraud_typology") or "?")] += 1

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "challenge_ledger.csv (model never fitted to it)",
        "pre_registered": {
            "seed": SEED, "amount_deciles": AMOUNT_DECILES,
            "require_gate_escalation": REQUIRE_GATE_ESCALATION,
            "matched_on": ["payment_rail", "mcc_group", "amount_decile"],
            "routing_key": "novelty gate escalation, NOT a model score band",
            "baseline_to_beat": f"{BASELINE_PRECISION:.1%} precision in the escalated queue",
            "primary_metric": "false_clear_rate",
            "acceptance_bar": "false clear < 2.1%  (12 / (668 * 0.85))",
            "stopping_rule": "fail at 8 false clears in first 150; pass at 221 clears with <=2",
            "arms": {"A": "with cross-reference", "B": "features only"},
        },
        "measured_context": {
            "model_p50_on_novel_fraud": 0.002,
            "novelty_recall_on_novel_fraud": 0.383,
            "novelty_fpr_on_legit": 0.038,
            "escalated_queue_size": pool_n,
            "escalated_queue_fraud": pool_f,
            "device_gate": "UNMEASURABLE here: 0 id overlap with the device graph",
            "sequence_gate": "UNMEASURABLE here: store holds 15 card rows",
        },
        "pairs": len(pairs), "cases": len(pairs) * 2,
        "fraud_typologies": dict(typ),
        "amount_decile_edges": [round(e, 2) for e in edges],
    }
    with open(a.out.replace(".jsonl", ".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nwrote {len(pairs)*2} cases -> {a.out}")
    print("typologies in the fraud arm:", json.dumps(dict(typ), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
