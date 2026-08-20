"""
experiments/card_joint_cells.py - does the card feature space support a joint-cell detector?

WHY THIS RUNS BEFORE ANY DETECTOR GETS WRITTEN. The card rail has no unsupervised layer. The
obvious move is to port the push rail's novelty gate, which is an IsolationForest, and prior-art
research argues that is the wrong algorithm family here for a mechanical reason: iForest splits
on numeric axes, and on one-hot categoricals ANY random threshold on a column that is 2% ones
perfectly isolates the ones in a single cut. Aggregated over trees it degenerates into ranking by
MARGINAL rarity of individual field values. Card fraud in an authorization is almost never a rare
VALUE. It is a rare COMBINATION of ordinary values: ecom + not tokenized + AVS match + CVV
no-match + high-risk MCC, where every field alone is unremarkable and only the joint cell is odd.

So the proposed alternative is a joint-cell model with two terms:

    rarity      -log P(cell)                          how unusual is this exact combination
    dependency   log P(cell) - SUM log P(field_i)     PMI: is it rarer than its parts predict

The dependency term is the answer to "unusual is not fraudulent". A customer having an atypical
day produces a cell that is rare AND whose individual fields are rare, so the ratio is near zero
and it does not fire. A record where every field is common but the combination has never
co-occurred produces a large negative PMI. That is the shape this file is testing for.

WHAT THIS FILE CANNOT TELL YOU, stated up front. auth_ledger.csv is the card model's TRAINING
set: 240k train + 120k test + 40k calibration = 400,000 = the whole file. There is no
out-of-typology card holdout anywhere in this project (challenge_ledger.csv carries no card
authorization fields at all, so it cannot grade the card rail). Any fraud separation measured
here is therefore IN-SAMPLE, and this project has already been burned once by exactly that:
the push model's AUC 0.969 was measured on a ledger whose flagship typology could not exist.
Structure (cardinality, coverage, concentration) is safe to read from in-sample data. Separation
is NOT a performance claim until a card challenge set exists.

Usage:  python3 experiments/card_joint_cells.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter, defaultdict

LEDGER = os.path.expanduser("~/pulseml_models/auth_ledger.csv")

# The card model's own categorical set, plus the two low-cardinality flags it treats as numeric.
# Deliberately NOT included: bin_fraud_rate and merchant_fraud_rate (target-encoded, see below),
# amount and amount_log (continuous, and duplicates of each other).
CELL_FIELDS = ("entry_mode", "channel", "card_type", "avs_result", "cvv_result",
               "three_ds", "tokenized")

HIGH_RISK_MCC = {"5967", "6051", "7995", "5122", "5912", "4816", "5816"}


def truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "t")


def load():
    rows = []
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            mcc = str(r.get("mcc_code", "") or "").split(".")[0]
            r["mcc_high_risk"] = "1" if mcc in HIGH_RISK_MCC else "0"
            rows.append(r)
    return rows


def cell_of(r) -> tuple:
    return tuple(str(r.get(f, "") or "").strip() for f in CELL_FIELDS)


def main():
    if not os.path.exists(LEDGER):
        print(f"missing {LEDGER}")
        return 1

    rows = load()
    n = len(rows)
    fraud_rows = [r for r in rows if truthy(r.get("is_fraud"))]
    print(f"auth_ledger.csv: {n:,} rows, {len(fraud_rows):,} fraud "
          f"({100*len(fraud_rows)/n:.3f}%)\n")

    # ---- 1. cardinality and the size of the joint space --------------------
    print("=" * 72)
    print("1. FIELD CARDINALITY AND THE SIZE OF THE JOINT SPACE")
    print("=" * 72)
    possible = 1
    for f in CELL_FIELDS:
        vals = Counter(str(r.get(f, "") or "").strip() for r in rows)
        possible *= max(1, len(vals))
        top = ", ".join(f"{k or '<empty>'}={v*100/n:.1f}%" for k, v in vals.most_common(4))
        print(f"  {f:<14} {len(vals):>3} values   {top}")

    cells = Counter(cell_of(r) for r in rows)
    print(f"\n  possible cells (product of cardinalities): {possible:,}")
    print(f"  OBSERVED cells                            : {len(cells):,}"
          f"   ({100*len(cells)/max(possible,1):.1f}% of the space)")
    print("  -> the space is small enough to estimate the joint directly, which is the whole"
          "\n     premise of a joint-cell model rather than a tree ensemble.")

    # ---- 2. concentration ---------------------------------------------------
    print("\n" + "=" * 72)
    print("2. VOLUME CONCENTRATION")
    print("=" * 72)
    ordered = cells.most_common()
    for k in (1, 5, 10, 20, 50, 100):
        if k <= len(ordered):
            share = sum(c for _, c in ordered[:k]) / n
            print(f"  top {k:>3} cells carry {share*100:5.1f}% of all volume")
    singles = sum(1 for _, c in ordered if c == 1)
    print(f"  cells seen exactly once: {singles:,} of {len(cells):,}")

    # ---- 3. where does fraud actually live? --------------------------------
    print("\n" + "=" * 72)
    print("3. FRAUD BY CELL  (in-sample, see module docstring)")
    print("=" * 72)
    fraud_cells = Counter(cell_of(r) for r in fraud_rows)
    print(f"  distinct cells containing >=1 fraud: {len(fraud_cells):,} of {len(cells):,}")
    print(f"\n  {'n':>7} {'fraud':>6} {'rate':>8}   cell")
    for cell, fc in fraud_cells.most_common(8):
        tot = cells[cell]
        desc = " | ".join(f"{v or '-'}" for v in cell)
        print(f"  {tot:>7,} {fc:>6,} {100*fc/tot:>7.2f}%   {desc}")

    # ---- 4. THE test: does PMI separate fraud from legitimate? -------------
    print("\n" + "=" * 72)
    print("4. THE DEPENDENCY (PMI) TERM  -- the question this file exists to answer")
    print("=" * 72)

    marg = {f: Counter(str(r.get(f, "") or "").strip() for r in rows) for f in CELL_FIELDS}

    def pmi(cell) -> float:
        """log P(cell) - SUM log P(field). Negative = rarer than its parts predict."""
        p_cell = cells[cell] / n
        if p_cell <= 0:
            return 0.0
        s = 0.0
        for f, v in zip(CELL_FIELDS, cell):
            p = marg[f][v] / n
            if p <= 0:
                return 0.0
            s += math.log(p)
        return math.log(p_cell) - s

    def rarity(cell) -> float:
        return -math.log(max(cells[cell] / n, 1e-12))

    pmi_cache = {c: pmi(c) for c in cells}
    rar_cache = {c: rarity(c) for c in cells}

    f_pmi = sorted(pmi_cache[cell_of(r)] for r in fraud_rows)
    l_pmi = sorted(pmi_cache[cell_of(r)] for r in rows if not truthy(r.get("is_fraud")))
    f_rar = sorted(rar_cache[cell_of(r)] for r in fraud_rows)
    l_rar = sorted(rar_cache[cell_of(r)] for r in rows if not truthy(r.get("is_fraud")))

    def pct(a, q):
        if not a:
            return float("nan")
        return a[min(len(a) - 1, int(len(a) * q))]

    print(f"  {'':<12}{'p10':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p90':>9}")
    for label, arr in (("FRAUD  pmi", f_pmi), ("legit  pmi", l_pmi),
                       ("FRAUD  rarity", f_rar), ("legit  rarity", l_rar)):
        print(f"  {label:<12}" + "".join(f"{pct(arr,q):>9.3f}"
                                          for q in (0.10, 0.25, 0.50, 0.75, 0.90)))

    # separation: AUC of each score alone, computed by rank (no sklearn needed)
    def auc(pos, neg):
        """Probability a random positive outranks a random negative. Rank-based, ties handled."""
        allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
        r_sum, i = 0.0, 0
        rank = 1
        while i < len(allv):
            j = i
            while j < len(allv) and allv[j][0] == allv[i][0]:
                j += 1
            avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
            for k in range(i, j):
                if allv[k][1] == 1:
                    r_sum += avg_rank
            rank += (j - i)
            i = j
        npos, nneg = len(pos), len(neg)
        if npos == 0 or nneg == 0:
            return float("nan")
        return (r_sum - npos * (npos + 1) / 2.0) / (npos * nneg)

    # LOWER pmi and HIGHER rarity are both meant to indicate fraud, so orient accordingly
    auc_pmi = auc([-v for v in f_pmi], [-v for v in l_pmi])
    auc_rar = auc(f_rar, l_rar)
    print(f"\n  AUC, negative-PMI alone as a fraud score : {auc_pmi:.4f}")
    print(f"  AUC, rarity alone as a fraud score       : {auc_rar:.4f}")
    print("  (0.500 = no signal. These are IN-SAMPLE and are structure, not performance.)")

    # ---- 5. confirm the two feature defects the research flagged -----------
    print("\n" + "=" * 72)
    print("5. THE TWO FEATURE DEFECTS, CONFIRMED ON THIS DATA")
    print("=" * 72)
    bin_fraud = defaultdict(lambda: [0, 0])
    for r in rows:
        b = str(r.get("bin", "") or "")
        bin_fraud[b][0] += 1
        if truthy(r.get("is_fraud")):
            bin_fraud[b][1] += 1
    clean_bins = sum(1 for b, (t, f) in bin_fraud.items() if f == 0)
    print(f"  distinct BINs: {len(bin_fraud):,}   with ZERO fraud in this ledger: {clean_bins:,}"
          f" ({100*clean_bins/max(len(bin_fraud),1):.1f}%)")
    print("  -> bin_fraud_rate is TARGET-ENCODED. A genuinely new typology runs on a clean BIN")
    print("     by construction, so the feature reads LOW exactly when novelty is highest. It")
    print("     imports the supervised model's blind spot into an unsupervised layer and must")
    print("     be excluded from any novelty detector. Same argument for merchant_fraud_rate.")
    print("  -> amount and amount_log are a monotone transform of one another. iForest picks")
    print("     split features uniformly, so carrying both silently doubles amount's weight.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if auc_pmi > 0.60 or auc_rar > 0.60:
        print("  A joint-cell model has real in-sample signal. Next step is a CARD CHALLENGE SET")
        print("  (card typologies the model was never trained on, in the auth_ledger schema),")
        print("  because in-sample separation is exactly the number this project has already")
        print("  been burned by once.")
    else:
        print("  Neither term separates fraud on its own in-sample. The signal is more likely in")
        print("  the NUMERICS CONDITIONED ON THE CELL (per-cell amount percentile) than in cell")
        print("  rarity itself. Build the conditional-amount model instead of the joint-cell one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
