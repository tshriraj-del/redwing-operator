"""
version_delta.py — MEASURE the training-serving-skew fix (the '0.3% -> 91%' claim).

Same retrained model, same alert threshold. Only the feature-reproduction path changes:
  • BROKEN serving  = raw passthrough — the 10 non-column features default to 0.0
                      (the exact pre-fix behavior, still in main.compute_features fallback)
  • FIXED serving   = the shared feature foundation recomputes all 23 identically to training

Catch = combined_score >= 0.65 (main's is_alert). Reports fraud catch-rate and legit
false-alert rate under each path — turning the asserted skew number into a measured one.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, random
sys.path.insert(0, os.path.expanduser("~/redwing-operator"))
import main
from match_engine import combined_score, is_alert, score_transaction

random.seed(20260629)
df = main.df_all
F = main.FEATURES


def feats_fixed(raw):
    return main.FEATURE_ENGINE.compute(raw)              # train == serve


def feats_broken(raw):
    return {f: float(raw.get(f, 0.0)) for f in F}        # 10 features -> 0.0 (pre-fix)


def alerted(feats):
    ml = main.ml_score_row(feats)
    m = score_transaction(feats)
    top = m[0] if m else None
    cs = combined_score(ml, top["confidence"]) if top else ml
    return is_alert(cs), ml


def run(idx):
    b_hits = f_hits = 0
    b_ml = []; f_ml = []
    for i in idx:
        raw = df.loc[i].to_dict()
        ab, mb = alerted(feats_broken(raw)); af, mf = alerted(feats_fixed(raw))
        b_hits += ab; f_hits += af; b_ml.append(mb); f_ml.append(mf)
    return b_hits, f_hits, sum(b_ml)/len(b_ml), sum(f_ml)/len(f_ml)


fr = random.sample(list(df[df["is_fraud"] == 1].index), 300)
lg = random.sample(list(df[df["is_fraud"] == 0].index), 1500)

fb, ff, fmlb, fmlf = run(fr)
lb, lf, lmlb, lmlf = run(lg)
nfr, nlg = len(fr), len(lg)

print("=" * 70)
print("TRAINING-SERVING SKEW — measured (same model, same 0.65 threshold)")
print("=" * 70)
print(f"Sample: {nfr} frauds, {nlg} legit.  10 of 23 features zero out on the broken path.")
print(f"\nFRAUD CATCH-RATE (frauds that fire an alert):")
print(f"  BROKEN serving (features->0) : {100*fb/nfr:5.1f}%   [{fb}/{nfr}]   mean ML score {fmlb:.3f}")
print(f"  FIXED  serving (foundation)  : {100*ff/nfr:5.1f}%   [{ff}/{nfr}]   mean ML score {fmlf:.3f}")
print(f"  → skew fix recovers {100*ff/nfr - 100*fb/nfr:+.1f} pts of fraud catch")
print(f"\nLEGIT FALSE-ALERT RATE:")
print(f"  BROKEN : {100*lb/nlg:4.1f}%     FIXED : {100*lf/nlg:4.1f}%")
print("=" * 70)
