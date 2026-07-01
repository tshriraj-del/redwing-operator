"""
score_agreement.py — turn the blind human labels into the headline eval-quality metric:
does an independent human analyst agree with the verifiers' answer key?

Reports, over the labeled gold set:
  • Human vs GOLD disposition  — agreement % + Cohen's kappa  (is the answer key trustworthy?)
  • Human vs AGENT disposition  — agreement %                  (does the rewarded policy match a human?)
  • Human vs ground-truth fraud — agreement %                  (does the human even see it as fraud?)
  • Inter-rater (if ≥2 raters) — agreement % + Cohen's kappa   (is human judgment itself reliable?)
  • Confusion + per-class breakdown + time-on-task.

Run:  python3 eval/score_agreement.py
"""
import os, json, glob
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

DECISIVE = {"confirm_fraud", "clear_false_positive", "deny_dispute_first_party"}
FRAUDISH = {"confirm_fraud", "deny_dispute_first_party"}   # disposition implies fraud finding


def cohen_kappa(a, b):
    """Cohen's kappa for two aligned label lists (no sklearn dependency)."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_truth():
    p = os.path.join(HERE, "goldset_truth.jsonl")
    return {r["case_id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


def load_labels():
    raters = {}
    for path in sorted(glob.glob(os.path.join(HERE, "goldset_labels__*.jsonl"))):
        rid = os.path.basename(path).split("__")[1].rsplit(".", 1)[0]
        raters[rid] = {r["case_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}
    return raters


def pct(x):
    return f"{100*x:.1f}%"


def report_rater(rid, labels, truth):
    ids = [cid for cid in labels if cid in truth]
    hum = [labels[cid]["disposition"] for cid in ids]
    gold = [truth[cid]["gold_disposition"] for cid in ids]
    agent = [truth[cid]["agent_disposition"] for cid in ids]
    gt_fraud = [bool(truth[cid]["ground_truth_is_fraud"]) for cid in ids]

    # decisive subset (exclude human "escalate_or_hold" which maps to no gold class)
    dec = [(h, g, ag) for h, g, ag in zip(hum, gold, agent) if h in DECISIVE]
    h_d = [h for h, _, _ in dec]; g_d = [g for _, g, _ in dec]; a_d = [ag for _, _, ag in dec]

    n = len(ids); nd = len(dec); n_hold = n - nd
    agree_gold = sum(1 for h, g in zip(h_d, g_d) if h == g) / nd if nd else 0
    kappa_gold = cohen_kappa(h_d, g_d) if nd else float("nan")
    agree_agent = sum(1 for h, a in zip(h_d, a_d) if h == a) / nd if nd else 0
    # binary fraud agreement (use full set; hold = "not decisive" → excluded)
    bin_pairs = [(h in FRAUDISH, gt) for h, gt in zip(hum, gt_fraud) if h in DECISIVE]
    agree_fraud = sum(1 for hh, gt in bin_pairs if hh == gt) / len(bin_pairs) if bin_pairs else 0

    secs = [labels[cid].get("seconds", 0) for cid in ids]
    print(f"\n── rater '{rid}'  ({n} cases labeled, {n_hold} marked escalate/hold) ──")
    print(f"  Human vs GOLD disposition .... {pct(agree_gold)}  (kappa {kappa_gold:.2f})  on {nd} decisive cases")
    print(f"  Human vs AGENT disposition ... {pct(agree_agent)}")
    print(f"  Human vs ground-truth fraud .. {pct(agree_fraud)}  (binary fraud / not-fraud)")
    if secs:
        print(f"  Median time-on-task .......... {sorted(secs)[len(secs)//2]:.0f}s/case")
    # disagreement detail
    disagree = [(cid, labels[cid]["disposition"], truth[cid]["gold_disposition"])
                for cid in ids if labels[cid]["disposition"] in DECISIVE
                and labels[cid]["disposition"] != truth[cid]["gold_disposition"]]
    if disagree:
        print(f"  Disagreements ({len(disagree)}):")
        for cid, h, g in disagree[:12]:
            note = labels[cid].get("note", "")
            print(f"    {cid}: human={h}  gold={g}" + (f"  · “{note}”" if note else ""))
    return {"rater": rid, "n": n, "decisive": nd, "agree_gold": agree_gold,
            "kappa_gold": kappa_gold, "agree_agent": agree_agent, "agree_fraud": agree_fraud}


def main():
    truth = load_truth()
    raters = load_labels()
    if not raters:
        raise SystemExit("No goldset_labels__*.jsonl yet — run: python3 eval/label_goldset.py --rater <id>")
    print("=" * 74)
    print("HUMAN ↔ VERIFIER AGREEMENT  (blind re-adjudication of the gold set)")
    print("=" * 74)
    print(f"Gold set: {len(truth)} cases.  Raters: {', '.join(raters)}.")
    summaries = [report_rater(rid, labels, truth) for rid, labels in raters.items()]

    # inter-rater (first two raters with overlap)
    rids = list(raters)
    if len(rids) >= 2:
        a, b = rids[0], rids[1]
        common = [cid for cid in raters[a] if cid in raters[b]]
        if common:
            la = [raters[a][cid]["disposition"] for cid in common]
            lb = [raters[b][cid]["disposition"] for cid in common]
            ag = sum(1 for x, y in zip(la, lb) if x == y) / len(common)
            print(f"\n── inter-rater '{a}' vs '{b}'  ({len(common)} shared cases) ──")
            print(f"  Agreement {pct(ag)}   ·   Cohen's kappa {cohen_kappa(la, lb):.2f}")

    print("\n" + "=" * 74)
    print("RESUME-READY (fill once labeling is done):")
    for s in summaries:
        print(f"  • Across {s['decisive']} blind-adjudicated cases, an independent human analyst "
              f"agreed with the verifier's answer key on {pct(s['agree_gold'])} "
              f"(Cohen's kappa {s['kappa_gold']:.2f}).")
    print("=" * 74)


if __name__ == "__main__":
    main()
