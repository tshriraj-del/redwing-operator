"""
build_goldset.py — assemble a BLIND, stratified gold set for human re-adjudication.

Why: the agent-eval verifiers are anchored to the synthetic ground-truth label and a
derived gold disposition. To claim the verifiers are *trustworthy* we must show a human
analyst, deciding independently from the same evidence, agrees with that answer key.

This builder:
  • deterministically samples cases stratified across the three gold-disposition
    branches (confirm_fraud / clear_false_positive / deny_dispute_first_party),
  • writes a CLEAN rater file with the ground-truth label and gold disposition REMOVED
    (so labeling is genuinely blind),
  • writes a separate sealed truth file the scorer reads after labeling.

Run:  python3 eval/build_goldset.py [--n 45]
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, argparse, random
sys.path.insert(0, os.path.expanduser("~/redwing-operator"))
import main, fraud_env

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260629

# Fields that would leak the answer — stripped from the rater's view.
LEAK_KEYS = {"recommended_disposition", "disposition_options", "sar_eligible",
             "sar_note", "_enrichment_note"}
LEAK_ALERT_KEYS = {"ground_truth_label"}


def rater_view(case: dict) -> dict:
    """The case as the human sees it — full evidence, zero answer leakage."""
    c = {k: v for k, v in case.items() if k not in LEAK_KEYS}
    a = dict(c.get("alert", {}))
    for k in LEAK_ALERT_KEYS:
        a.pop(k, None)
    c["alert"] = a
    return c


def main_build(n: int):
    random.seed(SEED)
    df = main.df_all
    # Draw a candidate pool (mix fraud/legit), assemble, bucket by gold disposition.
    fraud_idx = random.sample(list(df[df["is_fraud"] == 1].index),
                              min(220, int((df["is_fraud"] == 1).sum())))
    legit_idx = random.sample(list(df[df["is_fraud"] == 0].index), 120)
    pool = fraud_idx + legit_idx
    random.shuffle(pool)

    per_class = max(1, n // 3)
    buckets = {"confirm_fraud": [], "clear_false_positive": [], "deny_dispute_first_party": []}
    truth = {}
    seen = 0
    for i in pool:
        if all(len(v) >= per_class for v in buckets.values()):
            break
        try:
            case = main._assemble_case(df.loc[i].to_dict())
        except Exception:
            continue
        seen += 1
        gold = fraud_env.gold_disposition(case)
        if gold not in buckets or len(buckets[gold]) >= per_class:
            continue
        cid = case.get("case_id")
        buckets[gold].append(rater_view(case))
        # reference agent disposition (what the verifier-rewarded policy would do)
        ep = fraud_env.run_episode(case, agent="investigator")
        truth[cid] = {
            "case_id": cid,
            "transaction_id": case.get("transaction_id"),
            "ground_truth_is_fraud": fraud_env._is_fraud(case),
            "gold_disposition": gold,
            "agent_disposition": ep["scorecard"]["terminal_action"],
            "agent_correct": ep["scorecard"]["correct"],
            "typology": case.get("alert", {}).get("fraud_typology"),
        }

    cases = [c for b in buckets.values() for c in b]
    random.Random(SEED).shuffle(cases)

    cases_path = os.path.join(HERE, "goldset_cases.jsonl")
    truth_path = os.path.join(HERE, "goldset_truth.jsonl")
    with open(cases_path, "w") as f:
        for c in cases:
            f.write(json.dumps(c, default=str) + "\n")
    with open(truth_path, "w") as f:
        for c in cases:
            f.write(json.dumps(truth[c["case_id"]], default=str) + "\n")

    print(f"Assembled {seen} candidates → gold set of {len(cases)} blind cases.")
    print(f"  bucket sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    print(f"  rater file  : {cases_path}  (ground truth + gold disposition REMOVED)")
    print(f"  sealed truth: {truth_path}  (read only by the scorer, after labeling)")
    print(f"\nNext: python3 eval/label_goldset.py --rater <your_initials>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    a = ap.parse_args()
    main_build(a.n)
