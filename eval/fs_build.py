"""
fs_build.py — build a BLIND, labeled eval set for the FraudSense LLM copilot.

FraudSense is a single-shot LLM investigation (claude-sonnet-4-6). To measure its
quality we need cases with known answers. The operator already produces fully-labeled
case files, so we:
  • sample a balanced fraud/legit set,
  • serialize each into the SAME analyst brief FraudSense reads (mirrors caseToText),
    but STRIP the on-file ground-truth label and the pre-triage recommendation, so we
    test the model's INDEPENDENT judgment (no answer leakage),
  • seal the ground truth (is_fraud, typology, gold disposition) for the scorer.

Run:  python3 eval/fs_build.py [--n 30]
Output: eval/fs_cases.jsonl (prompts) + eval/fs_truth.jsonl (sealed)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, argparse, random
sys.path.insert(0, os.path.expanduser("~/redwing-operator"))
import main, fraud_env

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260629


def case_to_text(c: dict) -> str:
    """Python mirror of fraudsense/api.js caseToText, with answer leakage removed:
    the 'Ground-truth label on file' line and the pre-triage recommendation are dropped
    so FraudSense must reason independently."""
    cust = c.get("customer", {}) or {}; tx = c.get("transaction", {}) or {}
    ins = c.get("instrument", {}) or {}; d = c.get("dispute", {}) or {}
    dn = c.get("device_network", {}) or {}; a = c.get("alert", {}) or {}
    b = cust.get("baseline", {}) or {}
    L = []
    L.append(f'CASE {c.get("case_id")} - {c.get("queue")} queue, priority {c.get("priority")}, status "{c.get("status")}".')
    L.append("")
    L.append(f'ALERT: {a.get("trigger_label","ML risk")} - model score {a.get("model_score")}, '
             f'combined {a.get("combined_score")}, verdict {a.get("verdict")}, typology "{a.get("fraud_typology")}".')
    sig = a.get("matched_signals") or []
    if sig:
        L.append("Matched rule signals: " + ", ".join(s.get("label", str(s)) if isinstance(s, dict) else str(s) for s in sig) + ".")
    L.append("")
    L.append(f'CUSTOMER (CDD): {cust.get("name_masked")}; {cust.get("tenure_band")}, account age '
             f'{cust.get("account_age_days")}d; KYC {cust.get("kyc_status")}; ID {cust.get("id_type")}, '
             f'nationality {cust.get("nationality")}, location {cust.get("location")}; '
             f'PEP {"YES" if cust.get("pep") else "no"}, sanctions {"POTENTIAL MATCH" if cust.get("sanctions_match") else "clear"}; '
             f'customer risk rating {cust.get("risk_rating")} (drivers: {"; ".join(cust.get("risk_drivers", []) or [])}); '
             f'prior cases {cust.get("prior_cases")}, prior SARs {cust.get("prior_sars")}, prior disputes {cust.get("prior_disputes")}.')
    if b:
        L.append(f'Behavioural baseline: avg txn ${b.get("avg_txn")}, typical max ${b.get("typical_max")}, '
                 f'{b.get("known_devices")} known device(s), {b.get("known_recipients")} known payee(s).')
    L.append("")
    L.append(f'TRANSACTION: ${tx.get("amount")} {tx.get("currency","")} on {tx.get("rail")}; '
             f'merchant category "{tx.get("merchant_category")}", MCC {tx.get("mcc_code")} ({tx.get("mcc_label")}); '
             f'new recipient: {"yes" if tx.get("is_new_recipient") else "no"}.')
    if ins.get("type") == "card":
        L.append(f'CARD: {ins.get("network")} {ins.get("funding")}, {ins.get("presence")}, entry mode {ins.get("entry_mode")}; '
                 f'AVS={ins.get("avs_result")}, CVV={ins.get("cvv_result")}, 3-D Secure={ins.get("three_ds_result")}; '
                 f'cross-border {("YES (POS "+str(ins.get("pos_country"))+")") if ins.get("cross_border") else "no"}.')
    elif ins:
        L.append(f'INSTRUMENT: non-card ({ins.get("rail")}); instant rail {"yes" if ins.get("instant") else "no"}, '
                 f'irrevocable {"yes" if ins.get("irrevocable") else "no"}.')
    cfs = c.get("card_fraud_signals") or []
    if cfs:
        L.append(""); L.append("CARD-USAGE FRAUD SIGNALS DETECTED:")
        for s in cfs:
            L.append(f'  - [{s.get("severity")}] {s.get("label")}: {s.get("detail")}')
    if d.get("active"):
        ev = ", ".join(f'{k}={("n/a" if v is None else v)}' for k, v in (d.get("evidence", {}) or {}).items())
        L.append("")
        L.append(f'DISPUTE: active, reason code {d.get("reason_code")} ({d.get("reason")}). Evidence matrix: {ev}. '
                 f'First-party (friendly) fraud risk: {round((d.get("first_party_fraud_risk") or 0)*100)}%. '
                 f'System assessment: {d.get("assessment")}')
    else:
        L.append(f'DISPUTE: none active; {d.get("history_count", 0)} prior on file.')
    L.append("")
    L.append(f'DEVICE/NETWORK: device {dn.get("device_id")} (known device: {"yes" if dn.get("is_known_device") else "NO"}); '
             f'graph risk {dn.get("graph_risk_score")}; linked accounts {dn.get("linked_accounts")}; '
             f'fraud-ring flag: {"FLAGGED" if dn.get("ring_flag") else "clear"}.')
    tl = c.get("timeline") or []
    if tl:
        L.append(""); L.append("ACTIVITY TIMELINE (oldest -> newest):")
        for e in tl:
            L.append(f'  - {e.get("ts")} [{e.get("type")}] {e.get("detail")}')
    return "\n".join(L)


def main_build(n: int):
    random.seed(SEED)
    df = main.df_all
    fr = random.sample(list(df[df["is_fraud"] == 1].index), n)
    lg = random.sample(list(df[df["is_fraud"] == 0].index), n)
    idx = fr + lg
    random.Random(SEED).shuffle(idx)
    cases, truth = [], []
    for i in idx:
        try:
            c = main._assemble_case(df.loc[i].to_dict())
        except Exception:
            continue
        a = c.get("alert", {})
        case_type = a.get("fraud_typology") if a.get("fraud_typology") and a.get("fraud_typology") != "none" else "Payments fraud"
        cases.append({"case_id": c.get("case_id"), "case_type": case_type,
                      "case_text": case_to_text(c)})
        truth.append({"case_id": c.get("case_id"),
                      "is_fraud": fraud_env._is_fraud(c),
                      "typology": a.get("fraud_typology"),
                      "gold_disposition": fraud_env.gold_disposition(c),
                      "case_text": cases[-1]["case_text"]})  # kept for grounding check
    with open(os.path.join(HERE, "fs_cases.jsonl"), "w") as f:
        for x in cases:
            f.write(json.dumps(x, default=str) + "\n")
    with open(os.path.join(HERE, "fs_truth.jsonl"), "w") as f:
        for x in truth:
            f.write(json.dumps(x, default=str) + "\n")
    nf = sum(1 for t in truth if t["is_fraud"])
    print(f"Built {len(cases)} blind FraudSense eval cases ({nf} fraud / {len(cases)-nf} legit).")
    print(f"  prompts: eval/fs_cases.jsonl   sealed truth: eval/fs_truth.jsonl")
    print(f"  answer leakage removed (on-file label + pre-triage recommendation stripped).")
    print(f"\nNext: export ANTHROPIC_API_KEY=sk-...  &&  python3 eval/fs_run.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=15)
    main_build(ap.parse_args().n)
