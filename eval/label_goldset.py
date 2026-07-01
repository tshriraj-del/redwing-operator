"""
label_goldset.py — interactive BLIND labeler for the gold set.

Shows each case the way an analyst would see it (alert + full evidence, NO answer),
asks for a disposition, optional confidence, optional note. Resumable: re-running skips
cases this rater already labeled. Writes one JSONL per rater, so two raters → kappa.

Run:  python3 eval/label_goldset.py --rater st
      python3 eval/label_goldset.py --rater jd     # a second, independent rater
"""
import os, json, argparse, time, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

CHOICES = {
    "1": ("confirm_fraud",            "Fraud — confirm / block the instrument"),
    "2": ("clear_false_positive",     "Legitimate — clear the alert, no action"),
    "3": ("deny_dispute_first_party", "First-party fraud — deny the customer's dispute"),
    "4": ("escalate_or_hold",         "Not confident — escalate / step-up / hold for review"),
}
CONF = {"1": "low", "2": "medium", "3": "high"}


def load_cases():
    p = os.path.join(HERE, "goldset_cases.jsonl")
    if not os.path.exists(p):
        raise SystemExit("No goldset_cases.jsonl — run: python3 eval/build_goldset.py")
    return [json.loads(l) for l in open(p) if l.strip()]


def done_ids(rater):
    p = os.path.join(HERE, f"goldset_labels__{rater}.jsonl")
    if not os.path.exists(p):
        return set()
    return {json.loads(l)["case_id"] for l in open(p) if l.strip()}


def kv(label, val, w=26):
    return f"  {label:<{w}} {val}"


def render(case, n, total):
    a = case.get("alert", {}); cust = case.get("customer", {}); b = cust.get("baseline", {})
    txn = case.get("transaction", {}); disp = case.get("dispute", {}) or {}
    dev = case.get("device_network", {}) or {}; cfs = case.get("card_fraud_signals", []) or []
    line = "─" * 74
    print("\n" + "=" * 74)
    print(f"CASE {n}/{total}   ·   {case.get('case_id')}   ·   priority {case.get('priority')}")
    print("=" * 74)
    print("ALERT")
    print(kv("typology", a.get("fraud_typology")))
    print(kv("model / combined score", f"{a.get('model_score')}  /  {a.get('combined_score')}  ({a.get('verdict')})"))
    sig = a.get("matched_signals", [])
    if sig:
        print(kv("matched signals", ", ".join(s.get("label", "?") for s in sig[:6])))
    print(line)
    print("TRANSACTION")
    print(kv("amount / rail", f"{txn.get('amount')}  ·  {txn.get('rail', txn.get('payment_rail'))}"))
    for k in ("recipient_id", "device_id", "hour", "merchant", "channel"):
        if txn.get(k) is not None:
            print(kv(k, txn.get(k)))
    print(line)
    print("CUSTOMER 360")
    print(kv("tenure / KYC", f"{cust.get('tenure_band')}  ·  KYC {cust.get('kyc_status')}  ·  CIP {cust.get('cip_verified')}"))
    print(kv("risk rating", f"{cust.get('risk_rating')}   drivers: {', '.join(cust.get('risk_drivers', []) or []) or '—'}"))
    print(kv("PEP / sanctions", f"PEP {cust.get('pep')}  ·  sanctions match {cust.get('sanctions_match')}"))
    print(kv("prior cases/SARs/disp", f"{cust.get('prior_cases')} / {cust.get('prior_sars')} / {cust.get('prior_disputes')}"))
    if b:
        print(kv("baseline", f"avg {b.get('avg_txn')} · typ-max {b.get('typical_max')} · "
                             f"{b.get('known_devices')} devices · {b.get('known_recipients')} recipients"))
    if cfs:
        print(line); print("CARD-FRAUD SIGNALS")
        for s in cfs[:6]:
            print(kv(s.get("severity", "?").upper(), s.get("label") or s.get("name") or s))
    if disp:
        print(line); print("DISPUTE")
        print(kv("active / reason", f"{disp.get('active')}  ·  {disp.get('reason_code')} {disp.get('reason')}"))
        ev = disp.get("evidence", {}) or {}
        if ev:
            print(kv("evidence", ", ".join(f"{k}={v}" for k, v in ev.items() if v is not None) or "—"))
        if disp.get("assessment"):
            for ln in textwrap.wrap(str(disp.get("assessment")), 70):
                print("    " + ln)
    if dev:
        print(line); print("DEVICE / NETWORK")
        print(kv("ring flag / device", f"ring={dev.get('ring_flag')}  ·  device_familiarity={dev.get('device_familiarity')}"))
        for k in ("shared_device_users", "recipient_indegree", "community_fraud_rate"):
            if dev.get(k) is not None:
                print(kv(k, dev.get(k)))
    print(line)


def ask(prompt, valid):
    while True:
        r = input(prompt).strip().lower()
        if r in valid:
            return r
        if r == "q":
            return "q"
        print("  (enter one of: " + "/".join(valid) + ", or q to save & quit)")


def run(rater):
    cases = load_cases()
    done = done_ids(rater)
    todo = [c for c in cases if c["case_id"] not in done]
    out = os.path.join(HERE, f"goldset_labels__{rater}.jsonl")
    print(f"Rater '{rater}': {len(done)} already labeled, {len(todo)} remaining of {len(cases)}.")
    print("Decide each case from the evidence as if it were live. Blind — no answer key shown.")
    print("Menu: " + "  ".join(f"[{k}] {v[1]}" for k, v in CHOICES.items()))
    for n, case in enumerate(todo, len(done) + 1):
        render(case, n, len(cases))
        t0 = time.time()
        d = ask("\n  Disposition [1-4] (q=save&quit): ", set(CHOICES))
        if d == "q":
            break
        c = ask("  Confidence [1 low / 2 med / 3 high]: ", set(CONF))
        if c == "q":
            break
        note = input("  Note (optional, enter to skip): ").strip()
        rec = {
            "case_id": case["case_id"], "rater": rater,
            "disposition": CHOICES[d][0], "confidence": CONF[c],
            "note": note, "seconds": round(time.time() - t0, 1),
        }
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  ✓ saved {CHOICES[d][0]}")
    n_done = len(done_ids(rater))
    print(f"\nSaved {n_done}/{len(cases)} labels → {out}")
    if n_done >= len(cases):
        print("Complete. Next: python3 eval/score_agreement.py")
    else:
        print(f"Resume anytime: python3 eval/label_goldset.py --rater {rater}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", required=True, help="short id, e.g. your initials")
    run(ap.parse_args().rater)
