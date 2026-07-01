"""
fs_score.py — score FraudSense outputs into defensible generative-AI quality metrics.

Headline metrics (the ones not given away by the alert context):
  • Schema reliability  — % responses that parse and satisfy every enum/shape rule.
  • Evidence grounding  — of every signal the model tagged "Observed", the fraction whose
                          cited specifics (numbers + salient terms) actually appear in the
                          case input. 1 - this = hallucinated-observation rate. This directly
                          tests the product's own #1 rule ("Observed = explicitly stated").
  • Risk calibration    — mean risk score on fraud vs legit (separation), and whether the
                          severity band agrees with ground-truth fraud.
Secondary (alert-contextualized, so near-ceiling — reported with that caveat):
  • Action agreement    — model action vs the ground-truth disposition.

Run:  python3 eval/fs_score.py
"""
import os, json, re
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))

STOP = set("the a an and or of to in on for with is was are be by at from this that these those "
           "it its as not no any all over under into out off then than so such very more most "
           "you your their there here when while because during within between across about".split())
GENERIC = set("fraud fraudulent risk risky suspicious account transaction transactions signal signals "
              "activity activities pattern patterns customer customers high higher unusual unusthough "
              "amount amounts payment payments behavior behaviour evidence indicates indicating likely "
              "suggests consistent case alert score device recipient".split())
ENUMS = {
    "strength": {"Weak", "Moderate", "Strong"},
    "category": {"Identity", "Device", "Behavioral", "Payment", "Network", "Velocity"},
    "basis": {"Observed", "Inferred"},
    "confidence": {"Low", "Medium", "High"},
    "action": {"Approve", "Decline", "Escalate", "Monitor"},
    "severity": {"Low", "Medium", "High", "Critical"},
}


def norm(s): return re.sub(r"\s+", " ", str(s).lower())
def nums(s): return set(re.sub(r"[,$]", "", x) for x in re.findall(r"\d[\d,\.]*", str(s)))
def salient(s):
    toks = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", str(s).lower())
    return [t for t in toks if t not in STOP and t not in GENERIC]


def signal_grounded(sig, input_norm, input_nums):
    claim = f"{sig.get('name','')} {sig.get('reason','')}"
    for nv in nums(claim):
        if nv not in input_nums and nv.rstrip(".0") not in {n.rstrip('.0') for n in input_nums}:
            return False  # cited a number not in the case material
    sal = salient(claim)
    if not sal:
        return True
    hit = sum(1 for w in sal if w in input_norm)
    return (hit / len(sal)) >= 0.5


def enum_ok(analysis):
    try:
        rs = analysis["risk_score"]
        assert isinstance(rs["score"], int) and 0 <= rs["score"] <= 100
        assert rs["severity"] in ENUMS["severity"]
        for sig in analysis["signals"]:
            assert sig["strength"] in ENUMS["strength"]
            assert sig["category"] in ENUMS["category"]
            assert sig["basis"] in ENUMS["basis"]
        assert analysis["classification"]["confidence"] in ENUMS["confidence"]
        assert analysis["recommendation"]["action"] in ENUMS["action"]
        assert len(analysis["recommendation"]["next_steps"]) == 3
        return True
    except Exception:
        return False


def main():
    truth = {t["case_id"]: t for t in (json.loads(l) for l in open(os.path.join(HERE, "fs_truth.jsonl")) if l.strip())}
    outs = [json.loads(l) for l in open(os.path.join(HERE, "fs_outputs.jsonl")) if l.strip()] \
        if os.path.exists(os.path.join(HERE, "fs_outputs.jsonl")) else []
    if not outs:
        raise SystemExit("No fs_outputs.jsonl — run: export ANTHROPIC_API_KEY=... && python3 eval/fs_run.py")

    n = len(outs)
    parsed = [o for o in outs if o.get("parsed_ok")]
    enum_valid = [o for o in parsed if enum_ok(o["analysis"])]

    obs_total = obs_grounded = 0
    flagged = []
    fraud_scores, legit_scores = [], []
    sev_correct = act_correct = act_total = 0
    typ_match = 0

    DECLINE_SET = {"Decline", "Escalate"}
    for o in enum_valid:
        a = o["analysis"]; t = truth.get(o["case_id"], {})
        inp = norm(t.get("case_text", "")); inp_nums = nums(t.get("case_text", ""))
        for sig in a["signals"]:
            if sig["basis"] == "Observed":
                obs_total += 1
                if signal_grounded(sig, inp, inp_nums):
                    obs_grounded += 1
                elif len(flagged) < 15:
                    flagged.append((o["case_id"], sig.get("name"), sig.get("reason")))
        sc = a["risk_score"]["score"]
        (fraud_scores if t.get("is_fraud") else legit_scores).append(sc)
        sev_high = a["risk_score"]["severity"] in {"High", "Critical"}
        if sev_high == bool(t.get("is_fraud")):
            sev_correct += 1
        # action vs ground-truth disposition (alert-contextualized — see caveat)
        gold = t.get("gold_disposition")
        want_decline = gold in {"confirm_fraud", "deny_dispute_first_party"}
        act = a["recommendation"]["action"]
        act_total += 1
        if (act in DECLINE_SET) == want_decline:
            act_correct += 1
        # classification consistency (case_type is provided, so this is a sanity check)
        if t.get("typology") and t["typology"] != "none":
            if any(w in norm(a["classification"]["primary_type"]) for w in str(t["typology"]).split("_")):
                typ_match += 1

    def p(x, d): return f"{100*x/d:.1f}%" if d else "n/a"
    print("=" * 72)
    print("FRAUDSENSE LLM EVAL  (model claude-sonnet-4-6, blind briefs)")
    print("=" * 72)
    print(f"Cases run: {n}   ·   parsed clean: {len(parsed)} ({p(len(parsed),n)})   ·   "
          f"schema+enum valid: {len(enum_valid)} ({p(len(enum_valid),n)})")
    print("\nHEADLINE — generative quality:")
    print(f"  Evidence grounding (Observed signals supported by input): "
          f"{p(obs_grounded,obs_total)}  [{obs_grounded}/{obs_total}]")
    print(f"    → hallucinated-observation rate: {p(obs_total-obs_grounded,obs_total)}")
    print(f"  Schema reliability (parse + enums + shape): {p(len(enum_valid),n)}")
    if fraud_scores and legit_scores:
        print(f"\nRisk calibration:")
        print(f"  mean risk score  fraud {mean(fraud_scores):.0f}/100   vs   legit {mean(legit_scores):.0f}/100   "
              f"(separation {mean(fraud_scores)-mean(legit_scores):.0f} pts)")
        print(f"  severity band agrees with ground-truth fraud: {p(sev_correct,len(enum_valid))}")
    print(f"\nSecondary (alert-contextualized → expect near-ceiling; reported for completeness):")
    print(f"  Action agreement with ground-truth disposition: {p(act_correct,act_total)}")
    print(f"  Classification consistency with provided typology: {p(typ_match,act_total)}")
    if flagged:
        print(f"\nFlagged 'Observed' signals to eyeball (grounding check is conservative/lexical):")
        for cid, name, reason in flagged[:10]:
            print(f"  · {cid}: \"{name}\" — {str(reason)[:90]}")
    print("\n" + "=" * 72)
    print("RESUME-READY:")
    print(f"  • Built an automated eval for the FraudSense LLM copilot over {n} labeled cases: "
          f"{p(len(enum_valid),n)} schema-valid, evidence-grounding {p(obs_grounded,obs_total)} "
          f"(hallucinated-observation rate {p(obs_total-obs_grounded,obs_total)}), risk-score "
          f"separation {mean(fraud_scores)-mean(legit_scores):.0f} pts fraud-vs-legit." if fraud_scores else "")
    print("=" * 72)


if __name__ == "__main__":
    main()
