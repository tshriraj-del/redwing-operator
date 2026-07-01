"""
fs_run.py — run the REAL FraudSense pipeline over the eval set.

Uses the EXACT model (claude-sonnet-4-6), system prompt, schema, and JSON parsing from
fraudsense/api.js, so the metric reconciles with the shipped product. Reads the API key
from ANTHROPIC_API_KEY (or VITE_ANTHROPIC_API_KEY). Resumable: re-running skips cases
already scored. Writes raw model outputs to eval/fs_outputs.jsonl.

Run:  export ANTHROPIC_API_KEY=sk-...   &&   python3 eval/fs_run.py
"""
import os, sys, json, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

# ── verbatim from fraudsense/api.js ───────────────────────────────────────────
SYSTEM_PROMPT = """You are FraudSense, a senior fraud investigation copilot embedded in an analyst's triage console. You reason like a seasoned fraud/risk investigator across payments, account takeover, marketplace abuse, and identity fraud. Your output is auditable evidence used to action real accounts and refer cases - so calibration and intellectual honesty matter more than confident-sounding prose.

You run a rigorous investigation: extract signals, score risk, classify, estimate loss, separate fact from inference, reconstruct root cause, and recommend a data-grounded action.

EVIDENCE DISCIPLINE (most important):
- Distinguish what is OBSERVED (explicitly stated in the case material) from what is INFERRED (your reasoning). Tag every signal accordingly.
- NEVER present an inference as an established fact. Hedge inferences ("likely", "suggests", "consistent with").
- Do NOT assert a specific fraud-ring, "organized ring", "synthetic identity", "account takeover", or "stolen identity" conclusion unless the case contains DIRECT evidence for it. Shared devices, multiple suspended accounts, or multi-country logins indicate repeat/possibly-coordinated activity - they do NOT by themselves prove an organized ring or synthetic identities. Put such unproven theories in "hypotheses", not "classification".
- "secondary_type" must be null unless there is direct supporting evidence for a distinct second fraud type.
- Quantify. Never say "several thousand dollars" - give explicit numbers and ranges with a stated basis. Distinguish reported INCIDENTS from confirmed unique VICTIMS.
- Calibrate escalation. Recommend law-enforcement / cross-border / INTERPOL coordination ONLY when losses are large or jurisdictionally required; otherwise keep to internal + receiving-bank + platform steps.

CRITICAL OUTPUT RULES:
- Respond with ONE raw, valid JSON object and nothing else. No markdown fences, no preamble, no commentary.
- Use ONLY the enum values specified. Match capitalization exactly."""

SCHEMA_BLOCK = """Return a single JSON object matching this EXACT schema - no extra keys, no missing keys:

{
  "risk_score": { "score": 0, "severity": "Low | Medium | High | Critical",
    "factors": [ { "name": "contributing factor", "weight": 0 } ] },
  "signals": [ { "name": "short signal name", "reason": "1-2 sentences on why this is suspicious in THIS case",
      "strength": "Weak | Moderate | Strong",
      "category": "Identity | Device | Behavioral | Payment | Network | Velocity",
      "basis": "Observed | Inferred" } ],
  "classification": { "primary_type": "most likely fraud type", "secondary_type": "second fraud type or null",
    "confidence": "Low | Medium | High", "reasoning": "2-3 sentences explaining the classification" },
  "loss_estimate": { "confirmed": "confirmed loss with figure", "likely_low": "low end", "likely_high": "high end",
    "basis": "how the range was derived" },
  "fact_assessment": { "observed_facts": ["facts explicitly stated"], "assessments": ["inferences held with confidence"],
    "hypotheses": ["unverified theories"] },
  "root_cause_analysis": { "attack_narrative": "what likely happened", "entry_point": "how it started",
    "blast_radius": "what is at risk", "watch_for": "patterns to watch for" },
  "recommendation": { "action": "Approve | Decline | Escalate | Monitor", "confidence": "Low | Medium | High",
    "reasoning": "3-4 sentences", "decision_logic": ["reasoning step citing a specific signal/score/figure"],
    "next_steps": ["step 1", "step 2", "step 3"], "escalation_path": "what to do if the analyst disagrees" }
}

Requirements:
- "strength" in {Weak, Moderate, Strong}; "category" in {Identity, Device, Behavioral, Payment, Network, Velocity};
  "basis" in {Observed, Inferred} (Observed only when the material directly states the fact); "confidence" in {Low, Medium, High};
  "action" in {Approve, Decline, Escalate, Monitor}.
- "risk_score.score" integer 0-100. Bands: 0-39 Low, 40-64 Medium, 65-84 High, 85-100 Critical.
- Use null (not "null") for "secondary_type" when there is no direct evidence for a second type.
- Provide exactly 3 items in "next_steps". Extract every distinct signal the material supports; no filler."""


def build_prompt(case_text, case_type):
    return (f"Investigate the following fraud case.\n\nCase Type (analyst-selected): {case_type}\n"
            f'Case Description:\n"""\n{case_text}\n"""\n\n{SCHEMA_BLOCK}')


def call(api_key, case_text, case_type):
    body = json.dumps({"model": MODEL, "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT,
                       "messages": [{"role": "user", "content": build_prompt(case_text, case_type)}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-api-key": api_key,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["content"][0]["text"].strip(), data.get("usage", {})


def parse_analysis(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned), True
    except Exception:
        s = cleaned.find("{"); e = cleaned.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(cleaned[s:e+1]), True
            except Exception:
                pass
    return {"_raw": text}, False


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VITE_ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY (the same key the FraudSense app uses) and re-run.")
    cases = [json.loads(l) for l in open(os.path.join(HERE, "fs_cases.jsonl")) if l.strip()]
    out_path = os.path.join(HERE, "fs_outputs.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["case_id"] for l in open(out_path) if l.strip()}
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"FraudSense eval: {len(done)} done, {len(todo)} to run (model {MODEL}).")
    ok = parse_ok = 0
    for i, c in enumerate(todo, 1):
        try:
            text, usage = call(api_key, c["case_text"], c["case_type"])
            analysis, parsed = parse_analysis(text)
            rec = {"case_id": c["case_id"], "parsed_ok": parsed, "analysis": analysis,
                   "usage": usage}
            ok += 1; parse_ok += 1 if parsed else 0
        except urllib.error.HTTPError as e:
            rec = {"case_id": c["case_id"], "parsed_ok": False, "error": f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"}
        except Exception as e:
            rec = {"case_id": c["case_id"], "parsed_ok": False, "error": str(e)}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"  [{i}/{len(todo)}] {c['case_id']}: {'ok' if rec.get('parsed_ok') else rec.get('error','parse-fail')}")
        time.sleep(1.2)   # gentle pacing
    print(f"\nDone. {ok} calls, {parse_ok} parsed clean. Next: python3 eval/fs_score.py")


if __name__ == "__main__":
    main()
