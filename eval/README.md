# RedWing — Quality-Measurement Stack

Evals for the RedWing risk platform. Everything here drives the **live** operator
(same model + feature foundation the API serves), so every number reconciles with what
production would score. Deterministic (`seed = 20260629`) and reproducible.

## What's here

| Script | Measures |
|---|---|
| `version_delta.py` | **Training-serving skew** — same model, broken vs. fixed feature path. Measured field catch-rate **2% → 80%** (10 of 23 features silently zeroed at serving), false-alert steady ~1%. |
| `build_goldset.py` → `label_goldset.py` → `score_agreement.py` | **Human ↔ verifier agreement.** Builds a blind, stratified gold set; a human re-adjudicates it; scores agreement + Cohen's kappa against the verifier's answer key. |
| `fs_build.py` → `fs_run.py` → `fs_score.py` | **FraudSense LLM eval.** Blind labeled cases → run the real copilot (`claude-sonnet-4-6`) → score schema validity, risk calibration, and **evidence-grounding / hallucinated-observation rate**. |

The adversary simulator (`adversary.py`) and the process+outcome verifiers (`fraud_env.py`)
live in the repo root and are driven directly; see the parent README.

## Running

```bash
# Measured skew delta (no API key needed)
python3 eval/version_delta.py

# Human-agreement gold set
python3 eval/build_goldset.py --n 45
python3 eval/label_goldset.py --rater <initials>     # ~25 min, blind
python3 eval/score_agreement.py

# FraudSense LLM eval (needs the same key the app uses)
python3 eval/fs_build.py --n 15
export ANTHROPIC_API_KEY=sk-...
python3 eval/fs_run.py && python3 eval/fs_score.py
```

Generated `*.jsonl` sets are gitignored (regenerate via the build scripts).

## Honesty guardrails

- Results are on an **880K-transaction synthetic benchmark**, except the real-data payment
  model (PR-AUC 0.90, validated on real ULB labels).
- **No human-labeled set exists until you run `label_goldset.py`** — do not claim
  "verifier agrees with human X%" before that.
- FraudSense is a single-shot LLM call with **no retrieval** — there is no RAG
  precision@k or retrieval-recall metric to report.
