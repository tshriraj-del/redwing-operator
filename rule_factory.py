"""
Riposte Rule Factory — Self-improving rule engine.

Identifies fraud vectors that ML caught but rules missed (rule gaps),
sends them to Claude for pattern analysis, generates candidate rules,
backtests them, and implements rules that pass the quality gate.

Pipeline:
  extract_gaps() → analyze_with_llm() → generate_rules() →
  backtest() → quality_gate() → implement() → monitor()
"""

import json
import os
import re
import time
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

MODELS_DIR  = Path.home() / "pulseml_models"
RULES_FILE  = MODELS_DIR / "generated_rules.json"
LOG_FILE    = MODELS_DIR / "rule_factory_log.json"

# Quality thresholds
MIN_PRECISION     = 0.55   # minimum precision for shadow mode
AUTO_DEPLOY_PREC  = 0.78   # precision required for auto-deploy
MIN_RECALL        = 0.003  # must catch at least 0.3% of all fraud
MAX_OVERLAP       = 0.80   # max overlap with existing rules (avoid redundancy)
MIN_GAPS_TO_FIRE  = 5      # minimum rule gaps before triggering analysis

ALL_FEATURES = [
    'amount_zscore', 'amount_vs_max', 'hour_risk', 'rail_risk',
    'recipient_familiarity', 'device_familiarity',
    'velocity_1h', 'velocity_4h', 'velocity_24h', 'velocity_7d', 'velocity_30d',
    'new_recipient_streak', 'is_crypto', 'is_instant_rail', 'is_p2p',
    'is_round_amount', 'amount_just_below_threshold', 'is_new_maximum',
    'account_age_days', 'preferred_rail_deviation', 'merchant_category_shift',
    'recipient_global_fraud_rate', 'inter_tx_time_short',
]

RULE_GENERATION_SYSTEM = """You are Riposte's rule engine architect. Your job is to analyze confirmed fraud transactions that slipped past the existing rule engine and generate new detection rules.

You will receive:
1. Feature statistics for transactions that were confirmed fraud but had rule_score = 0
2. The existing rules that failed to catch them

Generate 2-3 candidate detection rules. Each rule MUST be a Python lambda that:
- Takes a dict `r` with float feature values
- Uses ONLY r.get('feature_name', default) — no other function calls
- Returns True if the transaction matches the fraud pattern

Output a JSON array of rule objects:
[
  {
    "name": "RULE_NAME_SCREAMING_SNAKE_CASE",
    "tier": 1,
    "score": 85,
    "typology": "pig_butchering|app_scam|account_takeover_ai|deepfake_social_engineering|synthetic_id_ai|card_testing_bot|cross",
    "reason": "1-2 sentences: what fraud behavior this detects and why these thresholds",
    "features_used": ["feature1", "feature2"],
    "fn_code": "lambda r: r.get('feature1', 0) > 0.5 and r.get('feature2', 1) < 0.2"
  }
]

Rules:
- tier 1 = instant block (score 88-100): very high confidence, low FP risk
- tier 2 = high risk (score 70-85): strong signal combination
- tier 3 = elevated (score 50-65): single or weak signals
- fn_code must be valid Python — use only r.get(), no imports, no side effects
- Thresholds must be grounded in the statistics you see, not arbitrary
- Output ONLY the JSON array, no other text"""


def load_transactions() -> Optional[pd.DataFrame]:
    """Load the full transaction dataset with scores."""
    path = MODELS_DIR / "transactions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ['timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def extract_rule_gaps(df: pd.DataFrame, min_gaps: int = MIN_GAPS_TO_FIRE) -> pd.DataFrame:
    """
    Find confirmed fraud transactions where ML fired but rules didn't.
    These are the training signal for new rule generation.
    """
    required = {'is_fraud', 'ensemble_score', 'rule_score'}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    gaps = df[
        (df['is_fraud'] == True) &
        (df['ensemble_score'] > 0.70) &   # ML was confident
        (df['rule_score'] < 30)            # rules completely missed it
    ].copy()

    return gaps if len(gaps) >= min_gaps else pd.DataFrame()


def format_gap_statistics(gaps: pd.DataFrame) -> str:
    """Format gap transaction statistics for the LLM prompt."""
    available = [f for f in ALL_FEATURES if f in gaps.columns]
    if not available:
        return "No feature data available."

    stats = gaps[available].describe().round(4)
    lines = [f"Gap transactions (n={len(gaps)}) — confirmed fraud, rule_score=0:\n"]

    for feat in available:
        if feat not in stats.columns:
            continue
        s = stats[feat]
        lines.append(
            f"  {feat:<35} mean={s['mean']:.3f}  std={s['std']:.3f}  "
            f"min={s['min']:.3f}  median={s['50%']:.3f}  max={s['max']:.3f}"
        )

    # Typology breakdown if available
    if 'fraud_typology' in gaps.columns:
        lines.append("\nTypology breakdown:")
        for typ, cnt in gaps['fraud_typology'].value_counts().items():
            lines.append(f"  {typ}: {cnt} ({cnt/len(gaps):.0%})")

    return "\n".join(lines)


def format_existing_rules(existing_rules: list) -> str:
    """Summarize current rules so LLM avoids duplication."""
    if not existing_rules:
        return "No existing rules."
    lines = []
    for r in existing_rules:
        lines.append(f"  [{r.get('tier','?')}] {r.get('name','?')}: {r.get('reason','')[:80]}")
    return "\n".join(lines)


def analyze_and_generate(gaps: pd.DataFrame, existing_rules: list, api_key: str) -> list[dict]:
    """Send gap statistics to Claude and get candidate rules back."""
    import urllib.request

    gap_stats    = format_gap_statistics(gaps)
    rule_summary = format_existing_rules(existing_rules)

    user_message = f"""Analyze these rule gaps and generate detection rules.

RULE GAPS (fraud ML caught, but 41 existing rules missed):
{gap_stats}

EXISTING RULES (do not duplicate):
{rule_summary}

Available features for rule conditions:
{', '.join(ALL_FEATURES)}

Generate 2-3 new rules that close these gaps."""

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
        "system": RULE_GENERATION_SYSTEM,
        "messages": [{"role": "user", "content": user_message}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    raw = data["content"][0]["text"].strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE).strip()
    candidates = json.loads(cleaned)
    return candidates if isinstance(candidates, list) else []


def _safe_lambda(fn_code: str):
    """
    Compile a lambda string into a callable.
    Only allows r.get() — no imports, no arbitrary calls.
    Raises ValueError if the code is unsafe.
    """
    # Security: only allow safe patterns
    forbidden = ['import', '__', 'exec', 'eval', 'open', 'os.', 'sys.', 'subprocess']
    for f in forbidden:
        if f in fn_code:
            raise ValueError(f"Unsafe pattern in rule code: {f}")
    if not fn_code.strip().startswith('lambda r:'):
        raise ValueError("fn_code must start with 'lambda r:'")

    fn = eval(fn_code, {"__builtins__": {}})  # restricted namespace
    if not callable(fn):
        raise ValueError("fn_code did not evaluate to a callable")
    return fn


def backtest_rule(candidate: dict, df: pd.DataFrame, existing_rules: list) -> dict:
    """
    Run a candidate rule against the full dataset.
    Returns precision, recall, F1, overlap with existing, and recommendation.
    """
    fn_code = candidate.get('fn_code', '')
    try:
        fn = _safe_lambda(fn_code)
    except Exception as e:
        return {"error": str(e), "recommendation": "REJECT"}

    available = [f for f in ALL_FEATURES if f in df.columns]
    triggered = []
    for row in df[available].fillna(0).to_dict('records'):
        try:
            triggered.append(bool(fn(row)))
        except Exception:
            triggered.append(False)

    triggered_arr = np.array(triggered)
    y_true = df['is_fraud'].astype(bool).values

    TP = (triggered_arr & y_true).sum()
    FP = (triggered_arr & ~y_true).sum()
    FN = (~triggered_arr & y_true).sum()
    N_flagged = triggered_arr.sum()
    N_fraud   = y_true.sum()

    precision = TP / N_flagged if N_flagged > 0 else 0.0
    recall    = TP / N_fraud   if N_fraud   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Overlap with existing rules
    existing_triggered = df.get('rule_triggered', pd.Series([False]*len(df))).astype(bool).values
    overlap = (triggered_arr & existing_triggered).sum() / max(N_flagged, 1)

    # Recommendation
    if precision < MIN_PRECISION or recall < MIN_RECALL:
        rec = "REJECT"
    elif overlap > MAX_OVERLAP:
        rec = "REJECT_DUPLICATE"
    elif precision >= AUTO_DEPLOY_PREC and recall >= MIN_RECALL:
        rec = "AUTO_DEPLOY"
    else:
        rec = "SHADOW"

    return {
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
        "n_flagged": int(N_flagged),
        "TP": int(TP), "FP": int(FP), "FN": int(FN),
        "overlap_with_existing": round(float(overlap), 4),
        "recommendation": rec,
    }


def save_generated_rule(candidate: dict, backtest: dict, status: str = "shadow"):
    """Persist a generated rule to the generated_rules.json store."""
    rules = []
    if RULES_FILE.exists():
        try:
            rules = json.loads(RULES_FILE.read_text())
        except Exception:
            rules = []

    rule_id = hashlib.md5(candidate['name'].encode()).hexdigest()[:8]
    entry = {
        "id":         rule_id,
        "name":       candidate.get("name"),
        "tier":       candidate.get("tier", 2),
        "score":      candidate.get("score", 70),
        "typology":   candidate.get("typology", "cross"),
        "reason":     candidate.get("reason", ""),
        "features":   candidate.get("features_used", []),
        "fn_code":    candidate.get("fn_code", ""),
        "status":     status,       # shadow | deployed | retired
        "created_at": datetime.utcnow().isoformat(),
        "backtest":   backtest,
        "performance_history": [],
    }
    rules.append(entry)
    RULES_FILE.write_text(json.dumps(rules, indent=2))
    return entry


def load_generated_rules() -> list:
    """Load all generated rules from the store."""
    if not RULES_FILE.exists():
        return []
    try:
        return json.loads(RULES_FILE.read_text())
    except Exception:
        return []


# Module-level dict of deployed rules — populated on import, kept in sync by
# deploy_rule() and retire_rule(). Imported by main.py /patterns endpoint.
_deployed_rules: dict = {
    r["id"]: r
    for r in load_generated_rules()
    if r.get("status") == "deployed"
}


def deploy_rule(rule_id: str):
    """Promote a shadow rule to deployed status."""
    rules = load_generated_rules()
    for r in rules:
        if r['id'] == rule_id:
            r['status'] = 'deployed'
            r['deployed_at'] = datetime.utcnow().isoformat()
    RULES_FILE.write_text(json.dumps(rules, indent=2))


def retire_rule(rule_id: str, reason: str = "manual"):
    """Retire a deployed rule."""
    rules = load_generated_rules()
    for r in rules:
        if r['id'] == rule_id:
            r['status'] = 'retired'
            r['retired_at'] = datetime.utcnow().isoformat()
            r['retire_reason'] = reason
    RULES_FILE.write_text(json.dumps(rules, indent=2))


def run_pipeline(api_key: str, existing_rules: list) -> dict:
    """
    Full rule factory pipeline: gap extraction → analysis → generation → backtest → save.
    Returns summary of what was generated.
    """
    df = load_transactions()
    if df is None:
        return {"error": "transactions.csv not found"}

    gaps = extract_rule_gaps(df)
    if gaps.empty:
        return {"status": "no_gaps", "message": f"Fewer than {MIN_GAPS_TO_FIRE} rule gaps found"}

    try:
        candidates = analyze_and_generate(gaps, existing_rules, api_key)
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}

    results = []
    for c in candidates:
        if 'fn_code' not in c or 'name' not in c:
            continue
        bt = backtest_rule(c, df, existing_rules)
        rec = bt.get("recommendation", "REJECT")
        status = "deployed" if rec == "AUTO_DEPLOY" else ("shadow" if rec == "SHADOW" else "rejected")

        if status != "rejected":
            entry = save_generated_rule(c, bt, status)
        else:
            entry = {**c, "backtest": bt, "status": "rejected"}

        results.append({
            "name":           c.get("name"),
            "typology":       c.get("typology"),
            "reason":         c.get("reason", "")[:120],
            "backtest":       bt,
            "recommendation": rec,
            "status":         status,
        })

    # Log run
    log = []
    if LOG_FILE.exists():
        try: log = json.loads(LOG_FILE.read_text())
        except: log = []
    log.append({"timestamp": datetime.utcnow().isoformat(), "gaps_analyzed": len(gaps),
                 "candidates": len(candidates), "deployed": sum(1 for r in results if r['status']=='deployed'),
                 "shadow": sum(1 for r in results if r['status']=='shadow')})
    LOG_FILE.write_text(json.dumps(log[-100:], indent=2))  # keep last 100 runs

    return {
        "status":       "ok",
        "gaps_analyzed": len(gaps),
        "candidates":    len(candidates),
        "results":       results,
    }
