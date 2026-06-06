"""
SyntheticID Operator — Real-time fraud pattern detection service.

Loads the trained ML models from ~/pulseml_models and exposes:
  GET  /health            → system health + model info
  GET  /patterns          → full pattern library
  POST /score             → score a single transaction
  GET  /monitor/stream    → SSE stream of live transaction scoring
  GET  /alerts            → recent high-confidence alerts
"""

import asyncio
import json
import os
import pickle
import random
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from match_engine import combined_score, is_alert, score_transaction
from patterns import PATTERNS
from xai import explain_score as _xai_explain, get_model_card as _xai_model_card, list_explanations as _xai_list, get_governance_metrics as _xai_governance
from rule_factory import (
    extract_rule_gaps, run_pipeline, load_generated_rules,
    deploy_rule, retire_rule, backtest_rule, _safe_lambda,
    load_transactions,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────

MODELS_DIR = Path.home() / "pulseml_models"

app = FastAPI(title="SyntheticID Operator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
try:
    scaler  = pickle.load(open(MODELS_DIR / "scaler.pkl",         "rb"))
    xgb     = pickle.load(open(MODELS_DIR / "xgboost.pkl",        "rb"))
    config  = json.load(open(MODELS_DIR  / "model_config.json"))
    df_all  = pd.read_csv(MODELS_DIR     / "transactions.csv")
    FEATURES = config["features"]
    MODEL_OK = True
    print(f"✓ Models loaded — {len(df_all):,} transactions available")
except Exception as e:
    MODEL_OK = False
    FEATURES = []
    df_all   = pd.DataFrame()
    print(f"⚠ Model load failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def ml_score_row(features: dict) -> float:
    """Run XGBoost on a feature dict; returns fraud probability 0–1."""
    if not MODEL_OK or not FEATURES:
        return 0.0
    X = np.array([[float(features.get(f, 0.0)) for f in FEATURES]])
    X_scaled = scaler.transform(X)
    return float(xgb.predict_proba(X_scaled)[0][1])


def build_event(row) -> dict:
    """Score a row and return a full event payload for SSE or REST."""
    if isinstance(row, pd.Series):
        row = row.to_dict()

    features = {f: float(row.get(f, 0.0)) for f in FEATURES}
    ml  = ml_score_row(features)
    matches = score_transaction(features)
    top = matches[0] if matches else None

    c_score = combined_score(ml, top["confidence"]) if top else ml
    alert   = is_alert(c_score) or bool(row.get("is_fraud", False))

    return {
        "transaction_id": str(row.get("transaction_id", f"txn_{random.randint(10000,99999)}")),
        "amount":         round(float(row.get("amount", 0.0)), 2),
        "user_id":        str(row.get("user_id", "unknown")),
        "rail":           str(row.get("payment_rail", "card")),
        "ml_score":       round(ml, 4),
        "top_pattern":    top["pattern_name"] if top and top["confidence"] > 0.35 else None,
        "top_pattern_id": top["pattern_id"]   if top and top["confidence"] > 0.35 else None,
        "pattern_color":  top["color"]         if top and top["confidence"] > 0.35 else "#64748b",
        "confidence":     round(top["confidence"], 4) if top else 0.0,
        "combined_score": round(c_score, 4),
        "is_alert":       alert,
        "matched_signals": top["matched_signals"] if top else [],
        "timestamp":      datetime.utcnow().isoformat() + "Z",
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_OK else "degraded",
        "models_loaded": MODEL_OK,
        "transaction_count": len(df_all),
        "features": FEATURES,
        "model_metrics": config.get("metrics", {}) if MODEL_OK else {},
        "patterns": len(PATTERNS),
    }


@app.get("/patterns")
def get_patterns():
    """Return merged static + deployed generated rules."""
    from rule_factory import _deployed_rules  # noqa: PLC0415
    merged = list(PATTERNS)
    for rule in _deployed_rules.values():
        if rule not in merged:
            merged.append(rule)
    return merged


@app.post("/score")
def score(body: dict):
    """
    Score a single transaction with full XAI explanation.

    Body: any subset of the 10 ML features, or a free-form transaction dict.
    Returns: ml_score, pattern matches, combined score, XAI explanation record.
    """
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded. Run the ML Fraud Engine notebook first.")

    features = {f: float(body.get(f, 0.0)) for f in FEATURES}
    ml       = ml_score_row(features)
    matches  = score_transaction(features)
    top      = matches[0] if matches else None
    c_score  = combined_score(ml, top["confidence"]) if top else ml

    transaction_id = str(body.get("transaction_id", f"txn_{uuid.uuid4().hex[:8]}"))
    explanation = _xai_explain(
        features       = features,
        ml_score       = ml,
        pattern_match  = top,
        combined_score = c_score,
        model          = xgb,
        scaler         = scaler,
        feature_names  = FEATURES,
        model_version  = config.get("version", "1.0.0"),
        transaction_id = transaction_id,
    )

    return {
        "transaction_id": transaction_id,
        "ml_score":       round(ml, 4),
        "combined_score": round(c_score, 4),
        "is_alert":       is_alert(c_score),
        "top_pattern":    top,
        "all_patterns":   matches,
        "explanation":    explanation,
    }


# ── XAI / Explainability Endpoints ───────────────────────────────────────────

@app.get("/xai/explanations")
def xai_list_explanations(
    limit: int = 100,
    verdict: str = "",
    min_score: float = 0.0,
    transaction_id: str = "",
):
    """
    Return recent XAI explanation records from the audit log.

    Query params:
      limit          max records to return (default 100)
      verdict        filter by verdict: LOW | MEDIUM | HIGH | CRITICAL
      min_score      filter by minimum combined score (0.0–1.0)
      transaction_id filter by transaction ID substring
    """
    return _xai_list(
        limit          = limit,
        verdict        = verdict or None,
        min_score      = min_score or None,
        transaction_id = transaction_id or None,
    )


@app.get("/xai/model-card")
def xai_model_card():
    """
    Return the model card for the active fraud detection model.
    Structured per EU AI Act Article 11/13 and Fed SR 26-02 requirements.
    """
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded.")
    return _xai_model_card(config, FEATURES)


@app.get("/xai/governance")
def xai_governance():
    """
    Return live model governance metrics computed from the explanation audit log.
    Includes verdict distribution, score histogram, and top risk drivers.
    """
    return _xai_governance()


@app.get("/xai/explain/{transaction_id}")
def xai_explain_transaction(transaction_id: str):
    """
    Retrieve the stored XAI explanation for a specific transaction.
    Returns the most recent record matching the transaction_id.
    """
    records = _xai_list(limit=1000, transaction_id=transaction_id)
    if not records:
        raise HTTPException(404, f"No explanation found for transaction_id '{transaction_id}'")
    return records[0]


@app.get("/monitor/stream")
async def monitor_stream(speed: float = 0.25, limit: int = 300):
    """
    SSE stream of transactions being scored in real-time.

    Query params:
      speed  — seconds between events (default 0.25 = 4 tx/sec)
      limit  — max transactions to stream (default 300)
    """
    if not MODEL_OK:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'Models not loaded'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Mix fraud + legit for a realistic stream
    fraud_rows  = df_all[df_all["is_fraud"] == True].head(60)  if "is_fraud" in df_all.columns else pd.DataFrame()
    legit_rows  = df_all[df_all["is_fraud"] == False].head(240) if "is_fraud" in df_all.columns else df_all.head(300)
    sample      = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=42).head(limit).reset_index(drop=True)

    async def event_stream():
        stats = {"processed": 0, "alerts": 0}

        for _, row in sample.iterrows():
            event = build_event(row)
            stats["processed"] += 1
            if event["is_alert"]:
                stats["alerts"] += 1

            payload = {**event, "stats": stats.copy()}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(speed)

        yield f"data: {json.dumps({'done': True, 'stats': stats})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/alerts")
def get_alerts(limit: int = 30):
    """Return the most recent high-confidence alerts from the transaction dataset."""
    if not MODEL_OK:
        return []

    # Prioritize confirmed fraud rows for alert demo
    if "is_fraud" in df_all.columns:
        fraud = df_all[df_all["is_fraud"] == True].head(limit)
    else:
        fraud = df_all.head(limit)

    alerts = []
    for _, row in fraud.iterrows():
        event = build_event(row)
        alerts.append(event)

    # Sort by combined score desc
    alerts.sort(key=lambda x: x["combined_score"], reverse=True)
    return alerts[:limit]


# ── Rule Factory Endpoints ────────────────────────────────────────────────────

@app.get("/rule-factory/gaps")
def get_rule_gaps(limit: int = 50):
    """
    Return transactions where ML fired (>0.70) but rules missed (rule_score<30).
    These are the training signal for new rule generation.
    """
    if not MODEL_OK:
        return {"gaps": [], "count": 0}

    # Always reload from disk so post-notebook saves are picked up
    try:
        df_live = pd.read_csv(MODELS_DIR / "transactions.csv")
    except Exception:
        df_live = df_all

    gaps = extract_rule_gaps(df_live)
    if gaps.empty:
        return {"gaps": [], "count": 0, "message": "No rule gaps found yet — good coverage!"}

    preview_cols = [c for c in [
        'transaction_id','amount','payment_rail','fraud_typology',
        'ensemble_score','rule_score','rule_name',
    ] if c in gaps.columns]

    sample = gaps[preview_cols].head(limit).fillna("").to_dict("records")
    return {
        "count":   len(gaps),
        "sample":  sample,
        "feature_means": {
            f: round(float(gaps[f].mean()), 4)
            for f in gaps.columns
            if f in [
                'amount_zscore','amount_vs_max','rail_risk','recipient_familiarity',
                'device_familiarity','velocity_1h','is_crypto','is_instant_rail',
            ] and f in gaps.columns and not gaps[f].isna().all()
        },
    }


@app.post("/rule-factory/run")
async def run_rule_factory(body: dict = {}):
    """
    Trigger the full rule factory pipeline:
    gap extraction → Claude analysis → rule generation → backtest → save.
    Returns candidates with recommendations.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VITE_ANTHROPIC_API_KEY")
    if not api_key:
        # Try reading from .env file
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "ANTHROPIC_API_KEY" in line or "VITE_ANTHROPIC_API_KEY" in line:
                    api_key = line.split("=", 1)[-1].strip()
                    break

    if not api_key:
        raise HTTPException(400, "ANTHROPIC_API_KEY not found in environment or .env file")

    # Use existing rule definitions as context for deduplication
    from patterns import PATTERNS as pattern_defs
    existing_rules = [{"name": p["name"], "tier": 0, "reason": p["description"]} for p in pattern_defs]

    result = run_pipeline(api_key, existing_rules)
    return result


@app.get("/rule-factory/rules")
def list_generated_rules():
    """Return all generated rules with their status and backtest metrics."""
    rules = load_generated_rules()
    return {
        "total":    len(rules),
        "deployed": sum(1 for r in rules if r["status"] == "deployed"),
        "shadow":   sum(1 for r in rules if r["status"] == "shadow"),
        "retired":  sum(1 for r in rules if r["status"] == "retired"),
        "rules":    rules,
    }


@app.post("/rule-factory/deploy/{rule_id}")
def deploy_generated_rule(rule_id: str):
    """Promote a shadow rule to deployed status."""
    deploy_rule(rule_id)
    return {"status": "deployed", "rule_id": rule_id}


@app.post("/rule-factory/retire/{rule_id}")
def retire_generated_rule(rule_id: str, body: dict = {}):
    """Retire a deployed or shadow rule."""
    retire_rule(rule_id, body.get("reason", "manual"))
    return {"status": "retired", "rule_id": rule_id}


@app.post("/rule-factory/test")
def test_candidate_rule(body: dict):
    """
    Quick backtest a candidate rule before saving.
    Body: { fn_code: "lambda r: ...", name: "...", reason: "..." }
    """
    if not MODEL_OK or df_all.empty:
        raise HTTPException(503, "Transactions not loaded")

    fn_code = body.get("fn_code", "")
    if not fn_code:
        raise HTTPException(400, "fn_code is required")

    result = backtest_rule(body, df_all, [])
    return result


# ── Network Graph ─────────────────────────────────────────────────────────────

@app.get("/network/graph")
def get_network_graph(
    typology: str = "",
    days: int = 90,
    min_score: float = 0.0,
    fraud_only: bool = False,
    limit_nodes: int = 400,
):
    """
    Build a fraud network graph from transactions.csv.
    Returns nodes (users, devices, recipients) and edges (transactions).
    """
    if not MODEL_OK:
        return {"nodes": [], "links": [], "stats": {}}

    try:
        df = pd.read_csv(MODELS_DIR / "transactions.csv")
    except Exception:
        return {"nodes": [], "links": [], "stats": {}}

    # Filters
    if fraud_only and "is_fraud" in df.columns:
        df = df[df["is_fraud"] == True]
    if min_score > 0 and "ensemble_score" in df.columns:
        df = df[df["ensemble_score"] >= min_score]
    if typology and "fraud_typology" in df.columns:
        df = df[df["fraud_typology"] == typology]

    # Work with a manageable sample — prioritise fraud rows
    if len(df) > limit_nodes * 3:
        fraud_df  = df[df["is_fraud"] == True] if "is_fraud" in df.columns else pd.DataFrame()
        legit_df  = df[df["is_fraud"] == False] if "is_fraud" in df.columns else df
        n_legit   = max(0, limit_nodes * 3 - len(fraud_df))
        df = pd.concat([fraud_df, legit_df.sample(min(n_legit, len(legit_df)), random_state=42)])

    nodes = {}
    links = []

    def ensure_node(nid, ntype, label, fraud_count=0, tx_count=0, score=0.0, typology="", cluster=None):
        if nid not in nodes:
            nodes[nid] = {
                "id":          nid,
                "type":        ntype,
                "label":       label,
                "fraud_count": 0,
                "tx_count":    0,
                "max_score":   0.0,
                "typology":    typology,
                "cluster":     cluster,
            }
        n = nodes[nid]
        n["fraud_count"] += fraud_count
        n["tx_count"]    += tx_count
        n["max_score"]   = max(n["max_score"], score)
        if typology and not n["typology"]:
            n["typology"] = typology

    for _, row in df.iterrows():
        uid   = str(row.get("user_id",    ""))
        did   = str(row.get("device_id",  ""))
        rid   = str(row.get("recipient_id",""))
        is_f  = bool(row.get("is_fraud",  False))
        score = float(row.get("ensemble_score", 0.0)) if not pd.isna(row.get("ensemble_score", float("nan"))) else 0.0
        typo  = str(row.get("fraud_typology", "")) if not pd.isna(row.get("fraud_typology", float("nan"))) else ""
        amt   = float(row.get("amount", 0.0)) if not pd.isna(row.get("amount", float("nan"))) else 0.0

        if uid:
            ensure_node(f"u_{uid}", "user", uid, int(is_f), 1, score, typo)
        if did and did not in ("nan", ""):
            ensure_node(f"d_{did}", "device", did, int(is_f), 1, score)
        if rid and rid not in ("nan", ""):
            ensure_node(f"r_{rid}", "recipient", rid, int(is_f), 1, score, typo)

        # user → recipient (transaction edge)
        if uid and rid and rid not in ("nan", ""):
            links.append({
                "source":  f"u_{uid}",
                "target":  f"r_{rid}",
                "is_fraud": is_f,
                "amount":   round(amt, 2),
                "score":    round(score, 4),
                "typology": typo,
            })
        # user → device (fingerprint edge)
        if uid and did and did not in ("nan", ""):
            links.append({
                "source":  f"u_{uid}",
                "target":  f"d_{did}",
                "is_fraud": is_f,
                "amount":   0,
                "score":    round(score, 4),
                "typology": "",
            })

    # Flag shared devices (≥3 distinct users sharing same device)
    device_user_counts: dict[str, set] = {}
    for row in df.itertuples():
        did = str(getattr(row, "device_id", ""))
        uid = str(getattr(row, "user_id", ""))
        if did and did != "nan":
            device_user_counts.setdefault(f"d_{did}", set()).add(uid)

    for nid, users in device_user_counts.items():
        if nid in nodes and len(users) >= 3:
            nodes[nid]["shared_device"] = True
            nodes[nid]["shared_users"]  = len(users)

    # Flag high-volume recipients (≥5 fraud txns)
    for nid, n in nodes.items():
        if n["type"] == "recipient" and n["fraud_count"] >= 5:
            n["mule_flag"] = True

    node_list = list(nodes.values())
    stats = {
        "total_nodes":    len(node_list),
        "user_nodes":     sum(1 for n in node_list if n["type"] == "user"),
        "device_nodes":   sum(1 for n in node_list if n["type"] == "device"),
        "recipient_nodes":sum(1 for n in node_list if n["type"] == "recipient"),
        "total_edges":    len(links),
        "fraud_edges":    sum(1 for l in links if l["is_fraud"]),
        "shared_devices": sum(1 for n in node_list if n.get("shared_device")),
        "mule_accounts":  sum(1 for n in node_list if n.get("mule_flag")),
    }

    return {"nodes": node_list, "links": links, "stats": stats}


@app.get("/network/typologies")
def get_typologies():
    """Return distinct fraud typologies available for filtering."""
    try:
        df = pd.read_csv(MODELS_DIR / "transactions.csv")
    except Exception:
        return []
    if "fraud_typology" not in df.columns:
        return []
    fraud = df[df["is_fraud"] == True] if "is_fraud" in df.columns else df
    typos = [t for t in fraud["fraud_typology"].dropna().unique().tolist() if t and t != "none"]
    return sorted(typos)


# ── SyntheticID Ingest ────────────────────────────────────────────────────────

_TYPOLOGY_MAP = {
    "synthetic": "synthetic_identity",
    "identity": "synthetic_identity",
    "ato": "ai_powered_ato",
    "account takeover": "ai_powered_ato",
    "credential": "ai_powered_ato",
    "deepfake": "deepfake_social_engineering",
    "social engineering": "deepfake_social_engineering",
    "pig": "pig_butchering",
    "romance": "pig_butchering",
    "investment": "pig_butchering",
    "app scam": "app_scam",
    "authorised push": "app_scam",
    "card": "card_testing_bot",
    "carding": "card_testing_bot",
    "bot": "card_testing_bot",
}

def _infer_typology(platform: str, step_name: str, step_desc: str) -> str:
    combined = (platform + " " + step_name + " " + step_desc).lower()
    for keyword, typology in _TYPOLOGY_MAP.items():
        if keyword in combined:
            return typology
    return "synthetic_identity"  # safe default for onboarding simulations


@app.post("/syntheticid/ingest")
def ingest_syntheticid(body: dict):
    """
    Accept a SyntheticID Lab simulation result and convert BYPASSED attack
    steps into labeled fraud gap rows appended to transactions.csv.

    These rows satisfy extract_rule_gaps criteria (is_fraud=True,
    ensemble_score>0.70, rule_score<30) with a named typology so Rule Factory
    can generate typology-specific rules.
    """
    csv_path = MODELS_DIR / "transactions.csv"
    if not csv_path.exists():
        raise HTTPException(503, "transactions.csv not found — run the ML notebook first")

    platform     = body.get("platform", "Fintech")
    sophistication = body.get("sophistication", "AI Fraud Agent")
    timeline     = body.get("attack_timeline", [])
    gap_map      = body.get("detection_gap_map", {})
    exp_scores   = body.get("exposure_scores", {})

    bypassed_steps = [s for s in timeline if s.get("outcome") == "BYPASSED"]
    if not bypassed_steps:
        return {"inserted": 0, "message": "No BYPASSED steps found — nothing to ingest"}

    df = pd.read_csv(csv_path)

    # Ensure rule_score column exists; fill NaN for legacy rows (won't match <30)
    if "rule_score" not in df.columns:
        df["rule_score"] = float("nan")

    overall_exposure = exp_scores.get("overall", 75)
    synthetic_rows = []

    for step in bypassed_steps:
        typology = _infer_typology(platform, step.get("name", ""), step.get("description", ""))
        # Scale amount by exposure: higher exposure → larger fraud amounts
        amount = round(500 + (overall_exposure / 100) * 4500 + random.uniform(-200, 200), 2)
        rail = "crypto" if "crypto" in platform.lower() else (
               "zelle" if "p2p" in platform.lower() or "neobank" in platform.lower() else "wire")

        row = {c: float("nan") for c in df.columns}
        row.update({
            "transaction_id":    f"synth_{uuid.uuid4().hex[:10]}",
            "user_id":           f"synth_user_{uuid.uuid4().hex[:6]}",
            "amount":            amount,
            "timestamp":         datetime.utcnow().isoformat(),
            "hour":              datetime.utcnow().hour,
            "payment_rail":      rail,
            "is_fraud":          True,
            "fraud_typology":    typology,
            "is_crypto":         1.0 if rail == "crypto" else 0.0,
            "is_instant_rail":   1.0,
            "ensemble_score":    round(0.82 + random.uniform(0, 0.12), 4),
            "rule_score":        0.0,
            "xgb_score":         round(0.80 + random.uniform(0, 0.15), 4),
            "iso_score":         round(0.70 + random.uniform(0, 0.20), 4),
            "velocity_1h":       random.randint(3, 8),
            "is_new_recipient":  1.0,
        })
        synthetic_rows.append(row)

    new_df = pd.DataFrame(synthetic_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    return {
        "inserted":   len(synthetic_rows),
        "typologies": list({r["fraud_typology"] for r in synthetic_rows}),
        "message":    f"Ingested {len(synthetic_rows)} adversarial gap rows from '{platform}' simulation. Run Rule Factory to generate new rules.",
    }


# ── LLM Proxy ─────────────────────────────────────────────────────────────────
# Provider-agnostic proxy: anthropic | openai | groq | mistral
# API key never touches the browser — stored in operator/.env only.
#
# operator/.env:
#   LLM_PROVIDER=anthropic     # anthropic | openai | groq | mistral
#   LLM_API_KEY=sk-ant-...
#   LLM_MODEL=claude-sonnet-4-6   # optional override

_LLM_OAI_ENDPOINTS = {
    "openai":  "https://api.openai.com/v1/chat/completions",
    "groq":    "https://api.groq.com/openai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "groq":      "llama-3.1-70b-versatile",
    "mistral":   "mistral-large-latest",
}

@app.post("/llm/proxy")
async def llm_proxy(body: dict):
    """
    Route LLM requests to anthropic / openai / groq / mistral.
    Reads LLM_PROVIDER, LLM_API_KEY, LLM_MODEL from environment.
    Streams back SSE for stream=true, returns JSON for stream=false.
    """
    import httpx

    provider   = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    api_key    = os.environ.get("LLM_API_KEY", "")
    model      = body.get("model") or os.environ.get("LLM_MODEL") or _DEFAULT_MODELS.get(provider, "claude-sonnet-4-6")
    system     = body.get("system", "")
    messages   = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 2000))
    stream     = bool(body.get("stream", False))

    if not api_key:
        raise HTTPException(400, "LLM_API_KEY not set in operator/.env")

    # ── Anthropic path ────────────────────────────────────────────────────────
    if provider == "anthropic":
        endpoint = "https://api.anthropic.com/v1/messages"
        payload  = {"model": model, "max_tokens": max_tokens, "messages": messages, "stream": stream}
        if system:
            payload["system"] = system
        headers  = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        if stream:
            async def generate_anthropic():
                async with httpx.AsyncClient(timeout=60) as client:
                    async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            return StreamingResponse(generate_anthropic(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, resp.text)
            data    = resp.json()
            content = data["content"][0]["text"] if data.get("content") else ""
            return {"content": content}

    # ── OpenAI-compatible path (openai / groq / mistral) ─────────────────────
    endpoint = _LLM_OAI_ENDPOINTS.get(provider)
    if not endpoint:
        raise HTTPException(400, f"Unsupported provider '{provider}'. Supported: anthropic, openai, groq, mistral")

    oai_messages = [{"role": "system", "content": system}] + messages
    payload      = {"model": model, "messages": oai_messages, "max_tokens": max_tokens, "stream": stream}
    headers      = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    if stream:
        async def generate_oai():
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        return StreamingResponse(generate_oai(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"content": content}


# ── Integration Hub ───────────────────────────────────────────────────────────

from integrations import hub as _hub
from integrations.base import EnrichRequest, ReportRequest, ConnectorCategory


@app.get("/integrations/connectors")
def list_integration_connectors():
    """Return all registered connectors with their configuration and status."""
    return _hub.list_connectors()


@app.get("/integrations/health")
def integration_health():
    """Return health status of every connector."""
    return _hub.health()


@app.post("/integrations/enrich")
def enrich_transaction(body: dict):
    """
    Enrich a transaction using one or more external connectors concurrently.

    Body:
      transaction_id  str   (required)
      user_id         str   (required)
      amount          float (required)
      device_id       str   (optional)
      ip_address      str   (optional)
      email           str   (optional)
      phone           str   (optional)
      connectors      list  connector IDs to run, e.g. ["ofac", "threatmetrix"]
      categories      list  category names, e.g. ["FINANCIAL_INTEL", "FRAUD_CONSORTIUM"]
                            ignored when connectors is provided
      timeout         int   per-connector timeout in seconds (default 5)
    """
    if not body.get("transaction_id") or not body.get("user_id"):
        raise HTTPException(400, "transaction_id and user_id are required")

    req = EnrichRequest(
        transaction_id = body["transaction_id"],
        user_id        = body["user_id"],
        amount         = float(body.get("amount", 0.0)),
        device_id      = body.get("device_id"),
        ip_address     = body.get("ip_address"),
        email          = body.get("email"),
        phone          = body.get("phone"),
        metadata       = {k: v for k, v in body.items()
                          if k not in ("transaction_id","user_id","amount","device_id",
                                       "ip_address","email","phone","connectors","categories","timeout")},
    )

    connector_ids = body.get("connectors") or None
    categories    = None
    if not connector_ids and body.get("categories"):
        categories = [ConnectorCategory(c) for c in body["categories"] if c in ConnectorCategory._value2member_map_]

    timeout = int(body.get("timeout", 5))
    return _hub.enrich(req, connectors=connector_ids, categories=categories, timeout=timeout)


@app.post("/integrations/report")
def report_fraud(body: dict):
    """
    Submit a fraud report or regulatory filing to specified connectors.

    Body:
      transaction_id  str          (required)
      user_id         str          (required)
      report_type     str          e.g. "SAR", "CTR", "FRAUD_RING_REFERRAL"
      connectors      list[str]    connector IDs to report to (required)
      amount          float
      description     str
      evidence        dict
      timeout         int          per-connector timeout in seconds (default 15)
    """
    if not body.get("transaction_id") or not body.get("user_id"):
        raise HTTPException(400, "transaction_id and user_id are required")
    if not body.get("connectors"):
        raise HTTPException(400, "connectors list is required — specify which agencies to report to")

    req = ReportRequest(
        transaction_id = body["transaction_id"],
        user_id        = body["user_id"],
        report_type    = body.get("report_type", "FRAUD_REFERRAL"),
        amount         = float(body.get("amount", 0.0)),
        description    = body.get("description", ""),
        evidence       = body.get("evidence", {}),
    )

    timeout = int(body.get("timeout", 15))
    return _hub.report(req, connectors=body["connectors"], timeout=timeout)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
