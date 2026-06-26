"""
SyntheticID Operator - Real-time fraud pattern detection service.

Loads the trained ML models from ~/pulseml_models and exposes:
  GET  /health            → system health + model info
  GET  /patterns          → full pattern library
  POST /score             → score a single transaction (one-shot, no pipeline routing)
  GET  /monitor/stream    → SSE stream - drains injection buffer, falls back to historical
  GET  /alerts            → recent high-confidence alerts
  POST /ingest            → inject a live transaction into the full scoring pipeline
  POST /ingest/batch      → inject up to 1 000 transactions in one call
  GET  /ingest/stats      → injection buffer + log stats
"""

import asyncio
import json
import os
import pickle
import random
import time
import uuid
from collections import deque
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
from agent import (
    agent_state, agent_config, run_agent,
    novel_attack_buffer, _event_subscribers,
    load_config, save_config, validate_config, THREAT_META,
)
import drift_monitor
import graph_features
import gnn_lite
import case_file
import fraud_env
import adversary
import feedback

# ── Bootstrap ─────────────────────────────────────────────────────────────────

# Path to the ML backend (pulseml_models / redwing-ml): its trained models AND its
# shared feature foundation (features.py, graph_layer.py) are loaded from here, so the
# operator computes features identically to training. Override for non-default deploys.
MODELS_DIR = Path(os.environ.get("REDWING_MODELS_DIR", Path.home() / "pulseml_models"))

app = FastAPI(title="SyntheticID Operator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
import sys
sys.path.insert(0, str(MODELS_DIR))   # share the ML backend's feature foundation
_feedback = None   # closed-loop feedback store (set once the reputation layer loads)
try:
    # Prefer the retrained, skew-free model + scaler when present.
    if (MODELS_DIR / "xgboost_retrained.pkl").exists():
        scaler = pickle.load(open(MODELS_DIR / "scaler_retrained.pkl", "rb"))
        xgb    = pickle.load(open(MODELS_DIR / "xgboost_retrained.pkl", "rb"))
        MODEL_TAG = "retrained"
    else:
        scaler = pickle.load(open(MODELS_DIR / "scaler.pkl",  "rb"))
        xgb    = pickle.load(open(MODELS_DIR / "xgboost.pkl", "rb"))
        MODEL_TAG = "original"
    config  = json.load(open(MODELS_DIR  / "model_config.json"))
    df_all  = pd.read_csv(MODELS_DIR     / "transactions.csv")
    FEATURES = config["features"]
    MODEL_OK = True
    print(f"✓ Models loaded ({MODEL_TAG}) - {len(df_all):,} transactions available")
    # Shared feature foundation - the SAME transform used to train the model, so the
    # operator computes features identically to training (no training-serving skew).
    try:
        import features as mlfeat
        from graph_layer import RecipientReputation
        _rep = (RecipientReputation.load()
                if (MODELS_DIR / "recipient_reputation.json").exists() else None)
        FEATURE_ENGINE = mlfeat.FeatureEngineer(mlfeat.build_profiles(), _rep)
        print(f"✓ Feature foundation loaded - {len(FEATURE_ENGINE.profiles):,} user profiles")
        # Closed feedback loop: dispositions online-update the SAME reputation instance
        # the feature foundation reads, so a confirmed-fraud payee scores higher at once.
        _feedback = feedback.FeedbackStore(MODELS_DIR / "feedback_log.jsonl", reputation=_rep)
        print(f"✓ Feedback loop wired - {_feedback.status()['labeled_total']} prior labels")
    except Exception as _fe:
        FEATURE_ENGINE = None
        print(f"⚠ Feature foundation unavailable ({_fe}); using raw feature passthrough")
    graph_features.precompute(df_all)
    print(f"✓ Graph features precomputed - {graph_features.get_stats()['entities']:,} entities indexed")
    gnn_lite.init(df_all)
    print(f"✓ GNN Tier 2 initialised - {gnn_lite.get_stats()['users']:,} user embeddings")
except Exception as e:
    MODEL_OK = False
    FEATURES = []
    df_all   = pd.DataFrame()
    FEATURE_ENGINE = None
    print(f"⚠ Model load failed: {e}")

# ── Real-data payment model (ULB Credit Card Fraud) - engine-validation anchor ──
# Independent of the synthetic pipeline above: this is the ONE model trained and
# validated on REAL labels. A missing artifact must not break the main operator.
import xgboost as xgblib  # noqa: E402
PAYMENT_REAL = None
try:
    _pm_booster = xgblib.Booster()
    _pm_booster.load_model(str(MODELS_DIR / "payment_real_xgb.json"))
    _pm_meta = json.load(open(MODELS_DIR / "payment_real_meta.json"))
    _pm_best_it = int(_pm_meta.get("model", {}).get("best_iteration", 0))
    PAYMENT_REAL = {
        "booster":   _pm_booster,
        "platt":     pickle.load(open(MODELS_DIR / "payment_real_platt.pkl", "rb")),
        "meta":      _pm_meta,
        "feats":     _pm_meta["feature_order"],
        "threshold": float(_pm_meta["metrics"]["threshold"]),
        # Serve with the SAME tree range training calibrated on - no serving skew.
        "iter_range": (0, _pm_best_it + 1) if _pm_best_it else None,
    }
    print(f"✓ Real-data payment model loaded - PR-AUC {_pm_meta['metrics']['pr_auc']} (ULB, real labels)")
except Exception as _pe:
    print(f"⚠ Real-data payment model unavailable ({_pe}) - run payment_real_model.py")

# ── Injection pipeline state ───────────────────────────────────────────────────

_ingest_buffer:   deque = deque(maxlen=500)   # ring buffer - latest injected events
_ingest_log_path: Path  = MODELS_DIR / "ingest_log.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_features(raw: dict) -> dict:
    """Compute the model's features through the shared foundation (the same transform
    used at training → no training-serving skew). Falls back to raw passthrough only
    if the foundation is unavailable - which is the legacy zero-fill behaviour."""
    if FEATURE_ENGINE is not None:
        return FEATURE_ENGINE.compute(raw)
    return {f: float(raw.get(f, 0.0)) for f in FEATURES}


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

    features = compute_features(row)
    ml  = ml_score_row(features)
    matches = score_transaction(features)
    top = matches[0] if matches else None

    c_score = combined_score(ml, top["confidence"]) if top else ml

    # ── Tier 2: GNN cascade (borderline transactions only) ────────────────────
    gnn_result = None
    if gnn_lite.should_invoke(c_score):
        gnn_result = gnn_lite.score(
            row.get("user_id"), row.get("device_id"), row.get("recipient_id")
        )
        cascade_score = gnn_lite.cascade_blend(c_score, gnn_result)
    else:
        cascade_score = c_score

    alert = is_alert(cascade_score) or bool(row.get("is_fraud", False))

    # ── Tier 3: offline graph context (O(1) lookup) ───────────────────────────
    graph_ctx = graph_features.get_features(
        user_id      = row.get("user_id"),
        device_id    = row.get("device_id"),
        recipient_id = row.get("recipient_id"),
    )

    # ── Drift monitoring - non-blocking, appends to rolling buffer ────────────
    drift_monitor.record(ml, features)

    return {
        "transaction_id":    str(row.get("transaction_id", f"txn_{random.randint(10000,99999)}")),
        "amount":            round(float(row.get("amount", 0.0)), 2),
        "user_id":           str(row.get("user_id", "unknown")),
        "rail":              str(row.get("payment_rail", "card")),
        "ml_score":          round(ml, 4),
        "top_pattern":       top["pattern_name"] if top and top["confidence"] > 0.35 else None,
        "top_pattern_id":    top["pattern_id"]   if top and top["confidence"] > 0.35 else None,
        "pattern_color":     top["color"]         if top and top["confidence"] > 0.35 else "#64748b",
        "confidence":        round(top["confidence"], 4) if top else 0.0,
        "tier1_score":       round(c_score, 4),
        "tier2_gnn_score":   round(gnn_result.score, 4) if gnn_result else None,
        "tier2_invoked":     gnn_result is not None,
        "combined_score":    round(cascade_score, 4),
        "is_alert":          alert,
        "matched_signals":   top["matched_signals"] if top else [],
        "graph_context":     graph_ctx,
        "graph_risk_score":  graph_ctx["graph_risk_score"],
        "timestamp":         datetime.utcnow().isoformat() + "Z",
    }


# ── Autonomous Agent Startup ──────────────────────────────────────────────────

@app.on_event("startup")
async def start_autonomous_agent():
    """Start the SyntheticID agent and schedule hourly graph feature refresh."""
    if MODEL_OK and not agent_state.running:
        asyncio.create_task(run_agent(build_event, df_all, FEATURES))
    asyncio.create_task(_graph_refresh_loop())


async def _graph_refresh_loop() -> None:
    """Refresh graph features every hour so the ring-detection context stays current."""
    while True:
        await asyncio.sleep(3600)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, graph_features.refresh_from_disk, MODELS_DIR)
        await loop.run_in_executor(None, gnn_lite.refresh_from_disk, MODELS_DIR)


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


@app.get("/privacy/curve")
def privacy_curve():
    """Differential-privacy utility curve from the ML engine (privacy_layer.py):
    event- and user-level DP trade-off on the cross-user graph signal."""
    p = MODELS_DIR / "privacy_utility_curve.json"
    if not p.exists():
        raise HTTPException(404, "privacy_utility_curve.json not found - run privacy_layer.py")
    return json.loads(p.read_text())


@app.get("/consortium/demo")
def consortium_demo():
    """Privacy-preserving cross-institution fraud network: the network-effect scale
    curve, the differential-privacy utility curve, flagship cross-bank mules, and the
    real-data anchor. Built by redwing-ml/consortium_build.py."""
    p = MODELS_DIR / "consortium_demo.json"
    if not p.exists():
        raise HTTPException(404, "consortium_demo.json not found - run consortium_build.py")
    return json.loads(p.read_text())


@app.get("/observability/skew")
def observability_skew():
    """Training-serving skew analysis - measured before/after the feature-foundation
    fix. Same model, same thresholds; only feature reproduction changed."""
    return {
        "offline_auc": 0.984,
        "field_catch_before_pct": 0.3,
        "field_catch_after_pct": 91.0,
        "feature_count": len(FEATURES) or 23,
        "features_reproducible_before": 13,
        "root_cause": [
            "10 of 23 features had no reproducible definition at serving time",
            "They silently defaulted to zero - including top-weighted features",
            "~24% of the model's signal was dead in production",
            "Invisible to offline AUC, computed where the features still exist",
        ],
        "fix": [
            "One feature foundation computed identically for training and serving",
            "train == serve → skew impossible by construction",
            "23/23 features restored; field catch 0.3% → 91%",
        ],
    }


@app.get("/payment/meta")
def payment_meta():
    """Real-data validation report for the ULB card-fraud model - PR-AUC headline,
    PR curve, confusion, feature importance, and honest held-out samples."""
    if not PAYMENT_REAL:
        raise HTTPException(404, "Real-data payment model not built - run payment_real_model.py")
    return PAYMENT_REAL["meta"]


@app.post("/score/payment")
def score_payment(body: dict):
    """Live inference through the REAL-data ULB model. Accepts V1..V28 + Amount
    (or a `features` dict). Returns Platt-calibrated P(fraud) + the decision against
    the calibration-tuned threshold."""
    if not PAYMENT_REAL:
        raise HTTPException(503, "Real-data payment model not loaded.")
    import math
    src = body.get("features", body)
    row = []
    for f in PAYMENT_REAL["feats"]:
        if f == "log_amount":
            row.append(float(src["log_amount"]) if "log_amount" in src
                       else math.log1p(float(src.get("Amount", src.get("amount", 0.0)))))
        else:
            row.append(float(src.get(f, 0.0)))
    _dm = xgblib.DMatrix(np.array([row], dtype=float))
    _ir = PAYMENT_REAL.get("iter_range")
    raw = float(PAYMENT_REAL["booster"].predict(_dm, iteration_range=_ir)[0] if _ir
                else PAYMENT_REAL["booster"].predict(_dm)[0])
    p = float(PAYMENT_REAL["platt"].predict_proba([[raw]])[0][1])
    thr = PAYMENT_REAL["threshold"]
    return {"p_fraud": round(p, 4), "raw_score": round(raw, 4), "threshold": round(thr, 4),
            "decision": "BLOCK" if p >= thr else "ALLOW"}


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

    features = compute_features(body)
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

    Drains the injection buffer first (real injected transactions), then falls
    back to historical dataset replay so the stream never goes silent.

    Query params:
      speed  - seconds between events (default 0.25 = 4 tx/sec)
      limit  - max transactions to stream (default 300)
    """
    if not MODEL_OK:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'Models not loaded'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Snapshot the injection buffer before building the historical fallback
    injected = list(_ingest_buffer)

    # Historical fallback: mix fraud + legit
    fraud_rows = df_all[df_all["is_fraud"] == True].head(60)  if "is_fraud" in df_all.columns else pd.DataFrame()
    legit_rows = df_all[df_all["is_fraud"] == False].head(240) if "is_fraud" in df_all.columns else df_all.head(300)
    historical = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=42).reset_index(drop=True)

    async def event_stream():
        stats = {"processed": 0, "alerts": 0, "injected": 0, "historical": 0}
        emitted = 0

        # 1. Drain injection buffer (already scored - emit directly)
        for event in injected:
            if emitted >= limit:
                break
            stats["processed"] += 1
            stats["injected"]   += 1
            if event.get("is_alert"):
                stats["alerts"] += 1
            yield f"data: {json.dumps({**event, 'source': 'injected', 'stats': stats.copy()})}\n\n"
            await asyncio.sleep(speed)
            emitted += 1

        # 2. Historical replay to fill remaining quota
        for _, row in historical.iterrows():
            if emitted >= limit:
                break
            event = build_event(row)
            stats["processed"]  += 1
            stats["historical"] += 1
            if event["is_alert"]:
                stats["alerts"] += 1
            yield f"data: {json.dumps({**event, 'source': 'historical', 'stats': stats.copy()})}\n\n"
            await asyncio.sleep(speed)
            emitted += 1

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


# ── Investigator Case File ────────────────────────────────────────────────────

def _assemble_case(row) -> dict:
    """Score a transaction row and assemble the full investigator case file."""
    scored = build_event(row)
    graph_ctx = scored.get("graph_context") or {}

    # Best-effort XAI explanation for the alert panel's top features.
    explanation = None
    try:
        features = compute_features(row)
        ml = ml_score_row(features)
        matches = score_transaction(features)
        top = matches[0] if matches else None
        c_score = combined_score(ml, top["confidence"]) if top else ml
        explanation = _xai_explain(
            features=features, ml_score=ml, pattern_match=top, combined_score=c_score,
            model=xgb, scaler=scaler, feature_names=FEATURES,
            model_version=config.get("version", "1.0.0"),
            transaction_id=str(row.get("transaction_id", "")),
        )
    except Exception:
        explanation = None

    case = case_file.assemble(row, scored, graph_ctx=graph_ctx, explanation=explanation)

    # External enrichment via the connector hub (credit bureaus, fraud consortia,
    # sanctions, open banking). Live API when credentialed, else derived signals -
    # this is what populates the identity/device view the feature families scaffolded.
    try:
        er = _hub.enrich(EnrichRequest(
            transaction_id=str(row.get("transaction_id", "")),
            user_id=str(row.get("user_id", "")),
            amount=float(row.get("amount", 0.0) or 0.0),
            payment_rail=str(row.get("payment_rail", row.get("rail", ""))),
            recipient_id=str(row.get("recipient_id", "")),
            fraud_typology=str(row.get("fraud_typology", "")),
            raw=row,
        ))
        case["enrichment"] = er
    except Exception:
        case["enrichment"] = None

    return case


@app.get("/case/{transaction_id}")
def get_case(transaction_id: str):
    """Full investigator case file for one transaction - the decisioning surface a
    fraud analyst works from: Customer 360 / CDD, card-usage detail, card-fraud
    signals, dispute-evidence study, device/network context, timeline, and a
    recommended disposition. SAR is a downstream action, not the entry point."""
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded.")
    if df_all.empty or "transaction_id" not in df_all.columns:
        raise HTTPException(404, "No transaction dataset loaded.")

    match = df_all[df_all["transaction_id"].astype(str) == str(transaction_id)]
    if match.empty:
        raise HTTPException(404, f"transaction_id '{transaction_id}' not found.")
    return _assemble_case(match.iloc[0].to_dict())


@app.post("/case")
def post_case(body: dict):
    """Assemble a case file from an ad-hoc transaction payload (e.g. an injected or
    streamed transaction not in the historical dataset)."""
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded.")
    return _assemble_case(body)


# ── Agent-Evaluation Environment ──────────────────────────────────────────────
# The investigator case workbench, exposed as a resettable environment an agent can
# be evaluated against: known case state → bounded action space → trajectory →
# process + outcome verifiers. See fraud_env.py.

def _case_for_env(transaction_id: str) -> dict:
    if not MODEL_OK or df_all.empty or "transaction_id" not in df_all.columns:
        raise HTTPException(404, "No transaction dataset loaded.")
    match = df_all[df_all["transaction_id"].astype(str) == str(transaction_id)]
    if match.empty:
        raise HTTPException(404, f"transaction_id '{transaction_id}' not found.")
    return _assemble_case(match.iloc[0].to_dict())


@app.get("/env/spec")
def env_spec():
    """The environment contract: observation schema, action space, reward design."""
    return fraud_env.env_spec()


@app.post("/env/run")
def env_run(body: dict):
    """Run a reference policy end-to-end on one case and return its trajectory +
    verifier scorecard. Body: {transaction_id, agent}. agent ∈ investigator |
    trigger_happy | cautious."""
    case = _case_for_env(str(body.get("transaction_id", "")))
    return fraud_env.run_episode(case, agent=str(body.get("agent", "investigator")))


@app.post("/env/run-all")
def env_run_all(body: dict):
    """Run every reference policy on one case - shows that the verifiers discriminate
    a disciplined investigator from naive baselines. Body: {transaction_id}."""
    case = _case_for_env(str(body.get("transaction_id", "")))
    return {
        "transaction_id": case.get("transaction_id"),
        "case_id": case.get("case_id"),
        "ground_truth_label": case.get("alert", {}).get("ground_truth_label"),
        "gold_disposition": fraud_env.gold_disposition(case),
        "runs": [fraud_env.run_episode(case, agent=a) for a in fraud_env.POLICIES],
    }


@app.post("/env/step")
def env_step(body: dict):
    """One stateless step so ANY agent (LLM or otherwise) can drive the environment.
    Body: {transaction_id, history:[actions], action} → observation, reward, done, info."""
    case = _case_for_env(str(body.get("transaction_id", "")))
    return fraud_env.step(case, body.get("history", []), str(body.get("action", "")))


# ── Adversary Simulator ───────────────────────────────────────────────────────
# Mutates a seed fraud with cost-tagged evasions and re-scores against the live
# model to measure detection decay. See adversary.py.

@app.get("/adversary/strategies")
def adversary_strategies():
    """The cost-tagged evasion registry (cheap = adversary controls for free)."""
    return adversary.strategies()


@app.post("/adversary/simulate")
def adversary_simulate(body: dict):
    """Run the cheap-vs-costly evasion sweep on one seed fraud. Body: {transaction_id}.
    Returns per-strategy ablation, a cheapest-first detection-decay curve, and a verdict."""
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded.")
    tid = str(body.get("transaction_id", ""))
    if df_all.empty or "transaction_id" not in df_all.columns:
        raise HTTPException(404, "No transaction dataset loaded.")
    match = df_all[df_all["transaction_id"].astype(str) == tid]
    if match.empty:
        raise HTTPException(404, f"transaction_id '{tid}' not found.")
    row = match.iloc[0].to_dict()
    features = compute_features(row)
    result = adversary.simulate(features, ml_score_row)
    result["transaction_id"] = tid
    result["typology"] = str(row.get("fraud_typology", "unknown"))
    result["rail"] = str(row.get("payment_rail", row.get("rail", "card")))
    return result


# ── Closed Feedback Loop ──────────────────────────────────────────────────────
# Analyst dispositions become labeled feedback that online-updates the reputation
# layer (immediate) and queues for retrain (logged). See feedback.py.

@app.post("/feedback")
def post_feedback(body: dict):
    """Record an analyst disposition. Body: {transaction_id, label, recipient_id, source}.
    label: confirm_fraud / clear_false_positive / etc. Returns the online reputation
    update so the caller can see the loop close."""
    if _feedback is None:
        raise HTTPException(503, "Feedback loop not available (reputation layer not loaded).")
    return _feedback.record(
        transaction_id=str(body.get("transaction_id", "")),
        label=str(body.get("label", "")),
        recipient_id=str(body.get("recipient_id", "")),
        source=str(body.get("source", "investigator")),
    )


@app.get("/feedback/status")
def feedback_status():
    """Loop status: labeled totals, online updates applied, retrain queue depth."""
    if _feedback is None:
        return {"loop": "unavailable", "labeled_total": 0}
    return _feedback.status()


# ── Injection Pipeline ────────────────────────────────────────────────────────

def _write_ingest_log(events: list[dict]) -> None:
    """Append scored events to the JSONL ingest log (blocking - run in executor)."""
    try:
        with open(_ingest_log_path, "a") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
    except Exception:
        pass


def _fan_out(event: dict) -> None:
    """Push a scored event to all active SSE subscribers."""
    for q in list(_event_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


@app.post("/ingest")
async def ingest_transaction(body: dict):
    """
    Inject a live transaction into the full RedWing scoring pipeline.

    Runs the complete 4-tier cascade (XGBoost → GNN → graph features → drift)
    then routes the scored event to every live output channel:
      • Drift monitor rolling buffer         (concept drift tracking)
      • Autonomous agent SSE fan-out         (all connected analyst clients)
      • In-memory ingest ring buffer         (feeds /monitor/stream)
      • Append-only JSONL log                (~/pulseml_models/ingest_log.jsonl)

    Accepts raw transaction fields (amount, user_id, device_id, recipient_id,
    payment_rail, …) or pre-computed ML features - or a mix of both.
    Any missing features default to 0.0.
    """
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded - run the ML Fraud Engine notebook first.")

    event = build_event(body)   # drift_monitor.record() already called inside build_event
    event["source"] = "injected"

    _ingest_buffer.appendleft(event)
    _fan_out(event)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_ingest_log, [event])

    return event


@app.post("/ingest/batch")
async def ingest_batch(body: dict):
    """
    Inject multiple transactions in a single call.

    Body: {"transactions": [{...}, {...}, ...]}  (max 1 000 per call)
    Returns: list of scored events + summary stats.
    """
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded - run the ML Fraud Engine notebook first.")

    transactions = body.get("transactions", [])
    if not transactions:
        raise HTTPException(400, "Body must contain a 'transactions' list.")
    if len(transactions) > 1000:
        raise HTTPException(400, "Batch limit is 1 000 transactions per call.")

    results, alerts = [], 0
    for tx in transactions:
        event = build_event(tx)
        event["source"] = "injected"
        _ingest_buffer.appendleft(event)
        _fan_out(event)
        if event["is_alert"]:
            alerts += 1
        results.append(event)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_ingest_log, results)

    return {
        "processed":   len(results),
        "alerts":      alerts,
        "alert_rate":  round(alerts / len(results), 4),
        "results":     results,
    }


@app.get("/ingest/stats")
def ingest_stats():
    """
    Injection pipeline health: buffer occupancy and persistent log size.
    """
    log_lines = 0
    log_bytes = 0
    if _ingest_log_path.exists():
        log_bytes = _ingest_log_path.stat().st_size
        try:
            with open(_ingest_log_path) as f:
                log_lines = sum(1 for _ in f)
        except Exception:
            pass

    return {
        "buffer_used":      len(_ingest_buffer),
        "buffer_capacity":  _ingest_buffer.maxlen,
        "log_transactions": log_lines,
        "log_size_bytes":   log_bytes,
        "log_path":         str(_ingest_log_path),
    }


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
        return {"gaps": [], "count": 0, "message": "No rule gaps found yet - good coverage!"}

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

    # Work with a manageable sample - prioritise fraud rows
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


# ── Drift Monitor ────────────────────────────────────────────────────────────

@app.get("/drift/status")
def get_drift_status():
    """
    ADWIN-style concept drift report.
    Returns PSI on model scores and 5 key features:
      state: warming_up | stable | warning | drift
      score_psi / feature_psi - Population Stability Index values
      drift_events - history of state transitions into warning/drift
    PSI < 0.10: stable · 0.10–0.20: warning · > 0.20: retrain recommended
    """
    return drift_monitor.get_status()


@app.post("/drift/reset")
def reset_drift_monitor():
    """
    Reset the drift monitor after a model retrain.
    Clears all rolling buffers and returns to warming_up state.
    """
    drift_monitor.reset()
    return {"status": "reset", "message": "Drift monitor cleared - warming up again"}


@app.get("/graph/stats")
def get_graph_stats():
    """
    Return graph feature store metadata: entity counts, last refresh time.
    The feature store is the offline precomputed embeddings layer (BRIGHT Tier 3).
    """
    return graph_features.get_stats()


@app.get("/gnn/stats")
def get_gnn_stats():
    """
    Return GNN Tier 2 table coverage: user/device/recipient counts and
    precomputed 1-hop neighbourhood aggregate counts.
    """
    return gnn_lite.get_stats()


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
        raise HTTPException(503, "transactions.csv not found - run the ML notebook first")

    platform     = body.get("platform", "Fintech")
    sophistication = body.get("sophistication", "AI Fraud Agent")
    timeline     = body.get("attack_timeline", [])
    gap_map      = body.get("detection_gap_map", {})
    exp_scores   = body.get("exposure_scores", {})

    bypassed_steps = [s for s in timeline if s.get("outcome") == "BYPASSED"]
    if not bypassed_steps:
        return {"inserted": 0, "message": "No BYPASSED steps found - nothing to ingest"}

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
# API key never touches the browser - stored in operator/.env only.
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


# ── Autonomous Agent Endpoints ────────────────────────────────────────────────

@app.get("/agent/status")
def get_agent_status():
    """Return current state of the autonomous fraud detection agent."""
    uptime_seconds = None
    if agent_state.start_time:
        uptime_seconds = int((datetime.utcnow() - agent_state.start_time).total_seconds())
    return {
        "running":           agent_state.running,
        "uptime_seconds":    uptime_seconds,
        "blocked_count":     agent_state.blocked_count,
        "flagged_count":     agent_state.flagged_count,
        "allowed_count":     agent_state.allowed_count,
        "patterns_learned":  agent_state.patterns_learned,
        "event_buffer_size": len(agent_state.recent_events),
        "case_queue_size":   len(agent_state.case_queue),
        "novel_buffer_size": len(novel_attack_buffer),
    }


@app.get("/agent/events")
async def agent_events_stream():
    """
    SSE fan-out stream of autonomous agent decisions.
    Each connected browser tab gets its own queue (fan-out pattern).
    Backfills the last 20 events immediately on connect.
    """
    my_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _event_subscribers.add(my_queue)

    async def generate():
        try:
            # Backfill
            for event in list(agent_state.recent_events)[:20]:
                yield f"data: {json.dumps(event)}\n\n"
            # Stream live events
            while True:
                try:
                    event = await asyncio.wait_for(my_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        finally:
            _event_subscribers.discard(my_queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent/start")
async def start_agent():
    """Start the autonomous agent if not already running. Idempotent."""
    if agent_state.running:
        return {"status": "already_running"}
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded - run the ML notebook first")
    asyncio.create_task(run_agent(build_event, df_all, FEATURES))
    return {"status": "started"}


@app.post("/agent/stop")
def stop_agent():
    """Gracefully stop the agent. It will finish its current tick then exit."""
    agent_state.running = False
    return {"status": "stopping"}


@app.get("/agent/config")
def get_agent_config():
    """Return the current agent configuration."""
    return agent_config


@app.put("/agent/config")
def update_agent_config(body: dict):
    """
    Update agent config. Changes apply immediately on the next tick - no restart.
    Accepts partial updates; merges deeply with current config.
    """
    try:
        validated = validate_config(body)
    except Exception as e:
        raise HTTPException(400, f"Invalid config: {e}")

    # Mutate the module-level dict in place so run_agent sees the changes
    agent_config.clear()
    agent_config.update(validated)
    save_config(validated)
    return agent_config


@app.get("/agent/cases")
def get_agent_cases(status: str = None):
    """
    Return the case review queue.
    Optional ?status=pending|approved|declined filter.
    """
    cases = list(agent_state.case_queue)
    if status:
        cases = [c for c in cases if c.get("status") == status]
    return cases


@app.post("/agent/cases/{case_id}/resolve")
async def resolve_agent_case(case_id: str, body: dict):
    """
    Analyst resolves a case: approve (confirm agent action) or decline (override).
    body: { action: "approve"|"decline", analyst_id: str, note: str }
    """
    action = body.get("action", "")
    if action not in ("approve", "decline"):
        raise HTTPException(400, "action must be 'approve' or 'decline'")

    # Find case in deque
    found = None
    for case in agent_state.case_queue:
        if case.get("case_id") == case_id:
            found = case
            break
    if not found:
        raise HTTPException(404, f"Case '{case_id}' not found")

    found["status"]         = "approved" if action == "approve" else "declined"
    found["analyst_action"] = action
    found["analyst_id"]     = body.get("analyst_id", "analyst_1")
    found["analyst_note"]   = body.get("note", "")
    found["resolved_at"]    = datetime.utcnow().isoformat() + "Z"

    # Broadcast resolution to SSE clients so Live Feed updates
    resolution_event = {
        "type":          "case_resolved",
        "case_id":       case_id,
        "analyst_action":action,
        "timestamp":     found["resolved_at"],
    }
    from agent import _broadcast
    _broadcast(resolution_event)

    return found


@app.post("/agent/override/{tx_id}")
async def override_agent_decision(tx_id: str, body: dict = {}):
    """
    Human analyst directly overrides a live feed decision by transaction ID.
    body: { action: "allow"|"escalate", analyst_id: str, reason: str }
    """
    override_action = body.get("action", "allow")
    if override_action not in ("allow", "escalate"):
        raise HTTPException(400, "action must be 'allow' or 'escalate'")

    matching = [e for e in agent_state.recent_events if e.get("transaction_id") == tx_id]
    if not matching:
        raise HTTPException(404, f"No recent event for transaction '{tx_id}'")

    override_record = {
        "type":            "human_override",
        "transaction_id":  tx_id,
        "original_action": matching[0].get("action"),
        "override_action": override_action,
        "analyst_id":      body.get("analyst_id", "analyst_1"),
        "reason":          body.get("reason", ""),
        "timestamp":       datetime.utcnow().isoformat() + "Z",
    }
    from agent import _broadcast
    _broadcast(override_record)
    return {"status": "override_recorded", **override_record}


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
        raise HTTPException(400, "connectors list is required - specify which agencies to report to")

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
