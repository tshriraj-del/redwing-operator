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
import hashlib
import math
import json
import hmac
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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

# -- Bootstrap -----------------------------------------------------------------

# Path to the ML backend (pulseml_models / redwing-ml): its trained models AND its
# shared feature foundation (features.py, graph_layer.py) are loaded from here, so the
# operator computes features identically to training. Override for non-default deploys.
MODELS_DIR = Path(os.environ.get("REDWING_MODELS_DIR", Path.home() / "pulseml_models"))

app = FastAPI(title="SyntheticID Operator", version="1.0.0")

# -- API surface protection ----------------------------------------------------
# This process serves 75 endpoints over customer case data, an endpoint that spends real
# Anthropic credit (/rule-factory/run), and live rule deploy/retire controls. None of that
# should be reachable by an unauthenticated caller the moment it has a routable address.

# CORS is an allowlist, not "*". A wildcard lets any page open in the operator's browser
# read every response this API returns. Defaults to the local dev origins, so nothing
# changes on a laptop; set REDWING_ALLOWED_ORIGINS (comma-separated) for a real deploy.
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "REDWING_ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:5179,http://localhost:3000,"
    "http://127.0.0.1:5173,http://127.0.0.1:5179,http://127.0.0.1:3000",
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The fingerprint collector has to be servable to the client it runs in. Mounted best-effort so a
# missing directory cannot stop the API booting - the collector is one surface, not the platform.
try:
    from fastapi.staticfiles import StaticFiles
    _STATIC_DIR = Path(__file__).resolve().parent / "static"
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
except Exception as _e:                                               # noqa: BLE001
    print(f"⚠ static mount unavailable ({type(_e).__name__}); /static/redwing-fp.js will 404")

# Auth is opt-in via REDWING_API_KEY. Set it and every request except /health (and CORS
# preflight) must carry a matching X-API-Key. Leave it unset and the API stays open, which
# is the right default for a laptop demo and is warned about loudly at startup, because it
# is only safe while the socket is bound to localhost.
_API_KEY = os.environ.get("REDWING_API_KEY", "").strip()
_OPEN_PATHS = {"/health"}


@app.middleware("http")
async def _require_api_key(request: Request, call_next):
    # OPTIONS is skipped explicitly rather than relying on middleware ordering, so CORS
    # preflight still succeeds when a key is set.
    if _API_KEY and request.method != "OPTIONS" and request.url.path not in _OPEN_PATHS:
        # compare_digest, not ==, so a wrong key cannot be recovered by timing the response
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), _API_KEY):
            return JSONResponse({"detail": "invalid or missing X-API-Key"}, status_code=401)
    return await call_next(request)

def gate_is_compatible(iso, features) -> bool:
    """Is this novelty artifact trained on the feature set the model currently uses?

    Named and separate so it can be tested against a genuinely stale artifact, rather than only
    asserted about whatever happens to be on disk. The stale case is not hypothetical: the
    isolation forest sitting in the ML repo was trained on 23 features while the supervised
    model had moved to 32, and loading it would have scored a feature space that no longer
    means what it did.
    """
    n = getattr(iso, "n_features_in_", None)
    return n is None or int(n) == len(features)


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

    # THE ML LAYER. Every model that can touch a decision is registered, contract-checked and
    # lifecycle-governed here rather than pickle-loaded ad hoc. Before this there were six
    # scattered loads and `decisions.model_version` was set on 0 of 692 rows, which is workable
    # with two models and not with the five this is heading for.
    from core.model_registry import (REGISTRY, CHAMPION, TIER_1, TIER_2, ModelSpec)
    _xgb_art = MODELS_DIR / ("xgboost_retrained.pkl" if MODEL_TAG == "retrained"
                             else "xgboost.pkl")
    REGISTRY.register(ModelSpec(
        model_id="supervised_scorer", purpose="transaction_risk", tier=TIER_1,
        features=config["features"], state=CHAMPION, artifact=_xgb_art,
        notes="XGBoost over point-in-time features; drives the score directly"))
    REGISTRY.load("supervised_scorer", lambda: xgb, features=config["features"])
    # The NOVELTY GATE. Trained by pulseml_models/anomaly_layer.py, saved to disk, and until
    # now never loaded by the operator: the README described it as 30% of an ML ensemble while
    # nothing in the live path had ever opened the file.
    #
    # It is deliberately NOT a blend. anomaly_layer.py measured that blending the anomaly score
    # into XGBoost DILUTES the supervised model on known fraud, so the design is a gate:
    # XGBoost drives the decision, and this only speaks up to escalate something XGBoost was
    # about to let through. It can raise a score, never lower one.
    ANOMALY = None
    try:
        _acfg = config.get("anomaly") or {}
        _iso = pickle.load(open(MODELS_DIR / "anomaly_iforest.pkl", "rb"))
        _span = (float(_acfg["hi"]) - float(_acfg["lo"])) or 1.0
        REGISTRY.register(ModelSpec(
            model_id="novelty_gate", purpose="novelty", tier=TIER_2,
            features=config["features"], state=CHAMPION,
            artifact=MODELS_DIR / "anomaly_iforest.pkl",
            notes="unsupervised isolation forest; escalate-only, capped at the alert line"))
        REGISTRY.load("novelty_gate", lambda: _iso, features=config["features"])
        if not gate_is_compatible(_iso, config["features"]):
            # A stale gate is worse than none: it was trained on a different feature set, so
            # its notion of "unusual" describes a model that no longer exists. Refusing is the
            # only safe read, and it is loud rather than silent.
            print(f"⚠ novelty gate NOT loaded: trained on {_iso.n_features_in_} features, "
                  f"the current set has {len(config['features'])}. Re-run anomaly_layer.py.")
        else:
            ANOMALY = {"iso": _iso, "hi": float(_acfg["hi"]), "span": _span,
                       "threshold": float(_acfg.get("novelty_threshold", 1.0)),
                       "auc": _acfg.get("standalone_auc")}
            print(f"✓ Novelty gate loaded (standalone AUC {_acfg.get('standalone_auc')}, "
                  f"escalate-only)")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as _e:
        print(f"⚠ novelty gate unavailable ({type(_e).__name__}); "
              f"supervised scoring continues unaffected")

    # THE CARD SCORER. Separate from `supervised_scorer` because a card authorization is a
    # different message: BIN, entry mode, AVS, CVV, 3DS outcome, MCC, token status. The push
    # model looks for velocity, recipient familiarity and session telemetry that an auth does
    # not carry, so it returned 0.0 on EVERY card message.
    #
    # That was not a degraded score, it was a blind one, and it read as "no risk". Measured on
    # the live path before this was wired: a $2,500 e-commerce purchase on a four-day-old
    # account with AVS failure, CVV failure and no 3DS scored IDENTICALLY to a $40 chip grocery
    # purchase, and both were approved 00.
    #
    # `featurise` is imported from the trainer rather than reimplemented here. MODELS_DIR is
    # already on sys.path, and sharing the function is the only way to guarantee the serving
    # features are the training features. A second copy is how training-serving skew starts.
    CARD = None
    try:
        import card_model as _cm
        _card_art = MODELS_DIR / "card_xgb.pkl"
        _card = pickle.load(open(_card_art, "rb"))
        _card_features = list(_card["vectorizer"].feature_names_)
        REGISTRY.register(ModelSpec(
            model_id="card_scorer", purpose="card_authorization_risk", tier=TIER_1,
            features=_card_features, state=CHAMPION, artifact=_card_art,
            notes="XGBoost over the authorization message; drives /authorize. Trained by "
                  "pulseml_models/card_model.py, featurised by that module's own function so "
                  "train and serve cannot diverge."))
        REGISTRY.load("card_scorer", lambda: _card["model"], features=_card_features)
        CARD = {"art": _card, "featurise": _cm.featurise,
                "calibrator": _card.get("calibrator")}
        if not _card.get("calibrator"):
            # Loud, because an uncalibrated score reaching liability.py is worse than a missing
            # one: it is a confident number that is wrong by two orders of magnitude, and it
            # over-blocks silently.
            print("⚠ Card scorer has NO CALIBRATOR: predicted probabilities are inflated by "
                  "scale_pos_weight and must not be used for priced decisions. Re-run "
                  "pulseml_models/card_model.py.")
        print(f"✓ Card scorer loaded ({len(_card_features)} features, "
              f"{'calibrated' if _card.get('calibrator') else 'UNCALIBRATED'}) "
              f"- /authorize can see an authorization")
    except Exception as _e:                                       # noqa: BLE001
        # DELIBERATELY BROAD, and it was not. The first version caught four named exception
        # types; pickle raises AttributeError when an artifact references a class it cannot
        # resolve, which is not in that list, so a card-scorer problem propagated out of the
        # enclosing block and took supervised_scorer and the novelty gate down with it. A model
        # this endpoint owns must never be able to unload the models it does not.
        #
        # Loud, and it must stay loud. An unavailable card model does not mean a card is safe;
        # it means nothing here can judge one, and /authorize says so in its response rather
        # than approving on a 0.0 that looks like a clean score.
        print(f"⚠ CARD SCORER NOT LOADED ({type(_e).__name__}): /authorize cannot score a card "
              f"message. Run pulseml_models/card_model.py.")

    # THE DEVICE GATE. Escalate-only, on the same terms as the novelty gate: it may RAISE a
    # score the model was about to let through and may never lower one.
    #
    # It is a GATE and not a feature, and that was measured rather than assumed. Adding the
    # device-graph columns to the model made it worse on every one (0.9082 -> 0.8432 together,
    # 0.8547 from the per-device rate alone): a per-entity rate on a high-cardinality key gives a
    # tree endless splits in mostly-noise and crowds out the features that generalise. The
    # composite is sharp but fires on 0.013% of traffic, so it cannot move an aggregate metric
    # and only offers a rare column to overfit. See pulseml_models/device_graph.py.
    DEVICE_GRAPH = None
    try:
        import device_graph as _dg
        DEVICE_GRAPH = _dg.DeviceGraph.load(MODELS_DIR / "device_graph.json")
        _b = _dg.bands(DEVICE_GRAPH)
        _thin = next((b for b in _b if b["band"].startswith("<=5")), None)
        print(f"✓ Device gate loaded ({len(DEVICE_GRAPH.table):,} devices"
              + (f", thin-shared band {_thin['vs_solo']}x base rate" if _thin else "")
              + ", escalate-only)")
    except Exception as _e:                                       # noqa: BLE001
        # Advisory, unlike screening: an unavailable device graph means one escalation path is
        # blind, not that traffic goes unscreened, so scoring continues. Stated rather than
        # silent, because a gate that quietly stops firing looks exactly like a gate that has
        # nothing to fire on.
        print(f"⚠ device gate unavailable ({type(_e).__name__}); scoring continues without the "
              f"device escalation. Run pulseml_models/device_graph.py.")

    df_all  = pd.read_csv(MODELS_DIR     / "transactions.csv")
    FEATURES = config["features"]
    MODEL_OK = True
    print(f"✓ Models loaded ({MODEL_TAG}) - {len(df_all):,} transactions available")
    if _API_KEY:
        print("✓ API key required (X-API-Key) on every route except /health")
    else:
        print("⚠ API is UNAUTHENTICATED - safe only while bound to localhost. "
              "Set REDWING_API_KEY before exposing this process.")
    print(f"✓ CORS allowlist: {', '.join(_ALLOWED_ORIGINS)}")
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

# -- Real-data payment model (ULB Credit Card Fraud) - engine-validation anchor --
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

# -- Phase 1 backbone: durable entity/event store ------------------------------
# The platform's shared nervous system (core/store.py). build_event() writes every
# scored transaction here as entities + events, so scoring leaves a durable trail
# instead of an ephemeral dict - the substrate the closed loop (WS2) and the
# cross-institution network (WS3) are built on. A store failure must never break
# scoring, so every write is best-effort.
# Pure helpers (stdlib-only) import at module scope so they exist even if the SQLite
# store fails to open; only the Store() instantiation is guarded.
from core.store import Store, DEFAULT_DB_PATH, eid, FRAUD_TRUE
from core.record import record_scored_event, row_from_backbone
from core.adjudication import schema as adjudication_schema, validate as adjudication_validate
from core.loop import close_loop, record_decision
from core.holdout import holdout_decision, holdout_rationale
from core.telemetry import assess_from_telemetry, derive_signals
from core.ingest_schema import (validate_event, contract as ingest_contract,
                                normalize_rail as _normalize_rail)
from core.stream import DurableQueue, BackpressureError
from core.connectors import FileConnector, DBConnector
from core.webhook import WebhookReceiver
from core.decision_policy import decide as decide_action
from core.screening import screen as screen_payment
from core.liability import expected_liability, price_decision
from core.narrative import scam_narrative
try:
    STORE = Store(DEFAULT_DB_PATH)
    _bb = STORE.stats()
    print(f"✓ Backbone online - {_bb['entities_total']:,} entities, "
          f"{_bb['events_total']:,} events ({STORE.path})")
    if _bb["events_total"] == 0:
        print("  (backbone empty - run `python3 -m core.seed_from_csv` to seed history)")
except Exception as _se:
    STORE = None
    print(f"⚠ Backbone unavailable ({_se}); scoring continues without a durable trail")

# Durable streaming transport: decouples intake (fast, validated, enqueued) from scoring
# (a background consumer drains it), so a slow model or a burst never drops an event.
try:
    TRANSPORT = DurableQueue(MODELS_DIR / "stream.db")
    print(f"✓ Stream transport online ({TRANSPORT.path}); depth {TRANSPORT.stats('ingest')['ready']}")
except Exception as _te:
    TRANSPORT = None
    print(f"⚠ Stream transport unavailable ({_te}); /stream/* disabled")

# Source connector: a pull ingestion source (a JSONL drop file), checkpointed on the backbone
# so it resumes where it left off, publishing into the durable transport.
try:
    FILE_CONNECTOR = (FileConnector("file_drop", TRANSPORT, STORE, MODELS_DIR / "ingest_drop.jsonl")
                      if (TRANSPORT is not None and STORE is not None) else None)
    if FILE_CONNECTOR is not None:
        print(f"✓ Source connector 'file_drop' watching {FILE_CONNECTOR.path} "
              f"(checkpoint {STORE.get_checkpoint('file_drop')})")
except Exception as _ce:
    FILE_CONNECTOR = None
    print(f"⚠ Source connector unavailable ({_ce})")

# Webhook receiver: the real-time PUSH source, authenticated by a per-source HMAC secret so an
# attacker cannot inject fabricated events. Secrets from env REDWING_WEBHOOK_SECRETS (a JSON map
# source -> secret); a demo source is registered so the path is demonstrable out of the box.
try:
    _wh_secrets = json.loads(os.environ.get("REDWING_WEBHOOK_SECRETS", "") or "{}")
    if not isinstance(_wh_secrets, dict):
        _wh_secrets = {}
except Exception:
    _wh_secrets = {}
_wh_secrets.setdefault("demo_processor", "whsec_demo_do_not_use_in_prod")
WEBHOOK = WebhookReceiver(TRANSPORT, _wh_secrets) if TRANSPORT is not None else None
if WEBHOOK is not None:
    print(f"✓ Webhook receiver online; authenticated sources: {WEBHOOK.sources()}")

# -- Injection pipeline state ---------------------------------------------------

_ingest_buffer:   deque = deque(maxlen=500)   # ring buffer - latest injected events
_ingest_log_path: Path  = MODELS_DIR / "ingest_log.jsonl"


# -- Helpers -------------------------------------------------------------------

def compute_features(raw: dict) -> dict:
    """Compute the model's features through the shared foundation (the same transform
    used at training → no training-serving skew). Falls back to raw passthrough only
    if the foundation is unavailable - which is the legacy zero-fill behaviour."""
    if FEATURE_ENGINE is not None:
        return FEATURE_ENGINE.compute(raw)
    return {f: float(raw.get(f, 0.0)) for f in FEATURES}


def ml_score_row(features: dict) -> float:
    """Run XGBoost on a feature dict; returns fraud probability 0-1."""
    if not MODEL_OK or not FEATURES:
        return 0.0
    X = np.array([[float(features.get(f, 0.0)) for f in FEATURES]])
    X_scaled = scaler.transform(X)
    return float(xgb.predict_proba(X_scaled)[0][1])


def _alert_line() -> float:
    """The live alert threshold, read from is_alert's own default rather than restated here.
    A second copy of 0.65 in this file is how the gate and the alert decision drift apart."""
    d = getattr(is_alert, "__defaults__", None)
    return float(d[0]) if d else 0.65


def novelty_view(features: dict) -> dict:
    """The unsupervised second opinion on one payment.

    Returns {available, anomaly, novel} where `anomaly` is the calibrated 0-1 score
    (anomaly_layer.py anchors p50 of train to 0 and p1 to 1) and `novel` says whether it
    crosses the ~99th-percentile threshold, i.e. roughly the most anomalous 1% of traffic.
    """
    if not MODEL_OK or ANOMALY is None or not FEATURES:
        return {"available": False, "anomaly": 0.0, "novel": False}
    try:
        X = scaler.transform(np.array([[float(features.get(f, 0.0)) for f in FEATURES]]))
        raw = float(ANOMALY["iso"].score_samples(X)[0])          # lower = more anomalous
        a = max(0.0, min(1.0, (ANOMALY["hi"] - raw) / ANOMALY["span"]))
        return {"available": True, "anomaly": round(a, 4),
                "novel": a >= ANOMALY["threshold"]}
    except Exception:                                            # noqa: BLE001
        # An unsupervised second opinion must never cost a decision.
        return {"available": False, "anomaly": 0.0, "novel": False}


def apply_novelty_gate(score: float, features: dict) -> tuple:
    """Escalate-only composition of the novelty gate onto a supervised score.

    Returns (score_after, view). The gate can only RAISE a score to the alert line, never lower
    one and never past it into an auto-decline: an unsupervised detector saying "this is
    unusual" is a reason to look, not a reason to be sure. Measured on held-out test, this
    recovers 751 of the 1,681 frauds XGBoost missed, taking catch from 11.2% to 50.9% for 1.04%
    of legitimate traffic sent to review.
    """
    view = novelty_view(features)
    if not view["available"] or not view["novel"]:
        return score, view
    raised = max(float(score), _alert_line())
    view["escalated"] = raised > float(score)
    return raised, view


def resolve_device_id(row: dict, store=None) -> dict:
    """Which device is this, and how much is that answer worth?

    THE FIX FOR THE SEVENTH BUILD-AND-DO-NOT-WIRE. core/fingerprint.py derives a device id from
    validated components behind an entropy floor, and until now that id reached nothing: the
    endpoint returned it to the caller and the gate went on reading `row["device_id"]` straight
    off the request body. So a 126x escalation was scored against an id the client simply named.

    A derived id and a client-asserted id are NOT the same kind of fact and are never collapsed
    into one string here. The derived one has passed component validation, an anchor-entropy
    floor and a high-entropy-anchor requirement. The client-asserted one has passed nothing.
    Both still flow, because the whole existing book is keyed on client-named ids and refusing
    them would switch the gate off for all current traffic, but the SOURCE travels with the id
    so a decision can be audited on it later.

    `contradicts_client` is the cheap extra. If the fingerprint says one device and the body
    claims another, that disagreement is itself a signal: it is what session replay or a shared
    account looks like from the server side.
    """
    row = row or {}
    claimed = str(row.get("device_id") or "")
    subject = str(row.get("transaction_id") or row.get("subject_ref") or "")
    derived = {}
    if store is not None and subject:
        try:
            derived = store.get_device_identity(subject) or {}
        except Exception:                                         # noqa: BLE001
            derived = {}     # identity lookup must never fail a decision
    if derived.get("device_id"):
        return {
            "device_id": derived["device_id"],
            "source": "derived",
            "confidence": derived.get("confidence", ""),
            "anchor_entropy_bits": derived.get("anchor_entropy_bits", 0),
            "contradicts_client": bool(claimed and claimed != derived["device_id"]),
            "client_claimed": claimed,
        }
    return {
        "device_id": claimed,
        "source": "client_asserted" if claimed else "absent",
        "confidence": "",
        "anchor_entropy_bits": 0,
        "contradicts_client": False,
        "client_claimed": claimed,
    }


def apply_device_gate(score: float, row: dict) -> tuple:
    """Escalate-only composition of the device gate onto a score. Returns (score_after, view).

    Shared by every scoring path DELIBERATELY, and defined next to apply_novelty_gate for the
    same reason. This is the control that ADR-001 counts as the sixth instance of a defect class:
    it was built in the ML repo with mutation-verified tests and then called from nowhere, so a
    measured 84x-precision signal changed no outcome. Adding it to one path and not the other
    would have been the seventh.

    The rule the graph encodes: fan-out ALONE is worthless (devices on 8+ accounts measured 0.86x
    the base rate, BELOW population, because a household device has high fan-out and far more
    traffic than a fraud farm). Fan-out with THINNESS is the discriminator, and on held-out
    traffic it fires on 0.013% of rows at 58.8% precision.
    """
    if DEVICE_GRAPH is None:
        return score, {"available": False}
    # Through the RESOLVER, not `row["device_id"]`. That direct read is what made the derived
    # fingerprint reach nothing: the gate escalated on an id the client named.
    ident = resolve_device_id(row, STORE)
    dev = ident["device_id"]
    if not dev:
        return score, {"available": True, "fired": False, "why": "no device on this event",
                       "identity_source": ident["source"]}
    try:
        import device_graph as _dg
        raised, view = _dg.gate(DEVICE_GRAPH, dev, float(score))
    except Exception as e:                                        # noqa: BLE001
        return score, {"available": False, "error": type(e).__name__}
    view["available"] = True
    view["escalated"] = raised > float(score)
    # The SOURCE of the id travels with the verdict. An escalation scored against a derived
    # identity and one scored against a string the client supplied are different evidence, and
    # an analyst reviewing a fired gate has to be able to tell them apart.
    view["identity_source"] = ident["source"]
    view["identity_confidence"] = ident["confidence"]
    if ident["contradicts_client"]:
        view["contradicts_client_device"] = True
        view["client_claimed"] = ident["client_claimed"]
    return raised, view


def _network_view(row: dict, local_score: float):
    """Authorization IQ applied to one payment: returns (pack_or_None, score_after_network).

    Shared by every scoring path so the network layer cannot drift between them.

    Two rules, both deliberate:
      ESCALATE ONLY. The network may raise a score the local book had no reason to raise -
      that is the entire reveal - but a clean network view must never talk the local model DOWN
      off a signal it found in its own data. max(), not a blend.
      EVIDENCE FLOOR. Below the consortium's floor the combined rate is noise, so the network
      stays silent rather than moving a real decision on it.
    Returns the local score unchanged if the index is still warming, so scoring never blocks.
    """
    if _aiq_index is None or not row.get("recipient_id"):
        return None, local_score
    try:
        from core.authorization_iq import authorize as _aiq_authorize
        rid = str(row["recipient_id"])
        pack = _aiq_authorize(
            {"recipient": rid if rid.startswith("recipient:") else f"recipient:{rid}",
             "sender": row.get("user_id"), "amount": row.get("amount", 0.0),
             "rail": row.get("payment_rail", "card")},
            _aiq_index)
        # ONE definition of the rule, in core/authorization_iq. It used to be inlined here
        # and copied into the tests, which is how a guarantee ends up with two
        # implementations and no audit of either.
        from core.authorization_iq import apply_escalate_only
        return pack, apply_escalate_only(local_score, pack)
    except Exception:
        return None, local_score      # the network layer must never fail a score


def _account_age(row) -> float:
    """Account age in days, defaulting to established. Defaulting the OTHER way would put every
    row with a missing field into the new-account tier and apply its harder floor to customers
    who have been here for years."""
    try:
        return max(0.0, float(row.get("account_age_days", 365)))
    except (TypeError, ValueError):
        return 365.0


def score_card_message(msg: dict) -> tuple:
    """THE CARD MODEL, not the push model. `(p_fraud, detail)`.

    Extracted rather than left inline in /authorize's scorer, because build_event() needed the
    identical logic: /ingest, /ingest/batch, the stream consumer and every source connector all
    funnel a card-rail row through build_event(), and it was still scoring every one of them with
    compute_features + ml_score_row, the PUSH-payment path. /authorize got fixed; this, the far
    busier entry point, did not, so the same $2,500-scores-like-$40 blindness that /authorize had
    for its whole life was still live on every other ingestion surface. One function now, so a
    third entry point cannot repeat it a third time.
    """
    if CARD is None:
        # NOT 0.0. A missing model is not a clean score, and the difference has to survive into
        # the response or the next reader draws the same wrong conclusion this endpoint invited
        # for the whole time it was mis-wired.
        return 0.0, {"scored": False, "model_unavailable": True,
                     "warning": "no card scorer loaded; this event was NOT risk-assessed and "
                                "the score below is not a risk opinion"}
    try:
        from core.card_message import normalise, quality
        art = CARD["art"]
        norm = normalise(msg)
        feats = CARD["featurise"](norm["row"], art["aggregates"])
        X = art["vectorizer"].transform([feats])
        p_raw = float(art["model"].predict_proba(X)[0][1])
        # CALIBRATE. The raw XGBoost output is not a probability: scale_pos_weight is what makes
        # a 0.26% base rate learnable and it inflates predicted p by 77x. Ranking is unaffected,
        # which is why AUC and the frontier never showed it, but core/liability.py computes
        # expected_liability = p x amount x rate, so an uncalibrated p over-blocks every card
        # decision by two orders of magnitude.
        cal = CARD.get("calibrator") or {}
        if cal.get("a") is not None:
            _z = math.log(min(max(p_raw, 1e-9), 1 - 1e-9) / (1 - min(max(p_raw, 1e-9), 1 - 1e-9)))
            p = 1.0 / (1.0 + math.exp(-(float(cal["a"]) * _z + float(cal["b"]))))
        else:
            p = p_raw
        return p, {"scored": True, "model": "card_scorer",
                   "version": REGISTRY.version_of("card_scorer"),
                   "ml": round(p, 6), "ml_raw": round(p_raw, 4),
                   "calibrated": bool(cal.get("a") is not None),
                   # What the message was missing travels WITH the score. A number computed
                   # without AVS, CVV and 3DS is reading three of the model's top five features
                   # as "not provided", and presenting it as the same number is how a
                   # data-quality problem becomes a risk decision.
                   "message_quality": quality(norm)}
    except Exception as e:                                        # noqa: BLE001
        # A scorer failure must not stop a decision. But it is reported as a failure, never as a
        # zero risk.
        return 0.0, {"scored": False, "error": type(e).__name__,
                     "warning": "card scorer raised; this event was NOT risk-assessed"}


def build_event(row) -> dict:
    """Score a row and return a full event payload for SSE or REST."""
    if isinstance(row, pd.Series):
        row = row.to_dict()

    # SCREENING FIRST. Not a risk input: a payment to a designated party cannot be approved at
    # any score, under any posture, past any policy ceiling. There is nothing to weigh it
    # against, so it runs before there is a score to weigh. It fails CLOSED, unlike every
    # advisory layer in this file, because approving unscreened traffic is a different class of
    # mistake from scoring without a novelty view.
    scr = screen_payment(counterparty=str(row.get("recipient_name", "")
                                          or row.get("recipient_id", "")),
                         member=str(row.get("user_name", "") or ""))

    features = compute_features(row)
    # CARD ROWS GO THROUGH THE CARD MODEL, not the push-payment score below. features/matches
    # still get computed either way, because the rule matcher and the graph/drift plumbing
    # further down expect them to exist; only `ml`, the number everything downstream is DERIVED
    # from (c_score, cascade_score, network_score, the priced decision), is replaced.
    #
    # This was the actual gap /authorize's fix did not close: every OTHER ingestion surface
    # (/ingest, /ingest/batch, the stream consumer, every source connector) calls build_event(),
    # and until this branch existed all of them still scored a card authorization with a model
    # that looks for velocity and recipient familiarity a card message does not carry - the
    # identical blindness /authorize had for its whole life, just on a busier door.
    # Canonicalise rather than compare to a literal. This line read
    # `str(row.get("payment_rail") or "").strip().lower() == "card"`, which meant
    # `debit_card` and `credit_card` - both already declared synonyms in
    # ingest_schema.RAILS - silently routed card traffic back to the PUSH model, exactly the
    # blindness this branch exists to remove. Callers that reach build_event() without going
    # through validate_event() (/alerts, _assemble_case, replay) never had the rail normalised
    # for them, so they were the ones losing the card model.
    _is_card = _normalize_rail(row.get("payment_rail"))[0] == "card"
    card_detail = None
    if _is_card:
        # The GATED scorer, which is the same function /authorize uses. Calling the ungated
        # score_card_message here is exactly how a control ends up on one door and not the other.
        ml, card_detail = score_card_message_gated(row)
    else:
        ml = ml_score_row(features)
    matches = score_transaction(features)
    top = matches[0] if matches else None

    c_score = combined_score(ml, top["confidence"]) if top else ml

    # -- Tier 2: GNN cascade (borderline transactions only) --------------------
    gnn_result = None
    if gnn_lite.should_invoke(c_score):
        gnn_result = gnn_lite.score(
            row.get("user_id"), row.get("device_id"), row.get("recipient_id")
        )
        cascade_score = gnn_lite.cascade_blend(c_score, gnn_result)
    else:
        cascade_score = c_score

    # -- Tier 4: Authorization IQ - the CROSS-INSTITUTION network view ---------
    # The tiers above are all computed from this institution's own book. On a push rail that is
    # structurally insufficient: the victim's bank sees an ordinary outgoing payment and the
    # mule's bank sees an ordinary inbound, so the mule is only visible in the aggregate. This
    # adds the consortium's view of the payee at authorization time.
    #
    # It ESCALATES, never de-escalates: the network can raise a score the local book had no
    # reason to raise (that is the whole reveal), but a clean network view must never talk the
    # local model DOWN off a signal it found in its own data. And the contribution is reported
    # separately rather than silently folded in, so an analyst can always see how much of a
    # score came from outside their own institution.
    aiq, network_score = _network_view(row, cascade_score)

    # The novelty gate, applied AFTER the network view and on the same escalate-only terms.
    # Order matters and is deliberate: the supervised model and the consortium both speak
    # first, and the unsupervised detector only gets to raise what those two were about to let
    # through. That keeps "this is unusual" in its proper place, a reason to look rather than a
    # reason to be sure.
    network_score, novelty = apply_novelty_gate(network_score, features)

    # The device gate, on the same escalate-only terms and applied AFTER the novelty gate. Order
    # is deliberate and matches the rest of the ladder: the supervised model, the consortium and
    # the unsupervised detector all speak first, and the device view only raises what those were
    # about to let through.
    network_score, device_view = apply_device_gate(network_score, row)

    # bool() on a CSV field is a trap: bool("False") is True, because a non-empty string is
    # truthy. Scoring a CSV-sourced row therefore flagged EVERY non-fraud transaction as an
    # alert. Measured over 3,000 replayed payments: the model's own call fired on 0.07% and
    # this line reported 98.33%, and essentially all of that gap was the string "False".
    _known_fraud = str(row.get("is_fraud", "")).strip().lower() in ("1", "true", "yes")
    alert = is_alert(network_score) or _known_fraud

    # -- Tier 3: offline graph context (O(1) lookup) ---------------------------
    graph_ctx = graph_features.get_features(
        user_id      = row.get("user_id"),
        device_id    = row.get("device_id"),
        recipient_id = row.get("recipient_id"),
    )

    # -- Drift monitoring - non-blocking, appends to rolling buffer ------------
    drift_monitor.record(ml, features)

    event = {
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
        "local_score":       round(cascade_score, 4),      # this institution's own book alone
        "combined_score":    round(network_score, 4),      # after the network view
        # Authorization IQ, reported separately so the network's contribution is never hidden
        # inside the headline number. network_lift > 0 means the consortium raised this score.
        "network_risk":      round(float(aiq["network_risk"]), 4) if aiq else None,
        "network_lift":      round(network_score - cascade_score, 4),
        "network_reveal":    bool(aiq.get("network_reveal")) if aiq else False,
        "network_codes":     [c["code"] for c in aiq["reason_codes"]] if aiq else [],
        "is_alert":          alert,
        # Reported separately, never folded silently into the score, for the same
        # reason the network contribution is: an analyst must be able to see that a
        # payment was raised by an unsupervised detector rather than by the model.
        "novelty":           novelty,
        # Reported separately for the same reason novelty and the network view are: an analyst
        # must be able to see that a payment was raised by the device layer rather than by the
        # model, and the gate fires rarely enough that every firing will be looked at.
        "device_gate":       device_view,
        "matched_signals":   top["matched_signals"] if top else [],
        "graph_context":     graph_ctx,
        "graph_risk_score":  graph_ctx["graph_risk_score"],
        "timestamp":         datetime.utcnow().isoformat() + "Z",
    }
    if card_detail is not None:
        # Rides alongside the score for the same reason it does on /authorize: a number
        # computed on a degraded card message is not the same number as one computed on a
        # complete one, and the caller must be able to see the difference rather than trust a
        # score that looks identical either way.
        event["card_score_detail"] = card_detail

    # WS4: price the decision in dollars of expected reimbursement liability, not just
    # probability - the number the buyer's P&L actually cares about post-regulation.
    #
    # Typology is the PREDICTED pattern, never row["fraud_typology"]. That column is the
    # dataset's adjudicated label: it exists in replay and does not exist at decision time, so
    # reading it here made the liability book quietly depend on knowing the answer. Measured on
    # 200K rows the distortion was -2.1%, small but the wrong KIND of number: it is right for a
    # reason production cannot reproduce, and it would silently change the day this ran on live
    # traffic. The predicted pattern is what a real deployment actually has.
    #
    # One variable, used by BOTH sides of the price below, so the two can never drift onto
    # different notions of what this transaction is.
    _pred_typology = str(event.get("top_pattern_id") or "")

    event["expected_liability"] = expected_liability(
        cascade_score, event["amount"], typology=_pred_typology, rail=event["rail"])

    # WS10: price the OTHER side too. Expected liability alone is one-sided, and a one-sided
    # objective always over-blocks, because the fraud loss lands on this team's P&L and the
    # wrongly-declined customer lands on someone else's. Measured at this platform's own
    # operating point that bias runs 2.6 false positives per real fraud caught.
    #
    # LTV band is proxied from the customer's own average spend, because this dataset carries
    # no CRM value band. That is a stand-in for a real retention model and is labelled as such
    # in the payload rather than presented as measured.
    try:
        _avg = float(row.get("user_avg", 0.0) or 0.0)
    except (TypeError, ValueError):
        _avg = 0.0
    _band = "high" if _avg >= 500 else "medium" if _avg >= 100 else "low"

    event["decision_economics"] = price_decision(
        cascade_score, event["amount"], typology=_pred_typology, rail=event["rail"],
        action=("HOLD" if alert else "ALLOW"), ltv_band=_band,
        account_age_days=row.get("account_age_days", 365))
    event["decision_economics"]["ltv_band_source"] = "proxied from user_avg spend, not a CRM value band"
    event["decision_economics"]["typology_source"] = "predicted pattern (serving-time available)"

    # Monitored-holdout policy: a capped, low-liability slice of would-be-holds is released and
    # observed, giving clean counterfactual ground truth. Decided BEFORE the durable trail is
    # written, so the backbone records the action actually enforced. (Deciding it after meant a
    # released case was still logged as an alert, and liability_at_risk counted it.)
    # The PRICED action is what the money supports; the POLICY bounds it. Until now these eight
    # actions collapsed into `"HOLD" if is_alert else "ALLOW"`, so a $12 card payment and a
    # $40,000 push to a three-day-old payee resolved identically. The policy applies the
    # institution's floor and ceiling per rail and risk band, and stamps the version of the
    # table that produced the decision so an outcome change can later be attributed to the
    # policy change that caused it.
    # A screening block is terminal and outranks the priced action entirely. Putting it into
    # the policy floor instead would let a ceiling soften it, and no institutional appetite
    # setting may soften a sanctions block.
    if scr.get("blocked"):
        event["screening"] = scr
        event["is_alert"] = True
        event["policy"] = {"action": "BLOCK", "priced_action": "BLOCK", "band": "screening",
                           "bounded_by": "screening", "policy_escalated": True,
                           "policy_deescalated": False, "rule": {"why": scr["reason"]},
                           "terminal": True, "screening_code": scr["code"]}
        return event
    event["screening"] = scr

    _priced = event["decision_economics"].get("recommended_action") or (
        "HOLD" if event["is_alert"] else "ALLOW")
    _tier = "new_account" if _account_age(row) < 30 else ""
    # Banded on network_score, the score that ACTUALLY drove the alert, not on cascade_score.
    # cascade_score is taken before the consortium view and the novelty gate, so banding on it
    # meant a payment those two had escalated to the alert line still got the low-band floor:
    # the escalation reached the alert boolean and never reached the policy.
    event["policy"] = decide_action(
        _priced, network_score, rail=event["rail"], direction="outbound", tier=_tier)

    ho = None
    if STORE is not None:
        try:
            proposed = event["policy"]["action"]
            ho = holdout_decision(event["transaction_id"], proposed, event["expected_liability"])
            if ho["release"]:
                event["is_alert"] = False          # actually let through, and monitored
                event["monitored"] = True
            event["holdout"] = ho["holdout"]
        except Exception:
            ho = None

    # Durable trail on the backbone (best-effort; never changes the score above).
    record_scored_event(STORE, event, row)

    # Training substrate: log this scoring as a POINT-IN-TIME decision (the feature snapshot
    # as it was used). Additive and best-effort; a failure here must never fail scoring.
    if STORE is not None and ho is not None:
        try:
            tid = event["transaction_id"]
            did = f"dec:{tid}" if tid else None
            record_decision(
                STORE, subject_ref=tid,
                entity_id=eid("user", str(row.get("user_id", "unknown"))),
                action=ho["enforced_action"], module="model",
                score=cascade_score, expected_liability=event["expected_liability"],
                features=features,
                rationale={**holdout_rationale(ho), "pattern": event.get("top_pattern")},
                shadow=False, institution_id=str(row.get("institution_id", "") or ""),
                # `config["version"]` does not exist in model_config.json, so this wrote
                # "" on every one of 692 decisions. The registry's stamp is a content hash of
                # the champion artifacts actually loaded, so it cannot disagree with what scored.
                model_version=REGISTRY.decision_versions(),
                # Same dead column, same fix: decision_policy already computes this and
                # record_decision was never handed it.
                policy_version=str((event.get("policy") or {}).get("policy_version", "")),
                decision_id=did,
            )

            # The scorer's OWN call, recorded so an analyst's adjudication has something to
            # pair against. Measured before this existed: 0 subjects in the whole substrate
            # carried both a machine prediction and a human label, so no amount of
            # adjudication could ever fill the gate. The only target that wrote a machine
            # label was intent.motive, which needs telemetry and so covered 6% of decisions;
            # whatever an analyst happened to pick, there was almost never a prediction
            # sitting there. This covers ~100%.
            #
            # READ IT AS AGREEMENT, NOT GRADUATION. For the intent targets the machine side is
            # a genuine expert rule and "should this become a model" is the real question. Here
            # the machine side is ALREADY the model, so pairing it with human labels measures
            # whether the model and the analysts agree. That is worth knowing on its own terms,
            # and kappa on it is a drift signal, but it is not the question graduation.py was
            # written to answer and should not be quoted as if it were.
            # Derived from the SCORE, not from the action, and deliberately not from
            # event["is_alert"]. Two traps, both of which this first got wrong:
            #
            #   1. the enforced action is not a fraud call. HOLD means "worth a look" and is
            #      two thirds of all decisions, against a 0.65% base rate, so recording
            #      HOLD -> is_fraud=1 would assert fraud on most of the book.
            #   2. event["is_alert"] is `is_alert(network_score) OR row["is_fraud"]`, so it
            #      carries the ground-truth label. Storing that as a "prediction" would write
            #      the answer into the substrate wearing a model's name, and the agreement
            #      measured against a human label later would be the label against itself.
            #
            # is_alert(network_score) alone is the model's own call, with nothing borrowed.
            try:
                model_call = 1 if is_alert(network_score) else 0
                STORE.add_label(
                    "outcome", "is_fraud", model_call,
                    source="heuristic", confidence=round(float(network_score or 0.0) / 100.0, 4),
                    decision_id=did, subject_ref=tid,
                    entity_id=eid("user", str(row.get("user_id", "unknown"))),
                    annotator="model_score_call")
            except Exception:
                pass   # a training label is never worth failing a score for

            # Real-telemetry actor read: ONLY when the client reported behaviour for this
            # subject. Derived from reported telemetry (never the typology), so the motive
            # heuristic label it bootstraps is honest. No telemetry -> the actor layer stays
            # silent, which is correct: motive cannot be inferred from an amount and a rail.
            tel = STORE.get_telemetry(tid)
            if tel:
                actor = assess_from_telemetry(tel)
                if actor.get("actor"):
                    m = actor["actor"]["motive"]["motive"]
                    event["actor_motive"] = m
                    event["actor_victim_stage"] = actor["victim"]["arc"]["stage_label"]
                    # only bootstrap a training label from a confident read, never from an
                    # inconclusive one (that would teach the model noise)
                    if m != "inconclusive":
                        STORE.add_label("intent", "motive", m, source="heuristic", confidence=0.3,
                                        decision_id=did, subject_ref=tid,
                                        entity_id=eid("user", str(row.get("user_id", "unknown"))),
                                        annotator="motive_from_telemetry")
        except Exception:
            pass   # the substrate is additive; a failure here must never fail scoring

    # NO witting-ness label is written here, and that is a decision rather than an omission.
    #
    # The graduation gate cannot fire for intent.witting_role because paired_with_heuristic is 0
    # and nothing here can raise it. The obvious fix, run the witting-ness heuristic during
    # scoring, was built and MEASURED, and it does not work: its twelve tells and the session
    # telemetry the scorer has share zero fields, and of the tells derivable from the ledger and
    # the store, every one points at guilt. Wired in, it labelled 99% of accounts "witting";
    # after correcting the worst of it, 68%, which turned out to be a fact about the decisions
    # table being 66% HOLD rather than anything about mules.
    #
    # Auto-recording those as training labels would launder our own data shape into the ground
    # truth a classifier is later trained on. core/mule_behaviour.py keeps the derivation and
    # refuses to return a verdict from one-directional evidence; witting-ness stays an
    # investigator's judgment, captured through the adjudication panel, until the substrate
    # carries the tells that only a person can observe.

    return event


# -- Autonomous Agent Startup --------------------------------------------------

@app.on_event("startup")
async def start_autonomous_agent():
    """Start the SyntheticID agent and schedule hourly graph feature refresh."""
    if MODEL_OK and not agent_state.running:
        asyncio.create_task(run_agent(build_event, df_all, FEATURES))
    asyncio.create_task(_graph_refresh_loop())
    if TRANSPORT is not None:
        asyncio.create_task(_stream_consumer_loop())        # drain the durable queue into scoring
    if FILE_CONNECTOR is not None:
        asyncio.create_task(_connector_poll_loop())         # auto-ingest source (file) drops
    # WS3/WS5: warm the consortium index and the fraud graph off the event loop so the
    # first lookups are instant
    if STORE is not None:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _get_consortium_index)
        loop.run_in_executor(None, lambda: backbone_graph(refresh=True))
        loop.run_in_executor(None, _get_aiq_index)          # Authorization IQ network index


async def _graph_refresh_loop() -> None:
    """Refresh graph features every hour so the ring-detection context stays current."""
    while True:
        await asyncio.sleep(3600)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, graph_features.refresh_from_disk, MODELS_DIR)
        await loop.run_in_executor(None, gnn_lite.refresh_from_disk, MODELS_DIR)


# -- Routes --------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_OK else "degraded",
        "models_loaded": MODEL_OK,
        "transaction_count": len(df_all),
        "features": FEATURES,
        "model_metrics": config.get("metrics", {}) if MODEL_OK else {},
        "patterns": len(PATTERNS),
        # The unsupervised second opinion's status and its measured contribution. Reported
        # here so "is the gate actually on?" is answerable without reading the boot log, which
        # is how it stayed unwired and unnoticed for as long as it did.
        "novelty_gate": ({
            "loaded": True,
            "standalone_auc": (config.get("anomaly") or {}).get("standalone_auc"),
            "threshold": (config.get("anomaly") or {}).get("novelty_threshold"),
            "composition": "escalate-only, capped at the alert line",
        } if ANOMALY else {"loaded": False,
                           "why": "artifact missing, stale, or unreadable; supervised "
                                  "scoring is unaffected"}),
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

    # Screening first here too. This is the THIRD control to need wiring on both paths
    # (novelty gate, decision policy, now screening), which is a design smell rather than three
    # separate oversights: build_event() and /score are two hand-maintained copies of one
    # pipeline. See the note in the PR.
    scr = screen_payment(counterparty=str(body.get("recipient_name", "")
                                          or body.get("recipient_id", "")),
                         member=str(body.get("user_name", "") or ""))

    features = compute_features(body)
    ml       = ml_score_row(features)
    matches  = score_transaction(features)
    top      = matches[0] if matches else None
    c_score  = combined_score(ml, top["confidence"]) if top else ml

    # Tier 4: the cross-institution network view (see _network_view). Escalate-only, and
    # reported separately below so the network's contribution is never hidden in the headline.
    aiq, net_score = _network_view(body, c_score)

    # Same gate, same order, same terms as build_event(). Applied here too because /score is a
    # second decision path, and a control that lives on only one of them is a control that
    # disagrees with itself depending on how the payment arrived.
    net_score, novelty = apply_novelty_gate(net_score, features)
    # Same gate, same order, same terms as build_event(). On BOTH paths from the first commit,
    # because a control that lives on only one of them is a control that disagrees with itself
    # depending on how the payment arrived - which is the whole subject of ADR-001.
    net_score, device_view = apply_device_gate(net_score, body)

    transaction_id = str(body.get("transaction_id", f"txn_{uuid.uuid4().hex[:8]}"))
    explanation = _xai_explain(
        features       = features,
        ml_score       = ml,
        pattern_match  = top,
        combined_score = c_score,
        model          = xgb,
        scaler         = scaler,
        feature_names  = FEATURES,
        # The REGISTRY, not config. `config["version"]` does not exist and never has, so this
        # read fell through to the "1.0.0" default and stamped a constant that tracks nothing:
        # an explanation attributed to "1.0.0" cannot be matched to the artifact that produced
        # it. Measured on the live store, 395 of 715 decisions (55.2%) carry no usable model
        # version. The registry returns the content hash of every model that could affect this
        # decision, which is the thing an outcome is later attributed to.
        model_version  = REGISTRY.decision_versions(),
        transaction_id = transaction_id,
    )

    alert = is_alert(net_score)

    if scr.get("blocked"):
        return {"transaction_id": str(body.get("transaction_id", "")),
                "screening": scr, "is_alert": True,
                "policy": {"action": "BLOCK", "bounded_by": "screening", "terminal": True,
                           "screening_code": scr["code"],
                           "rule": {"why": scr["reason"]}},
                "local_score": None, "note": "screened before scoring; no risk score computed"}

    # Same policy, same terms, on this path too. /score is a second decision path and a policy
    # that lives on only one of them is a policy that disagrees with itself depending on how the
    # payment arrived, which is exactly the drift _network_view was centralised to prevent.
    _sc_priced = price_decision(
        net_score, float(body.get("amount", 0) or 0), typology=(top or {}).get("name", ""),
        rail=str(body.get("payment_rail", "")), action=("HOLD" if alert else "ALLOW"),
        account_age_days=body.get("account_age_days", 365)).get("recommended_action") or (
            "HOLD" if alert else "ALLOW")
    policy = decide_action(
        _sc_priced, net_score, rail=str(body.get("payment_rail", "")), direction="outbound",
        tier=("new_account" if _account_age(body) < 30 else ""))

    # Training substrate: this endpoint is the what-if / sandbox path (the UI scores ad-hoc
    # transactions through it), so nothing here is enforced against a real customer. It is
    # still recorded, as a SHADOW decision: the point-in-time feature snapshot is exactly what
    # makes a scoring trainable later, and dropping it silently was leaving the ledger blind to
    # every score the UI produced. shadow=1 keeps these out of the enforced counts and out of
    # the holdout/liability accounting, where a counterfactual does not belong.
    #
    # No stable decision_id on purpose: log_decision is INSERT OR REPLACE, and reusing the
    # enforced `dec:{tid}` key would let a sandbox re-score overwrite a real enforced decision.
    # Each what-if is its own immutable row.
    if STORE is not None:
        try:
            record_decision(
                STORE, subject_ref=transaction_id,
                entity_id=eid("user", str(body.get("user_id", "unknown"))),
                action=("HOLD" if alert else "ALLOW"), module="model",
                score=c_score, features=features,
                rationale={"pattern": top, "path": "score_endpoint", "enforced": False},
                shadow=True,
                institution_id=str(body.get("institution_id", "") or ""),
                # `config["version"]` does not exist in model_config.json, so this wrote
                # "" on every one of 692 decisions. The registry's stamp is a content hash of
                # the champion artifacts actually loaded, so it cannot disagree with what scored.
                model_version=REGISTRY.decision_versions(),
            )
        except Exception:
            pass          # a substrate failure must never fail scoring

    return {
        "transaction_id": transaction_id,
        "ml_score":       round(ml, 4),
        "local_score":    round(c_score, 4),        # this institution's own book alone
        "combined_score": round(net_score, 4),      # after the cross-institution network view
        # Authorization IQ, reported separately: network_lift > 0 means the consortium raised
        # this score above anything the local book could justify on its own.
        "network_risk":   round(float(aiq["network_risk"]), 4) if aiq else None,
        "network_lift":   round(net_score - c_score, 4),
        "network_reveal": bool(aiq.get("network_reveal")) if aiq else False,
        # The unsupervised second opinion, reported beside the score rather than folded into
        # it, on the same principle as network_reveal above: an analyst must be able to see
        # that a payment reached them because it was UNUSUAL, not because the model was
        # confident. Those two justify very different next steps.
        "novelty":        novelty,
        "device_gate":    device_view,
        "screening":      scr,
        # The priced action, the policy that bounded it, and the version of the
        # table that did the bounding. Reported together because a decision you
        # cannot attribute to a policy is one you cannot defend later.
        "policy":         policy,
        "network_codes":  [c["code"] for c in aiq["reason_codes"]] if aiq else [],
        "is_alert":       alert,
        "top_pattern":    top,
        "all_patterns":   matches,
        "explanation":    explanation,
    }


# -- XAI / Explainability Endpoints -------------------------------------------

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
      min_score      filter by minimum combined score (0.0-1.0)
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


# -- Investigator Case File ----------------------------------------------------

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
            # Registry, not config. See the note on the same call in `score`: the config key
            # does not exist, so this stamped a constant "1.0.0" on every case-file explanation.
            model_version=REGISTRY.decision_versions(),
            transaction_id=str(row.get("transaction_id", "")),
        )
    except Exception:
        explanation = None

    case = case_file.assemble(row, scored, graph_ctx=graph_ctx, explanation=explanation)

    # WS4: a plain-language read of the con (deterministic; the copilot can enrich it).
    #
    # Same leak, and qualitatively the worse one: written from the PREDICTED pattern, not the
    # adjudicated label. Narrating the con from ground truth made the copilot look like it had
    # inferred the scam type when it had simply been told, and that flattery would evaporate in
    # production, where the column is absent and the narrative falls back to generic.
    case["scam_narrative"] = scam_narrative(
        typology=str(scored.get("top_pattern_id") or ""),
        signals={"amount": scored.get("amount", row.get("amount", 0.0)),
                 "rail": scored.get("rail", row.get("payment_rail", "")),
                 "is_new_recipient": row.get("is_new_recipient"),
                 "expected_liability": scored.get("expected_liability")})
    case["expected_liability"] = scored.get("expected_liability")   # WS4: priced in dollars

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
            # predicted, not adjudicated: enrichment signals derived from the true typology
            # are coherent for a reason no live connector could reproduce
            fraud_typology=str(scored.get("top_pattern_id") or ""),
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

    if not df_all.empty and "transaction_id" in df_all.columns:
        match = df_all[df_all["transaction_id"].astype(str) == str(transaction_id)]
        if not match.empty:
            return _assemble_case(match.iloc[0].to_dict())

    # Not in the historical dataset. Everything the ingestion pipeline brings in (a file drop,
    # a webhook push, a polled source table, /ingest, /stream/publish) is scored and persisted
    # to the backbone but never enters that dataset, so fall back to it. Without this the
    # analyst can open historical rows but nothing that actually flowed through the pipeline.
    row = row_from_backbone(STORE, transaction_id)
    if row is not None:
        return _assemble_case(row)
    raise HTTPException(404, f"transaction_id '{transaction_id}' not found.")


@app.post("/case")
def post_case(body: dict):
    """Assemble a case file from an ad-hoc transaction payload (e.g. an injected or
    streamed transaction not in the historical dataset)."""
    if not MODEL_OK:
        raise HTTPException(503, "ML models not loaded.")
    return _assemble_case(body)


# -- Agent-Evaluation Environment ----------------------------------------------
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


@app.post("/env/agent")
def env_agent(body: dict):
    """Run the LLM investigator on one case, graded by the environment's own verifiers.

    Body: {transaction_id, compare?}. With `compare: true` it also runs the three reference
    policies on the same case and ranks them together, which is the honest presentation: the
    `investigator` policy is a strong hand-written baseline, and an agent that does not beat it
    should be seen not beating it.

    Without an ANTHROPIC_API_KEY this returns available=false and says so. It does not fall
    back to a scripted policy and report the result as an agent run.
    """
    from core.investigator_agent import compare, run_episode
    case = _case_for_env(str(body.get("transaction_id", "")))
    if body.get("compare"):
        return compare(case)
    return run_episode(case)


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


# -- Adversary Simulator -------------------------------------------------------
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


# -- Closed Feedback Loop ------------------------------------------------------
# Analyst dispositions become labeled feedback that online-updates the reputation
# layer (immediate) and queues for retrain (logged). See feedback.py.

@app.post("/feedback")
def post_feedback(body: dict):
    """Record an analyst disposition. Body: {transaction_id, label, recipient_id, source}.
    label: confirm_fraud / clear_false_positive / etc. Returns the online reputation
    update so the caller can see the loop close."""
    if _feedback is None:
        raise HTTPException(503, "Feedback loop not available (reputation layer not loaded).")

    transaction_id = str(body.get("transaction_id", ""))
    label          = str(body.get("label", ""))
    recipient_id   = str(body.get("recipient_id", ""))
    source         = str(body.get("source", "investigator"))

    # Existing loop: online reputation update (next payment scores higher) + log.
    result = _feedback.record(transaction_id, label, recipient_id, source)

    # WS2: mirror the disposition onto the durable backbone and return the receipt
    # that makes the compounding visible (pending payments, exposure, retrain queue).
    if STORE is not None:
        try:
            online   = result.get("online_reputation_update") or {}
            rep_rate = online.get("recipient_global_fraud_rate") if online else None
            lc       = result.get("label_class")
            is_fraud = True if lc == "fraud" else (False if lc == "legit" else None)

            # Intent is the analyst's adjudication and becomes GOLD in the label store, so it
            # is validated against the vocabularies the heuristics actually emit. An
            # out-of-vocabulary gold label is worse than none: the gate would count it toward
            # readiness while being unable to compare it to anything.
            raw_intent = body.get("intent") if isinstance(body.get("intent"), dict) else None
            intent, rejected = adjudication_validate(raw_intent)
            if rejected:
                result["intent_rejected"] = rejected

            # WHEN the fact became true, not when this row was written. Without it the only
            # lag the substrate can measure is "how long until somebody typed it in", which
            # is a fact about staffing. With it, a year of chargebacks imported this morning
            # still carries its real lags, and core/label_maturity.py can tell an immature
            # window from a complete one. Optional, because an analyst often does not know.
            effective_ts = str(body.get("effective_ts", "") or "")

            result["receipt"] = close_loop(
                STORE, transaction_id, recipient_id, label, is_fraud, rep_rate, source,
                intent=intent or None, effective_ts=effective_ts)
            result["intent_recorded"] = sorted(intent.keys())
            result["effective_ts_recorded"] = bool(effective_ts)
        except Exception:
            pass   # receipt is additive; a backbone failure must not fail the disposition

    return result


@app.get("/feedback/status")
def feedback_status():
    """Loop status: labeled totals, online updates applied, retrain queue depth."""
    if _feedback is None:
        return {"loop": "unavailable", "labeled_total": 0}
    return _feedback.status()


@app.get("/substrate/stats")
def substrate_stats():
    """Health of the labeling substrate: decisions logged, enforced vs shadow, outcome
    coverage and label provenance. The training-readiness dashboard's data source."""
    if STORE is None:
        return {"substrate": "unavailable"}
    return STORE.labeling_stats()


@app.get("/adjudication/schema")
def get_adjudication_schema():
    """What an analyst can be asked to adjudicate when closing a case, and why each answer
    matters. Served from the modules that produce the competing heuristic, so the options an
    analyst picks from are exactly the classes the heuristic can emit.

    Carries `no_default_selected`: the UI must not pre-select the heuristic's guess. The
    graduation gate measures heuristic-versus-human agreement, and pre-filling the human's
    answer with the machine's would manufacture that agreement."""
    return adjudication_schema()


@app.get("/substrate/readiness")
def substrate_readiness():
    """Per-module graduation readiness: heuristic-vs-gold agreement and a verdict for each
    trainable target. Answers 'is this heuristic layer ready to become a trained model yet?'"""
    if STORE is None:
        return {"substrate": "unavailable"}
    from core.graduation import readiness_report
    return readiness_report(STORE)


@app.get("/substrate/next-questions")
def substrate_next_questions(limit: int = 10):
    """Which cases should the analyst adjudicate NEXT, and why.

    Adjudication is the scarcest resource here: the gate needs 50 gold labels and 30
    heuristic/gold pairs per target, and a human labels a handful of cases a day. Working the
    queue in arrival order spends most of that on cases that teach the system nothing.

    Only cases the heuristic has already scored are queued, because a gold label without a
    heuristic prediction cannot form the pair the verdict rests on. A fixed share of the queue
    is drawn representatively rather than by model uncertainty, so the labelled set does not
    collapse onto the decision boundary."""
    if STORE is None:
        return {"substrate": "unavailable", "queue": []}
    from core.active_learning import next_questions
    try:
        return next_questions(STORE, limit=max(1, min(50, limit)))
    except Exception as e:
        return {"substrate": "error", "detail": str(e)[:200], "queue": []}


def score_card_message_gated(msg: dict) -> tuple:
    """The card score with the sequence gate applied. THE one entry point both card paths use.

    Ordering matters and it is the same ordering the device gate uses: the model scores first,
    then the gate may raise that score, and only then do pricing and policy act. Gating after
    the policy has already chosen an action would produce a raised score nobody decided on.
    """
    p, detail = score_card_message(msg)
    raised, gd = apply_sequence_gate(msg, p)
    detail = dict(detail or {})
    detail["sequence_gate"] = gd
    if gd.get("fired"):
        detail["p_fraud_before_gate"] = round(float(p), 4)
    return raised, detail


def apply_sequence_gate(msg: dict, score: float) -> tuple:
    """The card sequence gate, on EVERY card path. Returns `(score, detail)`.

    SHARED BECAUSE OF ADR-001, not for tidiness. Six controls have been wired to one decision
    path and forgotten on another, and the conformance test exists because the seventh was
    inevitable. This is the seventh, so it gets one function and both callers use it.

    ESCALATE-ONLY. The gate may raise a score the model was about to let through and may never
    lower one. That contract lives in redwing-ml/card_sequence.gate and is imported rather than
    reimplemented, because the thresholds there were priced against held-out data and a second
    copy here would drift from the numbers that justified them.

    Degrades to the unmodified score whenever anything is missing: no card key, no substrate, no
    history, ML repo not importable. A gate that cannot see is not evidence of innocence.
    """
    detail = {"fired": False, "available": False}
    try:
        from card_sequence import gate as _seq_gate
    except Exception:                                             # noqa: BLE001
        detail["why"] = "redwing-ml card_sequence not importable"
        return score, detail

    from core.card_history import sequence_view
    from core.card_identity import card_key
    from core.store import eid

    ckey = card_key(msg)
    if not ckey:
        detail["why"] = ("no card identifier on the message; pass card_token or pan. A shared "
                         "fallback key would give every unidentified card one history and the "
                         "gate would fire on all of them")
        return score, detail

    view = sequence_view(STORE, eid("card", ckey), float(msg.get("amount", 0.0) or 0.0))
    detail["available"] = True
    detail["card_known"] = view["card_known"]
    raised, gd = _seq_gate(view, float(score))
    detail.update(gd)
    return raised, detail


def _persist_authorization(msg: dict, decision: dict) -> dict:
    """Write the authorization to the substrate and stamp the decision_id onto the response.

    ADR-001 action item 2. Before this, /authorize was the one decision path that recorded
    nothing, so the card rail had no labels, no outcome-ledger entries and no holdout membership,
    and its model could never be measured for decay or graduated.

    THE ORDER MATTERS AND IT IS NOT THE OBVIOUS ONE. `durable_record` runs FIRST and computes
    everything, including holdout membership, from the message alone. Only then does the store
    get touched. That is what makes the record safe to write behind a network deadline: if the
    write is slow, deferred, or fails outright, the sampling decision has already been made by a
    pure hash of the ARN and cannot drift. A holdout whose membership depends on whether a write
    succeeded is not a randomised holdout.

    The decision_id is deterministic on the ARN for the same reason, so a retried authorization
    overwrites its own row rather than creating a second decision for one payment, and so the id
    can be returned even when the substrate is unavailable.

    A substrate failure degrades LOUDLY into the response (`recorded: false`) rather than
    silently, because a caller that believes its decision was recorded when it was not is exactly
    how a rail ends up unmeasurable again.
    """
    from core.authorization import durable_record

    rec = durable_record(msg, decision, holdout_fn=holdout_decision)
    if not rec:
        decision["recorded"] = False
        decision["not_recorded_reason"] = (
            "no acquirer reference number on the message, so an outcome could never be joined "
            "back to this decision; pass arn, rrn or transaction_id")
        return decision

    # Deterministic and idempotent: one payment, one decision row, however many retries.
    decision["decision_id"] = "auth:" + hashlib.sha256(
        rec["subject_ref"].encode()).hexdigest()[:24]
    decision["holdout"] = rec["holdout"]
    if rec["released"]:
        # A released would-be-decline is actually let through and monitored. The response has to
        # reflect what was ENFORCED, not what was proposed, or the holdout is a fiction.
        decision["action"] = rec["action"]
        decision["monitored"] = True

    if STORE is None:
        decision["recorded"] = False
        decision["not_recorded_reason"] = "substrate unavailable"
        return decision
    try:
        record_decision(
            STORE, subject_ref=rec["subject_ref"], action=rec["action"], module="card_scorer",
            # The CARD, not the transaction. This is what makes the per-card trailing window
            # queryable, and it lands on the already-indexed entity column.
            entity_id=rec["entity_id"],
            score=rec["score"], expected_liability=rec["expected_liability"],
            features=rec["features"], rationale=rec["rationale"],
            model_version=REGISTRY.decision_versions(),
            decision_id=decision["decision_id"],
        )
        decision["recorded"] = True
    except Exception as e:                                        # noqa: BLE001
        decision["recorded"] = False
        decision["not_recorded_reason"] = f"{type(e).__name__}: {str(e)[:120]}"
    return decision


@app.post("/authorize")
def authorize_payment(body: dict):
    """A card authorization. Approve or decline, inside the window, with a response code.

    The question a network actually asks, and the one this platform could not answer until now:
    every other layer existed and none of them were reachable from an authorization.

    Body is the auth message: amount, merchant_id / merchant_name, mcc_code, entry_mode,
    cardholder_name, plus whatever the issuer knows (available_balance, daily_count,
    account_age_days). Everything is optional; missing fields simply mean that check cannot run.

    NOT connected to a network. There is no ISO 8583 wire format and no acquirer. This models
    the contract an issuer operates under so decisioning can be built and measured against it.
    """
    from core.authorization import authorize

    # score_card_message is shared with build_event() (the general ingestion path) rather than
    # kept local to this endpoint - see its own docstring for why that consolidation mattered.
    #
    # The version on main at merge time was the ORIGINAL push-payment scorer (compute_features +
    # ml_score_row), which is the defect this branch exists to fix: it returns ~0.0 on every card
    # message because the signal it wants is not in an authorization. Taking this side deliberately.
    msg = body or {}
    decision = authorize(msg, score_fn=score_card_message_gated)
    return _persist_authorization(msg, decision)


@app.post("/disputes")
def post_dispute(body: dict):
    """Ingest dispute events for one authorization and settle the label they imply.

    Body: {"subject_ref": "<arn or rrn>", "events": [{stage|terminal, ts, reason_code, amount,
    compelling_evidence}, ...]}. The full event list is accepted every time and re-folded, so
    re-posting a file is idempotent rather than double-counting.

    THE POINT OF THIS ENDPOINT IS WHAT IT REFUSES. A chargeback is a claim, not an adjudication,
    so an open dispute emits no label. A service dispute (13.1, 4853) is not fraud and never
    reaches the fraud label space. A fraud claim the merchant wins INVERTS to legit rather than
    disappearing. Every refusal is returned with its reason instead of being silently dropped,
    because a caller that cannot see what was withheld will assume everything was recorded.

    NOT connected to a network. There is no VROL or Mastercom feed; this is the contract those
    rails present, so the label pipeline can be built and measured against it.
    """
    from core.dispute import advance, derive_outcome, to_ledger_record

    subject_ref = str((body or {}).get("subject_ref", "") or "").strip()
    events = (body or {}).get("events") or []
    if not subject_ref:
        raise HTTPException(422, "subject_ref is required: it is the key that joins this "
                                 "dispute back to the authorization that caused it")
    if not isinstance(events, list):
        raise HTTPException(422, "events must be a list")

    state = advance(events)
    verdict = derive_outcome(state)
    out = {
        "subject_ref": subject_ref,
        "stage": state["stage"], "terminal": state["terminal"], "settled": state["settled"],
        "reason_code": state["reason_code"], "classification": state["classification"],
        "n_events": state["n_events"],
        "label": verdict,
        "recorded": False,
    }

    if STORE is None:
        out["not_recorded_reason"] = "substrate unavailable"
        return out

    try:
        STORE.append_event(
            event_type="dispute", payload={"subject_ref": subject_ref, "events": events},
            derived={"stage": state["stage"], "terminal": state["terminal"],
                     "reason_code": state["reason_code"], "emitted": verdict["emit"]},
        )
    except Exception as e:                                        # noqa: BLE001
        out["not_recorded_reason"] = f"event log failed: {type(e).__name__}"
        return out

    rec = to_ledger_record(state, subject_ref)
    if rec is None:
        # Withheld on purpose. Say so, and say why, rather than returning a bare success.
        out["recorded"] = True
        out["label_written"] = False
        out["withheld_because"] = verdict["reason"]
        return out

    from core.outcome_ledger import record_outcome
    try:
        out["ledger"] = record_outcome(STORE, rec)
        out["recorded"] = True
        out["label_written"] = True
    except Exception as e:                                        # noqa: BLE001
        out["not_recorded_reason"] = f"ledger write failed: {type(e).__name__}"
    return out


@app.get("/disputes/contract")
def dispute_contract():
    """The reason-code taxonomy this rail understands, and which codes may become fraud labels.

    The load-bearing column is `fraud_eligible`. Only the fraud family can ever reach
    `outcome.is_fraud`; a consumer dispute recorded as fraud trains the model that a late parcel
    is fraud, and consumer disputes are a large share of all chargebacks.
    """
    from core.dispute import (AMBIGUOUS_FRAUD_CODES, REASON_CODES, STAGES,
                              STAGE_WINDOW_DAYS, TERMINAL, classify)
    return {
        "stages": list(STAGES),
        "terminal_outcomes": TERMINAL,
        "stage_window_days": STAGE_WINDOW_DAYS,
        "window_note": ("ASSUMPTION, not sourced. Typical network windows used to compute a "
                        "maturity floor; a real deployment substitutes the operative rulebook."),
        "ambiguous_fraud_codes": list(AMBIGUOUS_FRAUD_CODES),
        "ambiguity_note": ("these codes span true fraud, friendly fraud and merchant error by "
                           "definition, so the code alone cannot separate them and they settle "
                           "at lower confidence"),
        "codes": [{"code": c, **classify(c)} for c in sorted(REASON_CODES)],
        "note": ("modelled, not connected to a network. No VROL or Mastercom feed; this is the "
                 "contract those rails present so the label pipeline can be measured."),
    }


@app.get("/authorize/contract")
def authorization_contract():
    """The response codes this issuer emits, and which of them permit a retry.

    Soft versus hard is contractual: networks limit re-attempts on specific codes and fine
    violations, so anything building a recovery flow needs to read this rather than guess.
    """
    from core.authorization import (AUTH_BUDGET_MS, HARD_DECLINES, RESPONSE_CODES,
                                    SOFT_DECLINES, STIP_THRESHOLD_MS)
    return {
        "budget_ms": AUTH_BUDGET_MS, "stand_in_threshold_ms": STIP_THRESHOLD_MS,
        "codes": [{"code": c, "text": t,
                   "class": ("approved" if c == "00" else
                             "soft" if c in SOFT_DECLINES else
                             "hard" if c in HARD_DECLINES else "other"),
                   "retry_allowed": c == "00" or c in SOFT_DECLINES}
                  for c, t in sorted(RESPONSE_CODES.items())],
        "note": ("modelled, not connected to a network. No ISO 8583 wire format and no "
                 "acquirer; this is the contract an issuer operates under, so decisioning can "
                 "be measured against it."),
    }


@app.get("/screening/status")
def screening_status():
    """Is sanctions screening actually in force?

    The question `case_file.py` used to answer with a hardcoded `True` sitting above a coin
    flip. This reads the loaded list, so "screened" is a fact rather than an assertion. Note
    `fails: closed` - unlike every advisory layer here, an unavailable list blocks rather than
    approving unscreened traffic.
    """
    from core.screening import status
    return status()

@app.get("/model/inventory")
def model_inventory():
    """Every model that can touch a decision: state, risk tier, version, and whether it loaded.

    The artifact model-risk guidance asks for, and the reason the registry is the load path
    rather than a catalogue describing loads that happen elsewhere. A model outside the registry
    is where governance collapses in practice.
    """
    from core.model_registry import REGISTRY
    inv = REGISTRY.inventory()
    inv["decision_stamp"] = REGISTRY.decision_versions()
    return inv


@app.get("/model/performance")
def model_performance(window_days: int = 30, windows: int = 6, mature_only: bool = True):
    """Did the model get worse, and how would we know.

    `drift_monitor` computes PSI over distributions, which is label-free: it can say the input
    moved and can never say the model decayed. This is the other half, and it separates the
    three explanations for a bad-looking month, because they demand opposite responses:

        degraded            retrain
        population_shift    the model may be fine, the traffic changed
        unmeasurable        nothing happened; the outcomes have not arrived yet

    Metrics are named `*_on_allowed`, never plain precision/recall, because outcomes exist only
    where the payment was allowed. Where a holdout exists, the released sample estimates what
    the block wall is hiding.
    """
    if STORE is None:
        return {"substrate": "unavailable"}
    try:
        from core.model_performance import diagnose, trend
        return {"diagnosis": diagnose(STORE, window_days=window_days),
                **trend(STORE, windows=windows, window_days=window_days,
                        mature_only=mature_only)}
    except Exception as e:                                        # never take the page down
        return {"substrate": "error", "detail": str(e)[:200]}


@app.post("/outcomes")
def post_outcomes(body: dict):
    """Ingest outcome reports: chargebacks, recalls, confirmed losses, victim reports.

    Body: {"records": [{subject_ref, outcome, source, effective_ts, reference, ...}]}, or a
    single record. `override_reason` on the body lets a weaker source deliberately overturn a
    stronger standing one, which real work needs: an analyst who reviews a chargeback and finds
    first-party abuse is exactly that. Without it, a weaker source is recorded as evidence and
    does not win.

    Of the five sources the graduation gate trusts, only `analyst` previously had a live path.
    This is the other four.
    """
    if STORE is None:
        raise HTTPException(503, "Substrate not available.")
    from core.outcome_ledger import ingest_outcomes
    recs = body.get("records")
    if recs is None:
        recs = [body] if body.get("subject_ref") or body.get("transaction_id") else []
    return ingest_outcomes(STORE, recs,
                           override_reason=str(body.get("override_reason", "") or ""))


@app.get("/outcomes/stats")
def outcomes_stats():
    """The outcome supply by source, plus the disagreements it has produced."""
    if STORE is None:
        return {"substrate": "unavailable"}
    from core.outcome_ledger import ledger_stats
    return ledger_stats(STORE)


@app.get("/outcomes/disagreements")
def outcomes_disagreements(limit: int = 100):
    """Cases where two ground-truth sources said different things.

    The most valuable rows in the substrate. When the analyst cleared a payment and a chargeback
    later says fraud, that is a labelled FALSE NEGATIVE found by the world rather than by us,
    with the point-in-time features still attached to the decision.
    """
    if STORE is None:
        return {"substrate": "unavailable"}
    from core.outcome_ledger import disagreements
    d = disagreements(STORE, limit=limit)
    return {"count": len(d), "disagreements": d,
            "missed_fraud": sum(1 for x in d if x["kind"] == "missed_fraud"),
            "reversals": sum(1 for x in d if x["reversal"])}


@app.get("/substrate/maturity")
def substrate_maturity(horizon_days: int = 90, coverage: float = 0.9):
    """How complete the label set is, and therefore what window is safe to train on.

    Fraud labels arrive late and the lateness is not random: the scams that do the damage on an
    irrevocable rail surface weeks or months after the payment, because the victim does not know
    yet. So the recent window is systematically missing its positives, and anything measured
    there flatters itself. This reports the arrival-lag curve per target and the maturity floor
    it implies.

    It very often answers "not derivable", and that is the honest answer rather than a failure:
    a curve needs gold labels on cohorts old enough to have settled, and a young substrate has
    none. A fitted curve there would license training on exactly the window it exists to
    withhold.
    """
    if STORE is None:
        return {"substrate": "unavailable"}
    try:
        from core.label_maturity import maturity_report
        return maturity_report(STORE, horizon_days=horizon_days, coverage=coverage)
    except Exception as e:                                        # never take the page down
        return {"substrate": "error", "detail": str(e)[:200]}


@app.get("/substrate/graduation")
def substrate_graduation():
    """The full evidence chain behind a rule being retired by a model that beat it.

    Two sections, deliberately separate, because conflating them would be the dishonest move:

      live       what THIS instance's substrate supports right now. On a fresh instance that
                 is "not enough data", and the gate saying no is a correct answer, not a bug.
      experiment a RECORDED run over 300K real labeled transactions, served with its
                 provenance (row counts, the rule, the labelling policy). This is where the
                 numbers come from; it is not recomputed per request and does not pretend to be.
    """
    out = {"live": {"substrate": "unavailable"}, "experiment": None}

    if STORE is not None:
        from core.graduation import readiness_report
        from core.train import train_target
        try:
            out["live"] = {
                "substrate": STORE.labeling_stats(),
                "readiness": readiness_report(STORE, targets=[("outcome", "is_fraud")]),
                "trained": train_target(STORE, "outcome", "is_fraud",
                                        observed_only=True, positive_label=FRAUD_TRUE),
            }
        except Exception as e:                                    # never take the page down
            out["live"] = {"substrate": "error", "detail": str(e)[:200]}

    art = MODELS_DIR / "phase2_experiment.json"
    if art.exists():
        try:
            out["experiment"] = json.load(open(art))
        except Exception:
            out["experiment"] = None
    return out


@app.post("/telemetry")
def post_telemetry(body: dict):
    """Ingest REAL behavioural telemetry a client SDK reports for a subject (session /
    transaction). Body: {subject_ref, entity_id?, telemetry:{...}}. This is what lets the actor
    modules run on genuine behaviour instead of on values derived from the typology (leakage).
    Returns the tells derived from the reported values so the caller can see what fired."""
    if STORE is None:
        return {"telemetry": "unavailable"}
    subject_ref = str(body.get("subject_ref", "")).strip()
    if not subject_ref:
        raise HTTPException(400, "subject_ref is required")
    tel = body.get("telemetry") if isinstance(body.get("telemetry"), dict) else {}
    STORE.record_telemetry(subject_ref, tel, entity_id=str(body.get("entity_id", "") or ""))
    return {"ok": True, "subject_ref": subject_ref, "derived_signals": derive_signals(tel)}


@app.post("/telemetry/fingerprint")
def post_fingerprint(body: dict):
    """Derive a device identity from components a client collector reported.

    Body: {subject_ref, components:{...}}. `static/redwing-fp.js` is the collector; see
    core/fingerprint.py for why identity comes from the ANCHOR components only and why a
    low-entropy fingerprint is explicitly refused as an identity.

    This is the producer the device layer was missing. core/telemetry.py has defined the
    reporting contract since it was written and nothing filled it; the device graph counts
    accounts per device_id and that id arrived assigned by nobody. This closes
    collect -> derive -> graph.

    NOT available on the card rail, and that is architectural rather than a gap: an ISO 8583
    authorization reaches an issuer from the acquirer with no browser attached. Fingerprinting
    only exists on surfaces where the institution owns the client.
    """
    from core.fingerprint import derive, to_telemetry

    subject_ref = str(body.get("subject_ref", "")).strip()
    if not subject_ref:
        raise HTTPException(400, "subject_ref is required")
    comps = body.get("components") if isinstance(body.get("components"), dict) else {}
    # Cheap early rejection. derive() bounds the payload itself, but only after FastAPI has
    # already parsed the whole body into memory, so this refuses an obviously hostile shape
    # before it is read. The real ceiling belongs at the ASGI/proxy layer as a request-body
    # limit; this is the application-level backstop, not a substitute for it.
    from core.fingerprint import MAX_COMPONENTS
    if len(comps) > MAX_COMPONENTS * 4:
        raise HTTPException(413, f"components payload has {len(comps)} keys; the contract "
                                 f"declares fewer than {MAX_COMPONENTS}")
    fp = derive(comps)

    # The automation and integrity tells feed the actor layer through the SAME path a client SDK
    # would use, rather than a second private channel. derive_signals() then fires only on what
    # was actually reported, which is what keeps the actor read honest.
    #
    # ONLY WHEN THERE IS SOMETHING TO REPORT, and this guard is a safeguarding control rather
    # than an optimisation. `get_telemetry` returns the NEWEST row for a subject, and
    # `to_telemetry` on a clean fingerprint returns {}. Without the guard, one unauthenticated
    # POST naming a victim's subject_ref wrote an empty row that became the newest, and the
    # operator's duress, coaching and remote-access read for an in-flight scam went to nothing.
    # Verified before the fix: a subject with 7 live coercion signals was reduced to 0 by a
    # single benign fingerprint POST returning HTTP 200.
    #
    # This closes the SILENCING direction only. The injection direction, where an attacker
    # ASSERTS integrity flags against someone else's subject_ref, needs the caller bound to the
    # subject by a server-issued session token and is NOT fixed here.
    # PERSIST THE DERIVED IDENTITY, which is what makes any of this reach a decision. Refused
    # inside the store unless the fingerprint cleared the entropy floor: one below it names a
    # crowd, and writing that as a device stacks unrelated accounts onto a single graph node.
    identity_written = False
    if STORE is not None:
        try:
            identity_written = STORE.record_device_identity(subject_ref, fp)
        except Exception:                                             # noqa: BLE001
            identity_written = False

    tel = to_telemetry(fp)
    if STORE is not None and tel:
        try:
            STORE.record_telemetry(subject_ref, tel,
                                   entity_id=str(body.get("entity_id", "") or ""))
        except Exception:                                             # noqa: BLE001
            pass    # telemetry is never worth failing a fingerprint for

    return {"ok": True, "subject_ref": subject_ref, "fingerprint": fp,
            # Whether this id will actually be used by the gate. False means the fingerprint was
            # refused as an identity, which is a normal and correct outcome for a hardened
            # browser, not an error.
            "identity_persisted": identity_written}


@app.get("/telemetry/fingerprint/contract")
def fingerprint_contract():
    """What a collector must report, split by how fast each component drifts."""
    from core.fingerprint import (ANCHOR_COMPONENTS, DRIFT_COMPONENTS,
                                  MIN_ENTROPY_BITS_FOR_IDENTITY, RELINK_SIMILARITY)
    return {
        "collector": "/static/redwing-fp.js",
        "anchor_components": list(ANCHOR_COMPONENTS),
        "drift_components": list(DRIFT_COMPONENTS),
        "min_entropy_bits_for_identity": MIN_ENTROPY_BITS_FOR_IDENTITY,
        "relink_similarity": RELINK_SIMILARITY,
        "note": ("identity is the ANCHOR hash only; DRIFT re-links a device whose anchor moved. "
                 "A fingerprint below the entropy floor names a crowd rather than a device and "
                 "is refused as an identity, because privacy-hardened browsers all collide onto "
                 "one value and would otherwise read as a shared fraud device."),
        "not_available_on": ("card authorization: an ISO 8583 message carries no client to "
                             "fingerprint"),
    }


@app.get("/actor/{subject_ref}")
def actor_read(subject_ref: str):
    """The telemetry-derived actor read for a subject: the offender view (motive, lifecycle,
    intervention) and the victim view (scam arc, coercion-in-flight, protection). Silent when no
    telemetry was reported, which is the honest answer with no behaviour to reason over."""
    if STORE is None:
        return {"actor": "unavailable"}
    from core.telemetry import assess_subject
    return assess_subject(STORE, subject_ref)


# -- Injection Pipeline --------------------------------------------------------

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

    # Schema gate: reject broken input instead of scoring a silently-defaulted 0.
    v = validate_event(body, source="ingest")
    if not v["valid"]:
        raise HTTPException(422, {"error": "schema_validation_failed",
                                  "errors": v["errors"], "warnings": v["warnings"]})

    event = build_event(v["event"])   # drift_monitor.record() already called inside build_event
    event["source"] = "injected"
    if v["warnings"]:
        event["ingest_warnings"] = v["warnings"]

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

    results, alerts, rejected = [], 0, []
    for tx in transactions:
        v = validate_event(tx, source="ingest_batch")
        if not v["valid"]:
            rejected.append({"input": tx, "errors": v["errors"]})   # dead-letter seed
            continue
        event = build_event(v["event"])
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
        "rejected":    len(rejected),
        "dead_letter": rejected,               # the events a real pipeline would route to a DLQ
        "alerts":      alerts,
        "alert_rate":  round(alerts / len(results), 4) if results else 0.0,
        "results":     results,
    }


@app.get("/ingest/schema")
def ingest_schema():
    """The ingestion contract: required / recommended fields, rail normalisation, and the
    label-only fields that must never be used as model features (leakage)."""
    return ingest_contract()


# -- Streaming transport: decoupled intake -> durable queue -> background scorer --

def _stream_handler(payload: dict) -> None:
    """Score one dequeued event. A raised exception here is what the transport retries and,
    on exhaustion, dead-letters, so a transient scorer failure never loses the event."""
    event = build_event(payload)
    event["source"] = "streamed"
    _ingest_buffer.appendleft(event)
    _fan_out(event)


async def _stream_consumer_loop() -> None:
    """Drain the durable queue into the scorer, off the request path. Idle-sleeps when empty."""
    loop = asyncio.get_event_loop()
    while True:
        res = {"processed": 0}
        try:
            if MODEL_OK and TRANSPORT is not None:
                res = await loop.run_in_executor(
                    None, lambda: TRANSPORT.consume_batch("ingest", _stream_handler, 50))
        except Exception:
            pass
        await asyncio.sleep(0.2 if res.get("processed", 0) else 1.0)


@app.post("/stream/publish")
def stream_publish(body: dict):
    """Validate and ENQUEUE an event onto the durable transport (fast), instead of scoring it
    inline. A background consumer drains the queue. Returns 429 under backpressure, so a burst
    is throttled rather than silently dropped."""
    if TRANSPORT is None:
        raise HTTPException(503, "stream transport unavailable")
    v = validate_event(body, source="stream")
    if not v["valid"]:
        raise HTTPException(422, {"error": "schema_validation_failed", "errors": v["errors"]})
    try:
        seq = TRANSPORT.publish("ingest", v["event"]["transaction_id"], v["event"])
    except BackpressureError as e:
        raise HTTPException(429, str(e))
    return {"queued": seq is not None, "seq": seq, "deduped": seq is None, "warnings": v["warnings"]}


@app.get("/stream/stats")
def stream_stats():
    """Transport health: ready depth, processed, dead-letter count, capacity."""
    if TRANSPORT is None:
        return {"stream": "unavailable"}
    return TRANSPORT.stats("ingest")


@app.get("/stream/dead_letter")
def stream_dead_letter():
    """Events that failed scoring past the retry limit (a real pipeline's DLQ)."""
    if TRANSPORT is None:
        return {"stream": "unavailable"}
    return {"dead_letters": TRANSPORT.dead_letters("ingest")}


@app.post("/stream/replay")
def stream_replay():
    """Reset dead-lettered events back to ready for reprocessing (replay from the DLQ)."""
    if TRANSPORT is None:
        return {"stream": "unavailable"}
    return {"replayed": TRANSPORT.replay("ingest")}


# -- Source connectors: pull ingestion (checkpointed, resumable, idempotent) --

async def _connector_poll_loop() -> None:
    """Poll the source connector on an interval so a file drop is auto-ingested. The connector
    checkpoints its offset, so this only ever picks up what is new."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            if FILE_CONNECTOR is not None and MODEL_OK:
                await loop.run_in_executor(None, FILE_CONNECTOR.poll)
        except Exception:
            pass
        await asyncio.sleep(5.0)


@app.get("/connectors")
def connectors_list():
    """Registered source connectors and their durable checkpoints (how far each has consumed)."""
    if STORE is None:
        return {"connectors": []}
    conns = []
    if FILE_CONNECTOR is not None:
        conns.append({"connector": FILE_CONNECTOR.connector_id, "source_type": FILE_CONNECTOR.source_type,
                      "path": FILE_CONNECTOR.path, "checkpoint": STORE.get_checkpoint(FILE_CONNECTOR.connector_id)})
    return {"connectors": conns, "checkpoints": STORE.checkpoints(),
            "webhook_sources": WEBHOOK.sources() if WEBHOOK is not None else []}


@app.post("/connectors/file/poll")
def connectors_file_poll():
    """Trigger a poll of the file connector now: read new lines since the checkpoint, validate,
    publish valid events to the transport, advance the checkpoint. Returns the poll stats."""
    if FILE_CONNECTOR is None:
        raise HTTPException(503, "file connector unavailable")
    return FILE_CONNECTOR.poll()


@app.post("/connectors/db/poll")
def connectors_db_poll(body: dict):
    """Poll a source SQL table for new transactions since its checkpoint. Body:
    {db_path, table, id_column?, field_map?, connector_id?}. Points at any SQLite transactions
    table, maps its columns to the canonical schema (field_map: source_col -> canonical_field),
    validates, and publishes new rows to the durable transport. Incremental by a monotonic
    watermark (id_column, default rowid), so it only ever ingests what is new."""
    if TRANSPORT is None or STORE is None:
        raise HTTPException(503, "transport/store unavailable")
    db_path = str(body.get("db_path", "")).strip()
    table = str(body.get("table", "")).strip()
    if not db_path or not table:
        raise HTTPException(400, "db_path and table are required")
    conn = DBConnector(
        connector_id=str(body.get("connector_id") or f"db:{table}"),
        transport=TRANSPORT, checkpoints=STORE, db_path=db_path, table=table,
        id_column=str(body.get("id_column") or "rowid"),
        field_map=body.get("field_map") if isinstance(body.get("field_map"), dict) else None)
    return conn.poll()


@app.post("/connectors/webhook/{source}")
async def connectors_webhook(source: str, request: Request):
    """Real-time PUSH ingestion from an AUTHENTICATED source. The caller signs the raw body with
    the source's shared secret and sends it as 'X-Signature: sha256=<hex>'. An unknown source, a
    bad signature, or a tampered body is rejected before the event reaches the pipeline, so the
    push path cannot be used to inject fabricated events."""
    if WEBHOOK is None:
        raise HTTPException(503, "webhook receiver unavailable")
    body = await request.body()
    r = WEBHOOK.accept(source, body, request.headers.get("x-signature", ""))
    if not r["accepted"]:
        detail = {"reason": r["reason"]}
        if "errors" in r:
            detail["errors"] = r["errors"]
        raise HTTPException(r["status"], detail)
    return r


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


# -- Rule Factory Endpoints ----------------------------------------------------

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


# -- Network Graph -------------------------------------------------------------

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


# -- Drift Monitor ------------------------------------------------------------

@app.get("/drift/status")
def get_drift_status():
    """
    ADWIN-style concept drift report.
    Returns PSI on model scores and 5 key features:
      state: warming_up | stable | warning | drift
      score_psi / feature_psi - Population Stability Index values
      drift_events - history of state transitions into warning/drift
    PSI < 0.10: stable · 0.10-0.20: warning · > 0.20: retrain recommended
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


# -- Backbone (entity/event store) ---------------------------------------------
# The Phase 1 substrate: every scored transaction leaves a durable entity+event
# trail here. These endpoints expose it for the loop receipt (WS2) and the UI.

@app.get("/backbone/stats")
def backbone_stats():
    """Entity and event counts in the durable store, by type, plus tenants."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    return STORE.stats()


@app.post("/narrative")
def narrative(body: dict):
    """Scam-narrative for a case (WS4): explains the con, not the transaction. Body:
    {typology, amount, rail, is_new_recipient, expected_liability}. No LLM key needed."""
    return scam_narrative(
        typology=str(body.get("typology", "") or ""),
        signals={"amount": body.get("amount", 0.0), "rail": body.get("rail", ""),
                 "is_new_recipient": body.get("is_new_recipient"),
                 "expected_liability": body.get("expected_liability")})


@app.get("/backbone/liability")
def backbone_liability(event_type: str = "alert"):
    """Portfolio liability-at-risk (WS4): expected reimbursement dollars summed over
    open alerts. The number a fraud-ops lead answers to, priced not just probability."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    return STORE.liability_at_risk(event_type)


@app.get("/backbone/recent")
def backbone_recent(event_type: str = "", limit: int = 50):
    """Most recent events on the backbone (optionally filtered by type)."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    evs = STORE.recent_events(event_type or None, limit=limit)
    return [e.__dict__ for e in evs]


@app.get("/backbone/entity/{entity_id:path}")
def backbone_entity(entity_id: str, events: int = 20):
    """One entity (with its live reputation) plus the events that touch it -
    the object an investigator and the network layer both read."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    ent = STORE.get_entity(entity_id)
    if ent is None:
        raise HTTPException(404, f"entity '{entity_id}' not found")
    return {"entity": ent.__dict__,
            "events": [e.__dict__ for e in STORE.events_for_entity(entity_id, limit=events)]}


import threading as _gthreading
_fraud_graph_cache = None
_fraud_graph_lock = _gthreading.RLock()


def _build_graph_edges() -> list:
    """Edges for the fraud graph: fraud rows from the in-memory ledger (fast) plus a
    small legit sample, overlaid with the store-only demo mule. Building from df_all
    avoids the per-recipient SQLite scan that made this ~40s."""
    edges = []
    if not df_all.empty and {"user_id", "recipient_id"}.issubset(df_all.columns):
        def _rows(frame, is_fraud):
            # vectorised column extraction (itertuples/_asdict is pathologically slow here)
            n = len(frame)
            users  = frame["user_id"].astype(str).tolist()
            recips = frame["recipient_id"].astype(str).tolist()
            devs   = frame["device_id"].astype(str).tolist() if "device_id" in frame else [""] * n
            amts   = pd.to_numeric(frame.get("amount", 0), errors="coerce").fillna(0.0).tolist()
            typs   = (frame["fraud_typology"].astype(str).tolist()
                      if is_fraud and "fraud_typology" in frame else ["none"] * n)
            for u, d, r, a, t in zip(users, devs, recips, amts, typs):
                edges.append({"user": u, "device": d, "recipient": r,
                              "is_fraud": is_fraud, "amount": float(a), "typology": t})
        mask = df_all["is_fraud"] == True if "is_fraud" in df_all.columns else None
        _rows(df_all[mask].head(5000) if mask is not None else df_all.iloc[0:0], 1)
        _rows(df_all[~mask].head(150) if mask is not None else df_all.head(150), 0)
    # overlay: the store-only demo mule
    try:
        for e in STORE.events_for_entity("recipient:DEMO-MULE-PIG-01", event_type="transaction", limit=40):
            uid = next((x[5:] for x in e.entities if x.startswith("user:")), None)
            edges.append({"user": uid, "device": None, "recipient": "DEMO-MULE-PIG-01",
                          "is_fraud": int(e.derived.get("is_fraud", 0) or 0),
                          "amount": float(e.payload.get("amount") or 0.0),
                          "typology": str(e.payload.get("typology") or "")})
    except Exception:
        pass
    return edges


@app.get("/backbone/graph")
def backbone_graph(refresh: bool = False):
    """A real fraud graph from the backbone: the top mule recipients as detected
    rings, their accounts and shared devices, plus a clean periphery. Cached (the
    graph is stable); pass refresh=true to rebuild. Replaces the curated demo."""
    global _fraud_graph_cache
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    with _fraud_graph_lock:
        if _fraud_graph_cache is None or refresh:
            from core.graph import build_fraud_graph
            _fraud_graph_cache = build_fraud_graph(_build_graph_edges())
        return _fraud_graph_cache


# -- Consortium: privacy-preserving cross-institution network (WS3) ------------
# The n=2 moat. A payee's fraud reputation combined across institutions via
# differential privacy, so a victim's bank can flag a mule it cannot see alone
# without any institution sharing raw data. Fictional tenants (see core/consortium).
#
# The cross-institution view is served from a cached index built in ONE pass over
# all transaction edges (a per-recipient JOIN scan does not scale). The index is
# warmed in the background at startup, so lookups are instant.

import threading as _cthreading
_consortium_index = None
_consortium_index_lock = _cthreading.Lock()


def _get_consortium_index(force: bool = False):
    """Build (once) and cache {recipient_id: {institution: [tx, fraud]}}. Built from
    the in-memory ledger with vectorised pandas (a SQLite per-recipient or triple-join
    scan does not scale to 880k rows), then overlaid with the few store-only recipients
    (e.g. the injected demo mule) that are not in the ledger."""
    global _consortium_index
    if STORE is None:
        return {}
    with _consortium_index_lock:
        if _consortium_index is not None and not force:
            return _consortium_index
        from core.consortium import institution_of, INSTITUTIONS
        idx: dict = {}
        # base: the 880k ledger already in memory, grouped by (recipient, institution)
        if not df_all.empty and {"user_id", "recipient_id"}.issubset(df_all.columns):
            d = df_all[["user_id", "recipient_id", "is_fraud"]].dropna(subset=["recipient_id"]).copy()
            d["is_fraud"] = pd.to_numeric(d["is_fraud"], errors="coerce").fillna(0).astype(int)
            # hash each of the ~1.4k users once, not once per row (see _get_aiq_index)
            _u = d["user_id"].astype(str)
            _im = {u: institution_of(u) for u in _u.unique()}
            d["inst"] = _u.map(_im)
            d["rid"]  = "recipient:" + d["recipient_id"].astype(str)
            g = d.groupby(["rid", "inst"])["is_fraud"].agg(cnt="count", frd="sum")
            for (rid, inst), r in g.iterrows():
                idx.setdefault(rid, {k: [0, 0] for k in INSTITUTIONS})[inst] = [int(r.cnt), int(r.frd)]
        # overlay: store-only recipients not present in the ledger (injected/demo).
        #
        # ONE streaming pass over all transaction edges, not a per-recipient JOIN. Measured on
        # the real backbone the per-recipient JOIN costs ~25ms cold, and there are ~6.1k
        # recipient entities, so the N+1 version spent ~150s building this index - which is the
        # whole reason the first request after boot took a minute. The single pass does the same
        # work in ~35s and, more importantly, is O(edges) instead of O(recipients x JOIN).
        #
        # Only recipients the ledger did NOT already cover are written, so the vectorised
        # groupby above stays authoritative and this cannot change an existing count.
        try:
            overlay: dict = {}
            for recip, usr, fr in STORE.all_transaction_edges():
                if recip in idx:
                    continue                      # the ledger already has this payee
                c = overlay.setdefault(recip, {k: [0, 0] for k in INSTITUTIONS})[institution_of(usr)]
                c[0] += 1
                c[1] += int(fr)
            for rid, counts in overlay.items():
                if any(c[0] for c in counts.values()):
                    idx[rid] = counts
        except Exception:
            pass
        _consortium_index = idx
        return _consortium_index


@app.get("/consortium/recipient/{recipient_id:path}")
def consortium_recipient(recipient_id: str, as_institution: str = "inst_neobank",
                         epsilon: float = 1.0):
    """Cross-institution reputation for one payee, from a querying institution's view:
    each institution's LOCAL view, the DP-COMBINED network view, and whether the
    network reveals a mule the querying institution could not see in its own data."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    from core.consortium import views_from_counts, network_reveal, INSTITUTIONS
    if as_institution not in INSTITUTIONS:
        raise HTTPException(400, f"as_institution must be one of {list(INSTITUTIONS)}")
    rid = recipient_id if recipient_id.startswith("recipient:") else f"recipient:{recipient_id}"
    counts = _get_consortium_index().get(rid)
    if not counts:
        raise HTTPException(404, f"no transactions found for payee '{recipient_id}'")
    local  = views_from_counts(counts)
    reveal = network_reveal(local, as_institution, epsilon)
    reveal["recipient_id"]          = rid
    reveal["all_institution_views"] = local
    return reveal


@app.get("/consortium/mules")
def consortium_mules(epsilon: float = 1.0, limit: int = 20):
    """Flagship cross-institution mules from the cached index: payees BELOW the alert
    line at an institution that banks with them, yet flagged by the DP-combined
    network. Exactly the mules no single bank can catch alone."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    from core.consortium import find_mules_in_index, ALERT_THRESHOLD
    idx   = _get_consortium_index()
    mules = find_mules_in_index(idx, epsilon=epsilon, limit=limit)
    return {"threshold": ALERT_THRESHOLD, "epsilon": epsilon,
            "found": len(mules), "mules": mules, "index_recipients": len(idx)}


# -- Authorization IQ ----------------------------------------------------------
# The consortium made consumable at AUTHORIZATION TIME: the push-rail analog of Mastercard's
# Authorization IQ (AQF) fields. Each insight carries its NETWORK DELTA - what the querying bank
# gains over its own book - and the headline is the reveal: a mule below the local alert line
# but above the network's. See core/authorization_iq.py.
#
# The index adds distinct-sender fan-in and per-rail amount norms to the cached consortium
# index, all vectorised over the in-memory ledger (the pure core.authorization_iq.build_index is
# the tested reference; this populates the same AIQIndex fields at ledger scale).

_aiq_index = None
_aiq_index_lock = _cthreading.Lock()


def _get_aiq_index(force: bool = False):
    global _aiq_index
    if STORE is None:
        return None
    with _aiq_index_lock:
        if _aiq_index is not None and not force:
            return _aiq_index
        from core.authorization_iq import AIQIndex
        from core.consortium import institution_of
        idx = AIQIndex()
        idx.consortium_index = _get_consortium_index()
        if not df_all.empty and {"user_id", "recipient_id"}.issubset(df_all.columns):
            d = df_all[["user_id", "recipient_id"]].dropna(subset=["recipient_id"]).copy()
            d["rid"] = "recipient:" + d["recipient_id"].astype(str)
            # institution depends ONLY on user_id and there are ~1.4k users, so hash each user
            # ONCE, not per row: .map(institution_of) over 897k rows is 897k sha256 calls (~50s);
            # mapping the unique users first makes it ~1.4k.
            uids = d["user_id"].astype(str)
            inst_map = {u: institution_of(u) for u in uids.unique()}
            d["inst"] = uids.map(inst_map)
            idx.fanin = d.groupby("rid")["user_id"].nunique().astype(int).to_dict()
            fbi: dict = {}
            for (rid, inst), v in d.groupby(["rid", "inst"])["user_id"].nunique().items():
                fbi.setdefault(rid, {})[inst] = int(v)
            idx.fanin_by_inst = fbi
            idx.recipient_tx = d.groupby("rid").size().astype(int).to_dict()
            idx.network_tx_total = int(len(d))
            rail_col = "payment_rail" if "payment_rail" in df_all.columns else "rail"
            if rail_col in df_all.columns and "amount" in df_all.columns:
                amt = pd.to_numeric(df_all["amount"], errors="coerce")
                gg = (df_all.assign(_amt=amt).dropna(subset=["_amt"])
                      .groupby(rail_col)["_amt"].agg(["count", "mean", "std"]))
                for rail, r in gg.iterrows():
                    idx.rail_norm[str(rail)] = {"n": int(r["count"]),
                                                "mean": round(float(r["mean"]), 2),
                                                "std": round(float(r["std"] or 0.0), 2)}
        _aiq_index = idx
        return _aiq_index


@app.post("/authorization-iq")
def authorization_iq(body: dict):
    """Authorization-time network intelligence for one PUSH payment.

    Body: {recipient_id, amount, rail, sender_id?, as_institution?, epsilon?}. Returns the
    AQF-style insight pack: each field with its network delta, the composed network_risk, the
    reason codes that fired, and whether the network REVEALS a payee the querying bank could not
    have flagged from its own book. This is the signal a single bank on a push rail cannot
    produce, delivered before the irrevocable payment settles."""
    if STORE is None:
        raise HTTPException(503, "Backbone store not available.")
    from core.authorization_iq import authorize
    from core.consortium import INSTITUTIONS
    rid = str(body.get("recipient_id") or "").strip()
    if not rid:
        raise HTTPException(400, "recipient_id is required")
    if not rid.startswith("recipient:"):
        rid = f"recipient:{rid}"
    # The network index is warmed in the background at startup and takes ~35s over 881k edges.
    # A request that arrives during the warm must NOT block on the lock for half a minute: say
    # so immediately and let the caller retry or fall back. 503 + Retry-After is the honest
    # answer to "not ready yet"; silently waiting looks like a hung endpoint.
    if _aiq_index is None:
        raise HTTPException(503, "Authorization IQ network index is still warming; retry shortly.",
                            headers={"Retry-After": "20"})
    as_inst = body.get("as_institution")
    if as_inst and as_inst not in INSTITUTIONS:
        raise HTTPException(400, f"as_institution must be one of {list(INSTITUTIONS)}")
    payment = {
        "recipient": rid,
        "sender": body.get("sender_id") or body.get("user_id"),
        "amount": body.get("amount", 0.0),
        "rail": body.get("rail") or body.get("payment_rail") or "unknown",
    }
    pack = authorize(payment, _get_aiq_index(),
                     querying_institution=as_inst, epsilon=float(body.get("epsilon", 1.0)))
    pack["recipient_id"] = rid
    return pack


# -- SyntheticID Ingest --------------------------------------------------------

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


# -- LLM Proxy -----------------------------------------------------------------
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

    # -- Anthropic path --------------------------------------------------------
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

    # -- OpenAI-compatible path (openai / groq / mistral) ---------------------
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


# -- Autonomous Agent Endpoints ------------------------------------------------

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


# -- Integration Hub -----------------------------------------------------------

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

    # Agent-drafted, HUMAN-ATTESTED. A narrative filed with a regulator under a named person's
    # signature must (a) say nothing the case file does not, and (b) be the exact text that
    # person signed. Both are enforced here rather than trusted to the caller, because the
    # failure mode is a filing that reads perfectly and is not true.
    description = body.get("description") or ""
    if description.strip():
        from core.sar_draft import check_grounding, narrative_sha
        case = body.get("evidence") or {}
        grounding = check_grounding(description, case)
        if not grounding["grounded"]:
            raise HTTPException(422, {
                "error": "narrative contains claims the case file does not support",
                "unsupported": grounding["unsupported"],
                "hint": "every amount, identifier and date in the narrative must appear in "
                        "`evidence`. Fix the narrative or supply the evidence it relies on.",
            })
        attester = (body.get("attested_by") or "").strip()
        if not attester:
            raise HTTPException(422, "attested_by is required: a SAR narrative is filed under "
                                     "a named human's signature, not an agent's")
        claimed = (body.get("narrative_sha") or "").strip()
        if claimed and claimed != narrative_sha(description):
            raise HTTPException(422, "narrative_sha does not match the description; the text "
                                     "changed after it was attested to")

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


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
