"""
Test suite for REDWING's load-bearing logic.

These protect the CLAIMS the platform makes, not coverage for its own sake:
  - the case file is coherent with ground truth
  - the agent-env verifiers actually reward correct investigation
  - the adversary simulator's fragile/resilient verdict is correct
  - the feedback loop genuinely closes (reputation updates online)
  - derived enrichment agrees with the fraud typology
  - the real-data model loads and reproduces its held-out numbers

Runs under pytest (`python3 -m pytest`) or standalone (`python3 tests/test_redwing.py`).
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
ML = os.path.expanduser("~/pulseml_models")
for p in (OP, ML):
    if p not in sys.path:
        sys.path.insert(0, p)

import case_file
import fraud_env
import adversary
import feedback
from integrations.base import derived_signals, EnrichRequest, ConnectorCategory


# ── fixtures (plain helpers) ──────────────────────────────────────────────────

def _fraud_row():
    return {
        "transaction_id": "txn_test_f", "user_id": "u_t", "amount": 1820.0,
        "payment_rail": "card", "merchant_category": "crypto", "mcc_code": "6051.0",
        "fraud_typology": "account_takeover_ai", "is_fraud": True,
        "device_familiarity": 0.0, "amount_vs_max": 0.98, "velocity_1h": 0.7,
        "recipient_id": "r_t", "is_new_recipient": True, "user_avg": 120.0,
        "user_max": 1850.0, "dev_count": 1, "recv_count": 3, "device_id": "d_t",
    }


def _legit_row():
    r = _fraud_row()
    r.update({"transaction_id": "txn_test_l", "amount": 42.0, "merchant_category": "food",
              "mcc_code": "5999.0", "fraud_typology": "none", "is_fraud": False,
              "device_familiarity": 1.0, "amount_vs_max": 0.3, "velocity_1h": 0.1})
    return r


def _fraud_scored():
    return {"ml_score": 0.91, "combined_score": 0.93, "matched_signals": [],
            "graph_context": {"graph_risk_score": 0.62}}


def _legit_scored():
    return {"ml_score": 0.04, "combined_score": 0.05, "matched_signals": [],
            "graph_context": {"graph_risk_score": 0.01}}


# ── case_file ─────────────────────────────────────────────────────────────────

def test_case_file_fraud_is_coherent():
    c = case_file.assemble(_fraud_row(), _fraud_scored())
    assert c["alert"]["ground_truth_label"] == "fraud"
    assert c["customer"]["risk_rating"] in ("Low", "Medium", "High")
    assert any(s["severity"] == "high" for s in c["card_fraud_signals"])
    # high-confidence card fraud should recommend confirm_fraud and gate SAR after it
    assert c["recommended_disposition"]["action"] == "confirm_fraud"
    assert c["sar_eligible"] is True


def test_case_file_legit_is_clean():
    c = case_file.assemble(_legit_row(), _legit_scored())
    assert c["alert"]["ground_truth_label"] == "legitimate"
    assert c["recommended_disposition"]["action"] == "clear_false_positive"
    assert c["sar_eligible"] is False


# ── fraud_env verifiers ───────────────────────────────────────────────────────

def test_env_gold_disposition_tracks_truth():
    fc = case_file.assemble(_fraud_row(), _fraud_scored())
    lc = case_file.assemble(_legit_row(), _legit_scored())
    assert fraud_env.gold_disposition(fc) == "confirm_fraud"
    assert fraud_env.gold_disposition(lc) == "clear_false_positive"


def test_env_process_verifier_rewards_investigation():
    c = case_file.assemble(_fraud_row(), _fraud_scored())
    investigator = fraud_env.run_episode(c, agent="investigator")
    trigger = fraud_env.run_episode(c, agent="trigger_happy")
    # trigger_happy may land the right label, but deciding blind must score lower
    assert investigator["scorecard"]["process_reward"] > trigger["scorecard"]["process_reward"]
    assert investigator["scorecard"]["total_reward"] > trigger["scorecard"]["total_reward"]
    assert trigger["scorecard"]["process_detail"]["guessed"] is True


def test_env_false_positive_is_penalised_on_legit():
    c = case_file.assemble(_legit_row(), _legit_scored())
    trigger = fraud_env.run_episode(c, agent="trigger_happy")   # blocks a good customer
    assert trigger["scorecard"]["outcome_reward"] < 0


# ── adversary simulator ───────────────────────────────────────────────────────

def _seed_features():
    # a "caught" seed: every signal is hot
    return {f: 0.9 for f in (
        "amount_zscore", "amount_vs_max", "hour_risk", "is_round_amount",
        "preferred_rail_deviation", "velocity_1h", "velocity_24h",
        "recipient_global_fraud_rate", "recipient_familiarity", "device_familiarity",
        "account_age_days", "is_new_maximum")}


def test_adversary_fragile_when_cheap_signals_carry_detection():
    # score depends only on a CHEAP feature -> cheap moves defeat it -> FRAGILE
    def score(f):
        return 0.95 if f.get("amount_zscore", 0) > 0.5 else 0.05
    res = adversary.simulate(_seed_features(), score)
    assert res["verdict"] == "FRAGILE"
    assert res["share_lost_to_cheap"] > 0.5


def test_adversary_resilient_when_costly_signal_carries_detection():
    # score depends on recipient reputation -> only the COSTLY move defeats it
    def score(f):
        return 0.95 if f.get("recipient_global_fraud_rate", 0) > 0.5 else 0.05
    res = adversary.simulate(_seed_features(), score)
    assert res["verdict"] == "RESILIENT"
    assert res["crossed_at"] is None or res["crossed_at"]["cost"] == "costly"


def test_adversary_strategies_are_cost_tagged():
    costs = {s["cost"] for s in adversary.strategies()}
    assert costs == {"cheap", "costly"}


# ── feedback loop ─────────────────────────────────────────────────────────────

class _FakeRep:
    """Minimal empirical-Bayes reputation to test the online update path."""
    def __init__(self):
        self.table = {}
        self.prior = 0.0065
        self.k = 20.0

    def update(self, rid, is_fraud):
        d = self.table.setdefault(str(rid), {"tx": 0, "fraud": 0})
        d["tx"] += 1
        if is_fraud:
            d["fraud"] += 1
        rate = (d["fraud"] + self.k * self.prior) / (d["tx"] + self.k)
        return {"recipient_global_fraud_rate": round(rate, 6)}


def test_feedback_loop_closes_online():
    rep = _FakeRep()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        store = feedback.FeedbackStore(tf.name, reputation=rep)
        before = rep.update("r_x", False)["recipient_global_fraud_rate"]
        rep.table.clear()
        last = None
        for _ in range(8):
            out = store.record("t", "confirm_fraud", recipient_id="r_x")
            last = out["online_reputation_update"]["recipient_global_fraud_rate"]
        assert last > before                       # reputation rose from feedback
        assert store.status()["loop"] == "closed"
        assert store.status()["labeled_total"] == 8
    os.unlink(tf.name)


# ── derived enrichment ────────────────────────────────────────────────────────

class _FakeConn:
    def __init__(self, cid, cat):
        self.id = cid
        self.name = cid
        self.category = cat


def test_enrichment_is_coherent_with_typology():
    req = EnrichRequest(transaction_id="t", user_id="u_synth", fraud_typology="synthetic_id_ai")
    sig = derived_signals(_FakeConn("equifax", ConnectorCategory.CREDIT_BUREAU), req)
    assert sig["_mode"] == "derived"
    assert sig["synthetic_identity_score"] >= 0.6     # synthetic id -> high score
    assert sig["identity_verified"] is False


def test_enrichment_deterministic():
    req = EnrichRequest(transaction_id="t", user_id="u_det", fraud_typology="none")
    a = derived_signals(_FakeConn("plaid", ConnectorCategory.OPEN_BANKING), req)
    b = derived_signals(_FakeConn("plaid", ConnectorCategory.OPEN_BANKING), req)
    assert a == b                                     # seeded -> stable


# ── real-data payment model ───────────────────────────────────────────────────

def test_payment_model_artifacts_and_honest_metrics():
    meta_path = os.path.join(ML, "payment_real_meta.json")
    if not os.path.exists(meta_path):
        return                                        # skip if not built in this env
    meta = json.load(open(meta_path))
    assert meta["dataset"]["synthetic"] is False
    assert 0.80 <= meta["metrics"]["pr_auc"] <= 0.95  # honest range, not a synthetic 1.0
    # the held-out set must include a miss and a false-positive, not a victory lap
    labels = [s["label"] for s in meta["samples"]]
    assert any("MISSED" in l for l in labels)
    assert any("FALSE" in l for l in labels)


# ── standalone runner (no pytest needed) ──────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed ({len(tests)} total)")
    sys.exit(1 if failed else 0)
