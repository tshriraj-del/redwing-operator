"""
Tests for the core entity/event backbone (Phase 1, WS0).

These protect the substrate the whole ecosystem stands on:
  - entities upsert idempotently and merge reputation without losing history
  - events link to the entities they touch and are queryable both ways
  - the reputation write path (the closed loop's mechanism) actually persists
  - state survives a reopen (real durability, the due-diligence gap)
  - the CSV importer produces a coherent graph

Runs under pytest or standalone (python3 tests/test_core_store.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
if OP not in sys.path:
    sys.path.insert(0, OP)

from core.store import Store, Entity, Event, eid


def _fresh_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


def test_entity_upsert_is_idempotent_and_merges():
    s = Store(_fresh_db())
    s.upsert_entity(eid("recipient", "r1"), "recipient", institution_id="inst_a",
                    reputation={"tx": 1, "fraud": 0})
    s.upsert_entity(eid("recipient", "r1"), "recipient",
                    attributes={"note": "seen again"}, reputation={"tx": 2})
    e = s.get_entity(eid("recipient", "r1"))
    assert e.type == "recipient"
    assert e.institution_id == "inst_a"          # not clobbered by the empty second write
    assert e.reputation["tx"] == 2               # merged/overwritten key
    assert e.reputation["fraud"] == 0            # untouched key preserved
    assert e.attributes["note"] == "seen again"
    s.close()


def test_event_links_are_queryable_both_ways():
    s = Store(_fresh_db())
    u, r = eid("user", "u1"), eid("recipient", "r1")
    ev = s.append_event("transaction", entities=[u, r],
                        payload={"amount": 1820.0}, derived={"ml_score": 0.82})
    for_r = s.events_for_entity(r, event_type="transaction")
    assert len(for_r) == 1
    assert for_r[0].event_id == ev
    assert set(for_r[0].entities) == {u, r}
    assert for_r[0].payload["amount"] == 1820.0
    assert s.events_for_entity(u)[0].event_id == ev   # reachable from the user too
    s.close()


def test_reputation_update_persists():
    s = Store(_fresh_db())
    s.upsert_entity(eid("recipient", "r1"), "recipient", reputation={"tx": 3, "fraud": 0})
    out = s.update_reputation(eid("recipient", "r1"), {"fraud": 1, "fraud_rate": 0.31})
    assert out.reputation["fraud"] == 1
    assert out.reputation["fraud_rate"] == 0.31
    assert out.reputation["tx"] == 3                  # prior key kept
    assert s.update_reputation(eid("recipient", "missing"), {"x": 1}) is None
    s.close()


def test_state_survives_reopen():
    path = _fresh_db()
    s = Store(path)
    s.upsert_entity(eid("user", "u1"), "user")
    s.append_event("transaction", entities=[eid("user", "u1")], payload={"amount": 5})
    s.close()
    s2 = Store(path)                                   # cold reopen
    assert s2.get_entity(eid("user", "u1")) is not None
    assert s2.stats()["events_total"] == 1
    s2.close()


def test_bulk_paths_dedupe_and_link():
    s = Store(_fresh_db())
    ents = [Entity(eid("user", "u1"), "user"), Entity(eid("user", "u1"), "user"),
            Entity(eid("recipient", "r1"), "recipient", reputation={"tx": 5, "fraud": 2})]
    n = s.bulk_upsert_entities(ents)
    assert n == 3                                      # 3 rows sent
    assert s.stats()["entities_by_type"]["user"] == 1  # deduped to 1 by primary key
    evs = [Event("ev1", "transaction", "2026-01-01T00:00:00Z",
                 entities=[eid("user", "u1"), eid("recipient", "r1")])]
    s.bulk_append_events(evs)
    assert len(s.events_for_entity(eid("recipient", "r1"))) == 1
    s.close()


def test_importer_builds_coherent_graph():
    # Only runs if the synthetic ledger is present in this env.
    from core.store import _MODELS_DIR
    from core import seed_from_csv
    csv_path = _MODELS_DIR / "transactions.csv"
    if not csv_path.exists():
        return
    s = Store(_fresh_db())
    stats = seed_from_csv.seed(s, csv_path, limit=5000, institution_id="inst_a")
    assert stats["rows_read"] == 5000
    assert stats["events_written"] == 5000
    assert stats["entities_written"] > 0
    # every transaction event should reference at least a user
    ev = s.recent_events("transaction", limit=1)[0]
    assert any(x.startswith("user:") for x in ev.entities)
    s.close()


def test_record_scored_event_is_idempotent_and_creates_alert():
    from core.record import record_scored_event
    s = Store(_fresh_db())
    event = {"transaction_id": "txn_9", "amount": 1820.0, "rail": "zelle",
             "ml_score": 0.82, "combined_score": 0.88, "is_alert": True}
    row = {"user_id": "u1", "device_id": "d1", "recipient_id": "r1",
           "fraud_typology": "pig_butchering", "is_fraud": True, "institution_id": "inst_a"}
    ids1 = record_scored_event(s, event, row)
    ids2 = record_scored_event(s, event, row)      # re-score the same transaction
    assert set(ids1) == {eid("user", "u1"), eid("device", "d1"), eid("recipient", "r1")}
    st = s.stats()
    assert st["events_by_type"]["transaction"] == 1   # idempotent, not duplicated
    assert st["events_by_type"]["alert"] == 1         # alert event created
    # the transaction event carries the derived score + ground truth
    tev = s.recent_events("transaction", 1)[0]
    assert tev.derived["combined_score"] == 0.88
    assert tev.derived["is_fraud"] == 1
    assert tev.institution_id == "inst_a"
    s.close()


def test_record_scored_event_no_alert_when_clean():
    from core.record import record_scored_event
    s = Store(_fresh_db())
    record_scored_event(s, {"transaction_id": "t", "is_alert": False},
                        {"user_id": "u1", "is_fraud": False})
    st = s.stats()["events_by_type"]
    assert st.get("transaction") == 1
    assert "alert" not in st                            # no alert on a clean score
    s.close()


def test_record_scored_event_store_absent_is_noop():
    from core.record import record_scored_event
    assert record_scored_event(None, {"transaction_id": "t"}, {"user_id": "u"}) == []


def test_row_from_backbone_reopens_a_pipeline_ingested_transaction():
    """Regression: everything the ingestion pipeline brings in is scored and persisted to the
    backbone but never enters the historical dataset, so the case file could not open any of
    it. The row must be reconstructable from the backbone alone."""
    from core.record import record_scored_event, row_from_backbone
    s = Store(_fresh_db())
    event = {"transaction_id": "pipe_tx_1", "amount": 4200.0, "rail": "zelle",
             "ml_score": 0.81, "combined_score": 0.88, "is_alert": True,
             "expected_liability": 3500.0}
    row = {"user_id": "u77", "device_id": "d77", "recipient_id": "r77",
           "fraud_typology": "pig_butchering", "institution_id": "inst_a", "is_fraud": True}
    record_scored_event(s, event, row)

    back = row_from_backbone(s, "pipe_tx_1")
    assert back is not None
    assert back["transaction_id"] == "pipe_tx_1"
    assert back["amount"] == 4200.0 and back["payment_rail"] == "zelle"
    assert back["fraud_typology"] == "pig_butchering"
    assert back["user_id"] == "u77" and back["recipient_id"] == "r77" and back["device_id"] == "d77"

    assert row_from_backbone(s, "never_seen") is None      # unknown id stays unknown
    assert row_from_backbone(None, "pipe_tx_1") is None    # no store, no crash
    s.close()


def test_get_event_returns_none_for_missing():
    s = Store(_fresh_db())
    s.append_event("transaction", event_id="ev1", payload={"amount": 10})
    assert s.get_event("ev1").payload["amount"] == 10
    assert s.get_event("nope") is None
    s.close()


def test_close_loop_moves_reputation_and_returns_receipt():
    from core.record import record_scored_event
    from core.loop import close_loop
    s = Store(_fresh_db())
    # two prior payments to recipient r1 are on the backbone (the "pending" exposure)
    record_scored_event(s, {"transaction_id": "t1", "amount": 500.0, "is_alert": False},
                        {"user_id": "u1", "recipient_id": "r1", "is_fraud": False})
    record_scored_event(s, {"transaction_id": "t2", "amount": 1500.0, "is_alert": True},
                        {"user_id": "u2", "recipient_id": "r1", "is_fraud": True})
    before = s.get_entity(eid("recipient", "r1")).reputation

    receipt = close_loop(s, "t2", "r1", "confirm_fraud", is_fraud=True, rep_rate=0.42)

    # reputation moved in the durable store
    after = s.get_entity(eid("recipient", "r1")).reputation
    assert after["fraud"] == before.get("fraud", 0) + 1
    assert after["fraud_rate"] == 0.42
    # the loop left a durable, inspectable trail
    ev = s.stats()["events_by_type"]
    assert ev["disposition"] == 1 and ev["feedback"] == 1 and ev["model_update"] == 1
    # the receipt makes the compounding visible
    assert receipt["pending_payments"] == 2
    assert receipt["exposure_usd"] == 2000.0
    assert receipt["labels_queued"] == 1
    assert "model_update" in receipt["events_emitted"]
    s.close()


def test_close_loop_clear_label_queues_no_retrain_label():
    from core.loop import close_loop
    s = Store(_fresh_db())
    s.upsert_entity(eid("recipient", "r1"), "recipient", reputation={"tx": 3, "fraud": 0})
    receipt = close_loop(s, "t9", "r1", "clear_false_positive", is_fraud=False, rep_rate=0.004)
    after = s.get_entity(eid("recipient", "r1")).reputation
    assert after["tx"] == 4 and after["fraud"] == 0    # a legit observation still counts tx
    assert "model_update" in receipt["events_emitted"] # legit is still a training label
    s.close()


def test_gate_counts_an_adjudication_even_without_a_scored_decision():
    """Regression. readiness derived its gold count from training_rows(), which INNER JOINs
    decisions because training needs a feature snapshot. Readiness asks a different question -
    does a human judgement exist, and does it agree with the rule - and neither needs features.

    The effect was that an adjudication on a case this instance had not scored was stored,
    counted in labels_current, and invisible to the gate. An analyst could adjudicate all day
    and watch readiness never move. `trainable_gold` now reports the feature-carrying subset
    separately, which is the number that actually gates training."""
    from core.graduation import evaluate_target
    from core.loop import record_decision
    s = Store(_fresh_db())

    # (a) an adjudication with NO decision behind it: the analyst opened a case whose
    #     transaction was scored somewhere else, or before this instance existed
    s.add_label("intent", "motive", "coerced_victim", source="analyst",
                confidence=0.9, subject_ref="unscored_case")

    # (b) an adjudication WITH a decision, so it also carries features
    record_decision(s, "scored_case", module="motive", features={"x": 1},
                    decision_id="dec:scored_case")
    s.add_label("intent", "motive", "survival", source="analyst", confidence=0.9,
                decision_id="dec:scored_case", subject_ref="scored_case")

    r = evaluate_target(s, "intent", "motive")
    assert r["gold_labels"] == 2, "an adjudication without a decision was dropped by the gate"
    assert r["trainable_gold"] == 1, "only the feature-carrying label can train a model"
    s.close()


def test_labels_for_target_does_not_require_a_decision():
    s = Store(_fresh_db())
    s.add_label("intent", "motive", "survival", source="analyst", confidence=0.9,
                subject_ref="no_decision")
    rows = s.labels_for_target("intent", "motive", sources=["analyst"])
    assert len(rows) == 1
    assert rows[0]["subject_ref"] == "no_decision"
    assert rows[0]["trainable"] is False        # flagged, not hidden
    # heuristic labels are excluded when filtering to gold sources
    s.add_label("intent", "motive", "opportunistic", source="heuristic", confidence=0.3,
                subject_ref="no_decision_2")
    assert len(s.labels_for_target("intent", "motive", sources=["analyst"])) == 1
    s.close()


def test_liability_prices_irrevocable_rails_higher_than_card():
    from core.liability import expected_liability, reimbursement_rate
    # same probability and amount: a Zelle pig-butchering scam should price far above
    # a card purchase, because the institution eats the APP-scam reimbursement
    zelle = expected_liability(0.9, 10000, "pig_butchering", "zelle")
    card  = expected_liability(0.9, 10000, "card_testing_bot", "card")
    assert zelle > card * 5                    # irrevocable + authorised dominates
    assert reimbursement_rate("pig_butchering", "zelle") > reimbursement_rate("", "card")
    assert expected_liability("bad", None, "x", "y") == 0.0     # never raises


def test_false_positive_cost_is_dominated_by_attrition_not_the_payment():
    """The reason fraud orgs over-block is that they price the fraud loss and not the customer.
    The attrition term must be the big one, or the model is just a rounding error on margin."""
    from core.liability import false_positive_cost
    fp = false_positive_cost(200.0, action="BLOCK", ltv_band="high", account_age_days=2000)
    assert fp["attrition"] > fp["lost_margin"] * 10
    assert fp["total"] > 200.0                      # costs more than the payment is worth
    assert fp["assumptions"]["ltv"] > 0             # never quote a number without its inputs


def test_false_positive_cost_scales_with_how_hard_the_action_is():
    """A step-up is cheap friction; a hard decline can end the relationship."""
    from core.liability import false_positive_cost
    block = false_positive_cost(500, action="BLOCK", ltv_band="medium")["total"]
    hold = false_positive_cost(500, action="HOLD", ltv_band="medium")["total"]
    step = false_positive_cost(500, action="STEP_UP", ltv_band="medium")["total"]
    assert block > hold > step
    assert false_positive_cost(500, action="ALLOW")["attrition"] == 0.0


def test_the_card_rails_liability_is_the_published_one_not_a_guess():
    """It sat at 0.15 on the reasoning that card fraud is "largely chargeback- and
    network-protected". Directionally right, quantitatively wrong by half. The Fed's Regulation
    II report on 2023 puts the ISSUER's share of US debit fraud losses at 28.3%, with 49.9% on
    merchants. Understating it made blocking a card payment look cheaper than allowing one."""
    from core.liability import _RAIL_LIABILITY
    assert abs(_RAIL_LIABILITY["card"] - 0.283) < 0.001, (
        "the card liability drifted off the published figure")
    # Still the lowest-liability rail, which was the original point and remains true.
    assert _RAIL_LIABILITY["card"] < _RAIL_LIABILITY["ach"] < _RAIL_LIABILITY["zelle"]


def test_forgone_revenue_on_a_card_is_interchange_and_has_a_fixed_leg():
    """MUTATION GUARD. A flat margin rate says declining a $5 purchase costs a tenth of a cent,
    when interchange on it is eleven cents. Small-ticket declines are where a risk policy and
    the P&L disagree most, so flattening the fee hides exactly the disagreement this module
    exists to price."""
    from core.liability import forgone_revenue
    small = forgone_revenue(5.0, rail="card") / 5.0
    large = forgone_revenue(500.0, rail="card") / 500.0
    assert small > large * 1.5, (
        f"effective rate is {small:.4%} on $5 and {large:.4%} on $500; card revenue has "
        "collapsed back to a flat percentage")
    # Non-card rails genuinely are proportional, so they must stay that way.
    assert abs(forgone_revenue(5.0, rail="zelle") / 5.0
               - forgone_revenue(500.0, rail="zelle") / 500.0) < 1e-9


def test_the_card_tariff_reproduces_the_published_averages():
    """A constant fitted to a source that does not give the source back is a typo with a
    citation attached. Fed 2023, Durbin-exempt dual-message: $0.62 per transaction and 1.41% of
    value, which together pin the ticket they were measured at to $43.97."""
    from core.liability import forgone_revenue
    ticket = 0.62 / 0.0141
    fee = forgone_revenue(ticket, rail="card")
    assert abs(fee - 0.62) < 0.005, f"tariff gives ${fee:.4f} where the Fed published $0.62"
    assert abs(fee / ticket - 0.0141) < 0.0002


def test_the_rail_reaches_the_false_positive_side_and_not_only_the_fraud_side():
    """The recurring defect in this codebase: a parameter threaded into one path and forgotten
    on the other. Posture was dropped by breakeven_p and price_decision exactly this way. If
    `rail` does not reach false_positive_cost, a card decline is priced on a flat margin while
    the fraud side prices it as a card, and the two halves disagree silently."""
    from core.liability import false_positive_cost
    card = false_positive_cost(3.0, "DECLINE", "medium", 365, None, rail="card")
    other = false_positive_cost(3.0, "DECLINE", "medium", 365, None, rail="zelle")
    assert card["lost_margin"] > other["lost_margin"], (
        "rail is not reaching the false-positive side; the fixed interchange leg is missing")
    assert card["assumptions"]["revenue_model"] == "interchange, two-part"
    assert other["assumptions"]["revenue_model"] == "flat margin rate"


def test_a_decline_contract_prices_itself_as_a_card_because_that_is_all_it_handles():
    """decline_contract works entirely on ISO 8583 response codes, so there is no other rail it
    could be pricing. Leaving the default would have quietly valued every card decline on a flat
    margin."""
    from core.decline_contract import contract
    c = contract(decline_id="d1", member_id="m1", code="05", cause="issuer_risk",
                 amount=4.0, ltv_band="medium", account_age_days=365)
    assert c["cost_detail"]["assumptions"]["rail"] == "card", (
        "the rail parameter reaches the fraud side of the price but not the customer side")
    assert c["cost_detail"]["assumptions"]["revenue_model"] == "interchange, two-part"


def test_breakeven_threshold_moves_the_right_way():
    """The whole point of WS10: the bar is derived per transaction, not tuned globally."""
    from core.liability import breakeven_p
    # more exposure -> block on weaker suspicion
    assert breakeven_p(9000, rail="wire") < breakeven_p(40, rail="wire")
    # irrevocable rail -> block on weaker suspicion than chargeback-protected card
    assert breakeven_p(1000, rail="crypto") < breakeven_p(1000, rail="card")
    # more valuable customer -> demand more certainty before blocking them
    assert breakeven_p(1000, rail="wire", ltv_band="high") > breakeven_p(1000, rail="wire", ltv_band="low")
    # cheaper action -> lower bar to use it
    assert breakeven_p(1000, rail="wire", action="STEP_UP") < breakeven_p(1000, rail="wire", action="BLOCK")


def test_price_decision_refuses_a_block_that_destroys_more_value_than_it_saves():
    """A small card payment from a long-tenured, high-value customer should not be blocked on
    a coin-flip score. Preventing $10 of loss by risking $380 of customer damage is a bad
    trade even though it 'stopped fraud', and a one-sided objective would take it."""
    from core.liability import price_decision
    d = price_decision(0.55, 120, rail="card", ltv_band="high", account_age_days=2000)
    assert d["recommended_action"] == "ALLOW"
    assert d["cost_of_blocking"] > d["cost_of_allowing"]
    assert d["net_benefit_of_blocking"] < 0


def test_price_decision_blocks_a_large_irrevocable_payment_from_a_new_account():
    from core.liability import price_decision
    d = price_decision(0.55, 9000, rail="crypto", ltv_band="low", account_age_days=20)
    assert d["recommended_action"] == "BLOCK"
    assert d["cost_of_allowing"] > d["cost_of_blocking"]
    assert d["p_fraud"] > d["breakeven_p"]          # above the derived bar


def test_pricing_withdraws_its_recommendation_on_reconnaissance():
    """A card-testing probe is deliberately tiny. Pricing the decision on its own amount says
    'never block a $0.31 payment', which is exactly backwards: what blocking buys is stopping
    the card being confirmed for a later, larger hit. The economics are still reported, but the
    recommendation is withdrawn rather than being confidently wrong."""
    from core.liability import price_decision
    probe = price_decision(0.85, 0.31, typology="card_testing_bot", rail="card",
                           ltv_band="high", account_age_days=2000)
    assert probe["recommended_action"] == "DEFER_TO_DETECTION"
    assert "reconnaissance" in probe["caveat"]
    assert probe["cost_of_allowing"] >= 0            # still priced, just not acted on

    # an ordinary small card payment keeps its normal two-sided recommendation
    normal = price_decision(0.85, 0.31, typology="app_scam", rail="card",
                            ltv_band="high", account_age_days=2000)
    assert normal["recommended_action"] == "ALLOW"
    assert "caveat" not in normal


def test_pricing_never_raises_on_bad_input():
    """It sits in the hot decision path; a malformed row must not take scoring down."""
    from core.liability import breakeven_p, false_positive_cost, price_decision
    assert false_positive_cost("x", action="???", ltv_band="nope", account_age_days="y")["total"] >= 0
    assert 0.0 <= breakeven_p(None, rail="???") <= 1.0
    d = price_decision("bad", None, rail="")
    assert d["recommended_action"] in ("ALLOW", "BLOCK")


def test_liability_flows_into_receipt_and_portfolio():
    from core.record import record_scored_event
    from core.loop import close_loop
    s = Store(_fresh_db())
    # two scored payments to a mule, each carrying an expected_liability
    for i in range(2):
        record_scored_event(s, {"transaction_id": f"t{i}", "amount": 5000.0,
                                "is_alert": True, "expected_liability": 4200.0},
                            {"user_id": f"u{i}", "recipient_id": "m1", "is_fraud": True})
    receipt = close_loop(s, "t1", "m1", "confirm_fraud", is_fraud=True, rep_rate=0.4)
    assert receipt["liability_at_risk"] == 8400.0          # summed across the payee
    assert s.liability_at_risk("alert")["liability_at_risk"] == 8400.0
    s.close()


def test_scam_narrative_reads_the_con():
    from core.narrative import scam_narrative
    n = scam_narrative("pig_butchering", {"amount": 23464, "rail": "crypto",
                                          "is_new_recipient": True, "expected_liability": 20000})
    assert n["headline"] == "Authorized-push-payment scam"
    assert "crypto" in n["stage"]
    assert "pig butchering" in n["narrative"]
    assert any("irrevocable" in c for c in n["cues"])
    # a card-testing case reads differently
    assert scam_narrative("card_testing_bot", {"amount": 1.2, "rail": "card"})["headline"] == "Card testing"


def test_institution_assignment_deterministic_and_binary():
    from core.consortium import institution_of, INSTITUTIONS
    a = institution_of("user_123")
    assert a in INSTITUTIONS
    assert institution_of("user_123") == a                       # stable
    assert institution_of("user:user_123") == a                  # prefix-invariant (same customer)
    assert {institution_of(f"u{i}") for i in range(60)} == set(INSTITUTIONS)  # both used


def test_network_reveals_mule_hidden_from_victim_bank():
    from core.consortium import network_reveal
    # victim bank sees only clean outgoing payments; the off-ramp saw the fraud
    local = {
        "inst_neobank": {"tx": 12, "fraud": 0, "rate": 0.0065, "alerts": False, "name": "Northwind Neobank"},
        "inst_crypto":  {"tx": 8,  "fraud": 7, "rate": 0.30,   "alerts": True,  "name": "Coastline Exchange"},
    }
    r = network_reveal(local, "inst_neobank", epsilon=1.0, seed=3)
    assert r["local_view"]["alerts"] is False        # the victim bank saw nothing
    assert r["consortium_view"]["alerts"] is True    # the network flags the payee
    assert r["only_visible_via_network"] is True     # the whole point of the consortium


def test_network_no_false_reveal_when_all_clean():
    from core.consortium import network_reveal
    local = {
        "inst_neobank": {"tx": 20, "fraud": 0, "rate": 0.0065, "alerts": False, "name": "Northwind Neobank"},
        "inst_crypto":  {"tx": 15, "fraud": 0, "rate": 0.0065, "alerts": False, "name": "Coastline Exchange"},
    }
    r = network_reveal(local, "inst_neobank", epsilon=8.0, seed=3)   # low noise
    assert r["only_visible_via_network"] is False


def test_consortium_dp_is_deterministic_per_seed():
    from core.consortium import consortium_view
    local = {"a": {"tx": 10, "fraud": 4}, "b": {"tx": 6, "fraud": 1}}
    assert consortium_view(local, 1.0, seed=5) == consortium_view(local, 1.0, seed=5)


def test_fraud_graph_builds_rings_from_edges():
    from core.graph import build_fraud_graph
    edges = []
    # a mule ring: 5 fraud senders, 2 sharing one device
    for i in range(5):
        edges.append({"user": f"u{i}", "device": ("shared" if i < 2 else f"dev{i}"),
                      "recipient": "mule1", "is_fraud": 1, "amount": 3000.0, "typology": "mule_cashout"})
    # clean periphery
    for i in range(4):
        edges.append({"user": f"c{i}", "device": f"cd{i}", "recipient": f"legit{i}",
                      "is_fraud": 0, "amount": 50.0, "typology": "none"})
    g = build_fraud_graph(edges, rings=2, per_ring=20, clean=10)
    assert any(n["label"] == "mule1" and n.get("mule_flag") for n in g["nodes"])
    assert any(r["name"] == "mule1" and r["typology"] == "mule_cashout" for r in g["rings"])
    assert g["stats"]["shared_devices"] >= 1                      # the shared device
    assert g["stats"]["fraud_edges"] >= 5
    assert all("type" in n and "label" in n for n in g["nodes"])  # renderer contract


def test_consortium_index_finds_the_cross_bank_mule():
    from core.consortium import build_index, find_mules_in_index, views_from_counts, institution_of
    # build edges for one mule: victims (clean) at one bank, flagged cash-outs at another
    victims = [u for u in (f"vv{i}" for i in range(200)) if institution_of(u) == "inst_neobank"][:10]
    cashout = [u for u in (f"cc{i}" for i in range(200)) if institution_of(u) == "inst_crypto"][:9]
    edges = [("recipient:MULE", u, 0) for u in victims] + \
            [("recipient:MULE", u, 1 if i < 7 else 0) for i, u in enumerate(cashout)]
    idx = build_index(edges)
    lv = views_from_counts(idx["recipient:MULE"])
    assert lv["inst_neobank"]["alerts"] is False        # victim bank blind
    assert lv["inst_crypto"]["alerts"] is True          # off-ramp sees it
    mules = find_mules_in_index(idx, epsilon=1.0)
    assert any(m["recipient_id"] == "recipient:MULE" for m in mules)


def test_consortium_stays_silent_below_evidence_floor():
    from core.consortium import consortium_view, network_reveal, MIN_EVIDENCE_TX
    # a tiny-volume payee must never alert on DP noise (the ws2_demo_mule bug)
    local = {"inst_neobank": {"tx": 0, "fraud": 0, "alerts": False},
             "inst_crypto":  {"tx": 2, "fraud": 0, "alerts": False}}
    cons = consortium_view(local, epsilon=1.0, seed=0)
    assert cons["sufficient_evidence"] is False
    assert cons["alerts"] is False                  # silence, not a phantom alert
    assert network_reveal(local, "inst_neobank")["only_visible_via_network"] is False


def test_store_recipient_sender_labels_and_fraudy_filter():
    from core.record import record_scored_event
    from core.consortium import local_views
    s = Store(_fresh_db())
    for i in range(6):
        record_scored_event(s, {"transaction_id": f"t{i}", "amount": 1000.0, "is_alert": i >= 3},
                            {"user_id": f"payer{i}", "recipient_id": "mule1", "is_fraud": i >= 3})
    labels = s.recipient_sender_labels(eid("recipient", "mule1"))
    assert len(labels) == 6 and sum(fr for _, fr in labels) == 3
    assert sum(d["tx"] for d in local_views(labels).values()) == 6
    # fraudy_recipients reads seeded reputation counts (record_scored_event does not set them)
    s.upsert_entity(eid("recipient", "mule1"), "recipient", reputation={"tx": 6, "fraud": 3})
    found = s.fraudy_recipients(min_fraud=3, limit=10)
    assert any(rid == eid("recipient", "mule1") for rid, _f, _t in found)
    s.close()


def test_attribute_fabric_is_coherent_with_typology():
    from core.attributes import evaluate, TOTAL_SURFACE
    assert TOTAL_SURFACE > 1000                              # bureau-scale leaf surface
    synth = evaluate("u_synth", "synthetic_id_ai")
    assert synth["identity"]["Identity linkage"]["thin_file"] is True
    assert synth["identity_risk"] >= 0.6                     # identity tells fire
    card = evaluate("u_bot", "card_testing_bot")
    assert card["device"]["Integrity & tamper"]["headless_browser"] is True
    assert card["device_risk"] >= 0.6                        # device tells fire
    # a scam VICTIM looks clean on device+identity; the tell is behavioural
    victim = evaluate("u_victim", "pig_butchering")
    assert victim["device_risk"] < 0.5 and victim["identity_risk"] < 0.5
    assert any(s["risk"] >= 0.6 for s in synth["top_signals"])


def test_motive_protects_the_coerced_victim():
    from core.motive import assess_actor
    r = assess_actor({"duress": 0.9, "coaching_copresence": 0.8, "script_reading": 0.7})
    assert r["motive"]["motive"] == "coerced_victim" and r["motive"]["is_victim"] is True
    assert r["offender"]["lifecycle"] == "coerced_victim"
    assert r["intervention"]["posture"] == "VICTIM-PROTECT"
    assert r["intervention"]["reportable"] is False          # never punish the victim


def test_motive_separates_survival_from_professional():
    from core.motive import assess_actor
    survival = assess_actor({"survival_spend": 0.9, "benefit_timing": 0.7, "essential_category": 0.6})
    assert survival["motive"]["motive"] == "survival"
    assert survival["intervention"]["posture"] == "SUPPORT"
    assert survival["intervention"]["reportable"] is False

    pro = assess_actor({"professional_execution": 0.9, "shared_device_ring": 0.8, "sophisticated_tooling": 0.7})
    assert pro["motive"]["motive"] in ("income_source", "organized_malicious")
    assert pro["offender"]["lifecycle"] in ("professional", "ring_operator")
    assert pro["intervention"]["reportable"] is True         # durable enforcement


def test_motive_loophole_closes_the_gap():
    from core.motive import assess_actor
    r = assess_actor({"boundary_probing": 0.9, "threshold_walking": 0.7})
    assert r["motive"]["motive"] == "loophole"
    assert "CLOSE-GAP" in r["intervention"]["posture"]


def test_gauntlet_onboards_clean_but_gates_a_farm():
    from core.onboarding import assess_onboarding
    clean = assess_onboarding("applicant_clean", "")
    farm  = assess_onboarding("applicant_farm", "synthetic_id_ai",
                              behavior={"shared_cohort": 0.9, "scripted_timing": 0.7})
    assert clean["is_coerced"] is False
    assert clean["tier"] < farm["tier"]                            # the farm hits more friction
    assert farm["decision"] in ("STEP-UP", "MANUAL REVIEW", "DECLINE")
    assert any(c["dimension"] == "coordination" for c in farm["challenges"])  # farm-specific challenge


def test_gauntlet_targets_the_weak_dimension():
    from core.onboarding import assess_onboarding
    # fumbling one's own PII (reverse-familiarity) -> knowledge-based auth, not a blanket wall
    r = assess_onboarding("applicant_kba", "", behavior={"pii_hesitation": 0.85})
    assert any(c["dimension"] == "knowledge" for c in r["challenges"])
    assert r["decision"] in ("STEP-UP", "MANUAL REVIEW")


def test_gauntlet_protects_the_coerced_applicant():
    from core.onboarding import assess_onboarding
    r = assess_onboarding("applicant_coerced", "", behavior={"coaching_pauses": 0.8, "app_switching": 0.7})
    assert r["is_coerced"] is True
    assert r["decision"] == "PROTECT" and r["posture"] == "VICTIM-PROTECT"
    assert not r["challenges"]                                     # a victim is not challenged, but protected


def test_scam_arc_locates_the_grooming_stage_and_playbook():
    from core.scam_arc import locate_on_arc
    # a pig-butchering victim mid-escalation: online-only relationship, first crypto payee,
    # escalating transfers funded by a new loan
    r = locate_on_arc({"online_only_relationship": 0.9, "never_met_in_person": 0.8,
                       "first_payee_new_crypto": 0.8, "escalating_transfers": 0.8,
                       "source_of_funds_new": 0.7})
    assert r["on_arc"] is True
    assert r["stage"] >= 3                                         # at least escalation
    assert r["playbook"] == "romance_pig_butchering"


def test_scam_arc_break_glass_stops_a_coached_payment():
    from core.scam_arc import assess_scam
    # impersonation "safe account" scam with a live handler on the line, remote access open
    r = assess_scam({"authority_urgency": 0.8, "safe_account_move": 0.9,
                     "coaching_copresence": 0.9, "remote_access_active": 0.8, "duress": 0.7})
    assert r["live"]["coached"] is True and r["live"]["duress"] is True
    assert "VICTIM-PROTECT" in r["intervention"]["posture"]
    assert r["intervention"]["punitive"] is False                 # the victim is never punished
    # the break-glass steps tell the customer to disconnect remote access
    assert any("remote-access" in s or "remote access" in s for s in r["intervention"]["steps"])


def test_scam_arc_second_loss_guard_blocks_recovery_scam():
    from core.scam_arc import assess_scam
    r = assess_scam({"prior_victim": 0.9, "recovery_promise": 0.9, "fee_to_release": 0.8})
    assert r["arc"]["playbook"] == "recovery_scam"
    assert "second-loss" in r["intervention"]["posture"]
    assert r["intervention"]["reportable_as_scam"] is True


def test_scam_arc_educates_early_without_friction():
    from core.scam_arc import assess_scam
    # only contact/grooming signals, no money moving yet
    r = assess_scam({"unsolicited_contact": 0.7, "too_good_returns": 0.6})
    assert r["arc"]["money_moving"] is False
    assert r["intervention"]["posture"] == "EDUCATE"              # awareness, not a gate
    assert r["intervention"]["punitive"] is False


def test_scam_arc_leaves_a_normal_customer_alone():
    from core.scam_arc import assess_scam
    r = assess_scam({})
    assert r["arc"]["on_arc"] is False
    assert r["intervention"]["posture"] == "MONITOR"             # no friction on a normal customer


def test_mule_protects_the_unwitting_but_enforces_on_the_witting():
    from core.mule_network import assess_mule
    # tricked "payment-processing agent": believes it is a job, forwards everything, stops when warned
    unwitting = assess_mule({"job_ad_referral": 0.8, "believes_legitimate_job": 0.9,
                             "forwards_full_amount": 0.8, "stops_on_warning": 0.7,
                             "first_inbound_unrelated": 0.8, "rapid_passthrough": 0.7})
    assert unwitting["role"]["role"] == "unwitting"
    assert unwitting["action"]["person_action"]["posture"] == "PROTECT + EDUCATE"
    assert unwitting["action"]["person_action"]["reportable"] is False   # not criminalised
    assert unwitting["action"]["person_action"]["punitive"] is False

    # a paid repeat mule who carries on after a warning
    witting = assess_mule({"keeps_consistent_cut": 0.9, "continues_after_warning": 0.8,
                           "many_victim_sources": 0.7, "high_passthrough_ratio": 0.9,
                           "cashout_crypto": 0.8})
    assert witting["role"]["role"] == "witting"
    assert "SAR" in witting["action"]["person_action"]["posture"]
    assert witting["action"]["person_action"]["reportable"] is True


def test_mule_freezes_funds_for_the_upstream_victim_regardless_of_culpability():
    from core.mule_network import assess_mule
    # even an unwitting mule holds a real victim's money: the FUND action fires while money is at risk
    r = assess_mule({"believes_legitimate_job": 0.9, "forwards_full_amount": 0.8,
                     "first_inbound_unrelated": 0.9, "rapid_passthrough": 0.8})
    assert r["lifecycle"]["money_at_risk"] is True
    assert "recover" in r["action"]["fund_action"].lower()               # hold to recover for the victim
    assert r["action"]["person_action"]["punitive"] is False             # but the person is not punished


def test_mule_herder_triggers_network_action():
    from core.mule_network import assess_mule
    r = assess_mule({"recruits_others": 0.9, "controls_multiple_accounts": 0.8,
                     "shared_device_across_accounts": 0.8, "fanin_fanout_topology": 0.7,
                     "launder_language": 0.6})
    assert r["role"]["role"] == "herder"
    assert r["herd"]["herd_role"] == "controller"
    assert "LAW-ENFORCEMENT" in r["action"]["person_action"]["posture"]
    assert "herd" in r["action"]["herd_action"].lower()


def test_mule_romance_recruit_routes_to_victim_protection():
    from core.mule_network import assess_mule
    r = assess_mule({"romance_to_mule": 0.9, "believes_legitimate_job": 0.6,
                     "forwards_full_amount": 0.8, "stops_on_warning": 0.7,
                     "first_inbound_unrelated": 0.8})
    assert r["recruitment"]["channel"] == "romance"
    assert r["role"]["is_victim_adjacent"] is True
    assert r["action"]["route_to_victim_protection"] is True             # hand to scam_arc safeguarding


def test_mule_undetermined_holds_before_judging():
    from core.mule_network import assess_mule
    # mule-pattern movement but no evidence of intent either way
    r = assess_mule({"first_inbound_unrelated": 0.8, "rapid_passthrough": 0.8,
                     "high_passthrough_ratio": 0.8})
    assert r["role"]["role"] == "undetermined"
    assert r["action"]["person_action"]["posture"] == "HOLD + ESTABLISH-INTENT"
    assert r["action"]["person_action"]["punitive"] is False


def test_first_party_presumes_good_faith():
    from core.first_party import assess_first_party
    # a real merchant error, first dispute, cooperative: must stay genuine and be honoured
    r = assess_first_party({"merchant_error_evidence": 0.9, "first_dispute_ever": 0.8,
                            "cooperative_provides_evidence": 0.7})
    assert r["intent"]["intent"] == "genuine" and r["intent"]["presumed_genuine"] is True
    assert r["action"]["posture"] == "HONOUR"
    assert r["action"]["punitive"] is False


def test_first_party_separates_opportunist_serial_and_bustout():
    from core.first_party import assess_first_party
    opp = assess_first_party({"dispute_after_delivery_confirmed": 0.8, "moral_licensing_language": 0.8})
    assert opp["intent"]["intent"] == "opportunistic"
    assert "EDUCATE" in opp["action"]["posture"] and opp["action"]["reportable"] is False

    serial = assess_first_party({"serial_disputer": 0.9, "selective_high_value_disputes": 0.7,
                                 "prior_disputes_lost": 0.7})
    assert serial["intent"]["intent"] == "serial"
    assert "RESTRICT" in serial["action"]["posture"] and serial["action"]["reportable"] is True

    bust = assess_first_party({"credit_build_then_maxout": 0.9, "never_intended_to_repay": 0.8,
                               "contact_change_pre_default": 0.7})
    assert bust["intent"]["intent"] == "bust_out"
    assert "LOSS-MITIGATE" in bust["action"]["posture"]


def test_first_party_coached_routes_to_verification():
    from core.first_party import assess_first_party
    r = assess_first_party({"coached_by_third_party": 0.8, "coached_dispute_template": 0.7, "duress": 0.6})
    assert r["intent"]["intent"] == "coached"
    assert r["action"]["posture"] == "VERIFY-COERCION"
    assert r["action"]["punitive"] is False              # never punish before establishing intent


def test_vulnerability_scores_and_maps_exposure():
    from core.vulnerability import assess_vulnerability
    # lonely, recently widowed, with a fresh windfall: high risk, romance + investment exposure
    r = assess_vulnerability({"elderly": 0.8, "recent_bereavement": 0.9, "social_isolation": 0.8,
                              "recent_windfall": 0.8})
    assert r["profile"]["band"] in ("High", "Critical")
    exposed = [e["playbook"] for e in r["profile"]["top_exposures"]]
    assert "romance_pig_butchering" in exposed
    assert r["protections"]["posture"] == "PROTECT-PROACTIVELY"


def test_vulnerability_is_dignified_never_restrictive():
    from core.vulnerability import assess_vulnerability
    r = assess_vulnerability({"elderly": 0.9, "cognitive_decline_signals": 0.8})
    joined = " ".join(r["protections"]["protections"]).lower()
    # measures are supportive / opt-in, and the dignity rule forbids de-banking on vulnerability alone
    assert "opt-in" in r["protections"]["dignity_note"].lower()
    assert "de-bank" in r["protections"]["dignity_note"].lower()
    assert "restrict" not in joined or "do not restrict" in r["protections"]["dignity_note"].lower()


def test_vulnerability_leaves_a_low_risk_customer_alone():
    from core.vulnerability import assess_vulnerability
    r = assess_vulnerability({"financially_experienced": 0.8, "strong_support_network": 0.7})
    assert r["profile"]["band"] == "Low"
    assert r["protections"]["posture"] == "STANDARD-CARE"


def test_loophole_detects_synthesizes_and_closes_the_gap():
    from core.loophole import assess_loophole
    r = assess_loophole({"repeated_just_below_threshold": 0.9, "velocity_just_under_limit": 0.7})
    assert r["exploit"]["family"] == "threshold_arbitrage"
    assert r["action"]["control"]["control"]                       # a concrete closing control is proposed
    assert "CLOSE-GAP" in r["action"]["posture"]


def test_loophole_systemic_flips_from_punish_to_patch():
    from core.loophole import assess_loophole
    # many actors on the same edge: the policy is the failure, do not mass-punish
    r = assess_loophole({"welcome_bonus_multi_signup": 0.8, "multi_account_same_beneficiary": 0.8,
                         "population_prevalence": 0.9})
    assert r["systemic"]["systemic"] is True
    assert "systemic" in r["action"]["posture"].lower()
    assert r["action"]["reportable"] is False                      # not a thousand SARs; fix the policy


def test_substrate_logs_point_in_time_features_and_trains_uncensored():
    from core.loop import record_decision
    s = Store(_fresh_db())
    # an enforced BLOCK and a SHADOW decision (scored but not enforced): both must appear
    # in the training set, or the model only ever learns from what we let through.
    record_decision(s, "t_enf", entity_id=eid("user", "u1"), action="BLOCK", module="model",
                    score=0.91, features={"amount": 4200.0, "rail": "zelle", "device_risk": 0.8})
    record_decision(s, "t_shadow", entity_id=eid("user", "u2"), action="SHADOW", module="model",
                    score=0.88, shadow=True, features={"amount": 3900.0, "rail": "zelle"})
    # outcomes arrive later, keyed only by the transaction id
    s.add_label("outcome", "is_fraud", 1, source="confirmed_loss", confidence=1.0, subject_ref="t_enf")
    s.add_label("outcome", "is_fraud", 1, source="chargeback", confidence=0.9, subject_ref="t_shadow")

    rows = s.training_rows("outcome", "is_fraud")          # include_shadow=True by default
    assert len(rows) == 2
    # features are the point-in-time snapshot, not recomputed
    enf = next(r for r in rows if r["subject_ref"] == "t_enf")
    assert enf["features"]["amount"] == 4200.0 and enf["features"]["device_risk"] == 0.8
    # the shadow row is present (uncensored), and can be filtered out when needed
    assert any(r["shadow"] for r in rows)
    assert len(s.training_rows("outcome", "is_fraud", include_shadow=False)) == 1
    s.close()


def test_substrate_captures_two_label_spaces_on_disposition():
    from core.loop import record_decision, close_loop
    s = Store(_fresh_db())
    record_decision(s, "t7", entity_id=eid("user", "u7"), action="HOLD", module="scam_arc",
                    score=0.7, features={"amount": 8000.0})
    # analyst adjudicates: it was fraud (outcome) AND the person was a coerced victim (intent)
    receipt = close_loop(s, "t7", "r7", "confirm_fraud", is_fraud=True, rep_rate=0.5,
                         intent={"motive": "coerced_victim", "scam_stage": "extraction"})
    assert receipt["labeling"]["outcome_written"] is True
    assert set(receipt["labeling"]["intent_labels"]) == {"motive", "scam_stage"}

    cur = {(l.label_space, l.label_key): l for l in s.current_labels(subject_ref="t7")}
    assert cur[("outcome", "is_fraud")].label_value == "1"
    assert cur[("intent", "motive")].label_value == "coerced_victim"
    assert cur[("intent", "motive")].source == "analyst"       # human ground truth, high trust
    # the intent label links back to the point-in-time decision features
    intent_rows = s.training_rows("intent", "motive")
    assert intent_rows and intent_rows[0]["features"]["amount"] == 8000.0
    s.close()


def test_substrate_revises_a_label_without_losing_history():
    from core.loop import record_decision, close_loop
    s = Store(_fresh_db())
    record_decision(s, "t9", entity_id=eid("user", "u9"), action="BLOCK", module="model",
                    score=0.8, features={"amount": 200.0})
    # first adjudicated as fraud, later re-adjudicated as friendly-fraud (not fraud)
    close_loop(s, "t9", "r9", "confirm_fraud", is_fraud=True, rep_rate=0.3)
    close_loop(s, "t9", "r9", "reclassify_friendly", is_fraud=False, rep_rate=0.1)

    current = s.current_labels(subject_ref="t9")
    outcome = [l for l in current if l.label_key == "is_fraud"]
    assert len(outcome) == 1 and outcome[0].label_value == "0"   # only the latest is current
    # but the history is preserved (audit trail of the revision)
    hist = [l for l in s.label_history(subject_ref="t9") if l.label_key == "is_fraud"]
    assert len(hist) == 2
    # training uses only the current label
    rows = s.training_rows("outcome", "is_fraud")
    assert len(rows) == 1 and rows[0]["label"] == "0"
    s.close()


def test_substrate_gold_vs_silver_by_source_and_confidence():
    from core.loop import record_decision
    s = Store(_fresh_db())
    record_decision(s, "t_h", entity_id=eid("user", "uh"), module="motive",
                    features={"amount": 100.0},
                    heuristic_labels=[{"space": "intent", "key": "motive",
                                       "value": "survival", "confidence": 0.3}])
    # a human later confirms a different case with high confidence
    record_decision(s, "t_a", entity_id=eid("user", "ua"), module="motive",
                    features={"amount": 100.0})
    s.add_label("intent", "motive", "organized_malicious", source="analyst",
                confidence=0.9, subject_ref="t_a")

    silver = s.training_rows("intent", "motive")                          # everything
    gold   = s.training_rows("intent", "motive", sources=["analyst", "confirmed_loss"])
    assert len(silver) == 2 and len(gold) == 1
    assert gold[0]["source"] == "analyst"
    # the heuristic self-label is present but weak, so a min-confidence gate drops it
    assert len(s.training_rows("intent", "motive", min_confidence=0.5)) == 1

    stats = s.labeling_stats()
    assert stats["decisions_total"] == 2
    assert stats["labels_by_source"].get("heuristic") == 1
    s.close()


def test_substrate_separates_observed_from_censored_outcomes():
    from core.loop import record_decision
    s = Store(_fresh_db())
    # an ALLOWED case (outcome actually observed) and a BLOCKED case (label is an inference)
    record_decision(s, "obs1", action="ALLOW", module="model", features={"amount": 100.0})
    record_decision(s, "cen1", action="HOLD", module="model", features={"amount": 100.0})
    s.add_label("outcome", "is_fraud", 0, source="chargeback", confidence=0.9, subject_ref="obs1")
    s.add_label("outcome", "is_fraud", 1, source="analyst", confidence=0.9, subject_ref="cen1")

    stats = s.labeling_stats()
    assert stats["decisions_observed"] == 1 and stats["decisions_censored"] == 1

    allrows = s.training_rows("outcome", "is_fraud")
    obs = s.training_rows("outcome", "is_fraud", observed_only=True)
    assert len(allrows) == 2 and len(obs) == 1                       # reject inference drops the block
    assert obs[0]["subject_ref"] == "obs1" and obs[0]["observed"] is True
    assert all(r["subject_ref"] != "cen1" for r in obs)             # censored block excluded
    s.close()


def test_stream_publishes_and_consumes_fifo_with_ack():
    from core.stream import DurableQueue
    q = DurableQueue(_fresh_db())
    for i in range(3):
        q.publish("ingest", f"k{i}", {"n": i})
    seen = []
    res = q.consume_batch("ingest", lambda p: seen.append(p["n"]))
    assert seen == [0, 1, 2]                                 # FIFO by offset
    assert res["succeeded"] == 3
    assert q.stats("ingest")["done"] == 3 and q.stats("ingest")["ready"] == 0
    q.close()


def test_stream_publish_is_idempotent_by_key():
    from core.stream import DurableQueue
    q = DurableQueue(_fresh_db())
    assert q.publish("ingest", "dup", {"a": 1}) is not None
    assert q.publish("ingest", "dup", {"a": 1}) is None      # duplicate key ignored
    assert q.stats("ingest")["ready"] == 1                   # only one enqueued
    q.close()


def test_stream_retries_then_dead_letters_then_replays():
    from core.stream import DurableQueue
    q = DurableQueue(_fresh_db(), max_attempts=2)
    q.publish("ingest", "boom", {"x": 1})

    def always_fails(_):
        raise RuntimeError("scorer down")

    q.consume_batch("ingest", always_fails)                  # attempt 1 -> stays ready
    assert q.stats("ingest")["ready"] == 1
    q.consume_batch("ingest", always_fails)                  # attempt 2 -> dead-lettered
    assert q.stats("ingest")["dead"] == 1 and q.stats("ingest")["ready"] == 0
    dl = q.dead_letters("ingest")
    assert dl and "scorer down" in dl[0]["last_error"]

    assert q.replay("ingest") == 1                           # DLQ -> ready
    ok = []
    q.consume_batch("ingest", lambda p: ok.append(p["x"]))   # now processes cleanly
    assert ok == [1] and q.stats("ingest")["done"] == 1
    q.close()


def test_stream_backpressure_raises_instead_of_dropping():
    from core.stream import DurableQueue, BackpressureError
    q = DurableQueue(_fresh_db(), max_depth=2)
    q.publish("ingest", "a", {})
    q.publish("ingest", "b", {})
    try:
        q.publish("ingest", "c", {})
        assert False, "expected BackpressureError"
    except BackpressureError:
        pass
    q.close()


def test_stream_is_durable_across_reopen():
    from core.stream import DurableQueue
    path = _fresh_db()
    q = DurableQueue(path)
    q.publish("ingest", "persist", {"v": 9})
    q.close()
    q2 = DurableQueue(path)                                  # reopen the same file
    assert q2.stats("ingest")["ready"] == 1                  # the event survived the restart
    q2.close()


def _write_jsonl(path, rows):
    import json as _json
    with open(path, "w") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else _json.dumps(r)) + "\n")


def test_file_connector_pulls_validates_and_resumes():
    import json as _json
    from core.stream import DurableQueue
    from core.connectors import FileConnector
    d = tempfile.mkdtemp()
    path = os.path.join(d, "drop.jsonl")
    q = DurableQueue(os.path.join(d, "q.db"))
    s = Store(os.path.join(d, "cp.db"))
    _write_jsonl(path, [
        {"transaction_id": "a", "amount": 100, "user_id": "u1"},
        {"transaction_id": "b", "amount": 200, "user_id": "u2"},
        "{not valid json}",                                   # malformed -> rejected (visible)
        "",                                                   # blank -> skipped
    ])
    c = FileConnector("filedrop", q, s, path)
    r1 = c.poll()
    assert r1["published"] == 2 and r1["rejected"] == 1 and r1["skipped"] == 1
    assert q.stats("ingest")["ready"] == 2
    assert s.get_checkpoint("filedrop") == 4                  # consumed all four lines

    assert c.poll()["consumed"] == 0                          # nothing new: resumable no-op

    with open(path, "a") as f:                                # a new drop arrives
        f.write(_json.dumps({"transaction_id": "c", "amount": 300, "user_id": "u3"}) + "\n")
    r3 = c.poll()
    assert r3["published"] == 1 and s.get_checkpoint("filedrop") == 5
    assert q.stats("ingest")["ready"] == 3
    q.close(); s.close()


def test_file_connector_reread_is_idempotent():
    from core.stream import DurableQueue
    from core.connectors import FileConnector
    d = tempfile.mkdtemp()
    path = os.path.join(d, "drop.jsonl")
    q = DurableQueue(os.path.join(d, "q.db"))
    s = Store(os.path.join(d, "cp.db"))
    _write_jsonl(path, [{"transaction_id": "a", "amount": 100, "user_id": "u1"},
                        {"transaction_id": "b", "amount": 200, "user_id": "u2"}])
    c = FileConnector("filedrop", q, s, path)
    c.poll()
    assert q.stats("ingest")["ready"] == 2

    s.set_checkpoint("filedrop", 0)                           # simulate a lost checkpoint / replay
    r = c.poll()
    assert r["deduped"] == 2 and r["published"] == 0          # transport dedupe: no double-processing
    assert q.stats("ingest")["ready"] == 2                    # still two, not four
    q.close(); s.close()


def test_checkpoint_survives_store_reopen():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cp.db")
    s = Store(p)
    s.set_checkpoint("conn_x", 42)
    s.close()
    s2 = Store(p)
    assert s2.get_checkpoint("conn_x") == 42                  # durable across restart
    s2.close()


def _make_source_table(path, rows):
    import sqlite3 as _sql
    c = _sql.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS txns (id INTEGER PRIMARY KEY, txn_amt REAL, "
              "cust_id TEXT, txn_ref TEXT, rail TEXT)")
    c.executemany("INSERT INTO txns (id, txn_amt, cust_id, txn_ref, rail) VALUES (?,?,?,?,?)", rows)
    c.commit(); c.close()


_FMAP = {"txn_ref": "transaction_id", "txn_amt": "amount", "cust_id": "user_id", "rail": "payment_rail"}


def test_db_connector_polls_incrementally_by_watermark():
    from core.stream import DurableQueue
    from core.connectors import DBConnector
    d = tempfile.mkdtemp()
    src = os.path.join(d, "source.db")
    _make_source_table(src, [(1, 100.0, "c1", "r1", "zelle"), (2, 250.0, "c2", "r2", "ach")])
    q = DurableQueue(os.path.join(d, "q.db"))
    cp = Store(os.path.join(d, "cp.db"))
    c = DBConnector("core_txns", q, cp, src, "txns", id_column="id", field_map=_FMAP)

    r1 = c.poll()
    assert r1["published"] == 2
    assert cp.get_checkpoint("core_txns") == 2               # watermark = max id consumed
    assert q.stats("ingest")["ready"] == 2

    assert c.poll()["consumed"] == 0                          # no new rows: resumable no-op

    _make_source_table(src, [(3, 900.0, "c3", "r3", "wire"), (4, 12.0, "c4", "r4", "card")])
    r3 = c.poll()
    assert r3["published"] == 2 and cp.get_checkpoint("core_txns") == 4
    assert q.stats("ingest")["ready"] == 4
    q.close(); cp.close()


def test_db_connector_maps_source_schema_to_canonical():
    from core.stream import DurableQueue
    from core.connectors import DBConnector
    d = tempfile.mkdtemp()
    src = os.path.join(d, "source.db")
    _make_source_table(src, [(1, 100.0, "c1", "r1", "faster_payments")])
    q = DurableQueue(os.path.join(d, "q.db"))
    cp = Store(os.path.join(d, "cp.db"))
    DBConnector("core_txns", q, cp, src, "txns", id_column="id", field_map=_FMAP).poll()

    seen = []
    q.consume_batch("ingest", lambda p: seen.append(p))
    assert len(seen) == 1
    ev = seen[0]
    assert ev["amount"] == 100.0 and ev["user_id"] == "c1"    # source columns mapped to canonical
    assert ev["transaction_id"] == "r1" and ev["payment_rail"] == "fps"   # + schema-normalised
    q.close(); cp.close()


def test_db_connector_refuses_unsafe_identifiers():
    from core.stream import DurableQueue
    from core.connectors import DBConnector
    d = tempfile.mkdtemp()
    src = os.path.join(d, "source.db")
    _make_source_table(src, [(1, 100.0, "c1", "r1", "zelle")])
    q = DurableQueue(os.path.join(d, "q.db"))
    cp = Store(os.path.join(d, "cp.db"))
    c = DBConnector("bad", q, cp, src, "txns; DROP TABLE txns", id_column="id")
    assert c.poll()["consumed"] == 0                          # injection-y identifier -> no read, no crash
    q.close(); cp.close()


def test_webhook_signature_verification():
    from core.webhook import sign, verify_signature
    secret, body = "whsec_test", b'{"transaction_id":"w1","amount":100}'
    assert verify_signature(secret, body, sign(secret, body)) is True
    assert verify_signature("wrong_secret", body, sign(secret, body)) is False   # wrong key
    assert verify_signature(secret, b'{"transaction_id":"w1","amount":999}', sign(secret, body)) is False  # tampered
    assert verify_signature(secret, body, "") is False                            # missing sig fails closed


def test_webhook_receiver_authenticates_then_publishes():
    import json as _json
    from core.stream import DurableQueue
    from core.webhook import WebhookReceiver, sign
    q = DurableQueue(_fresh_db())
    rx = WebhookReceiver(q, {"demo_proc": "whsec_demo"})
    body = _json.dumps({"transaction_id": "wh1", "amount": 2500, "user_id": "u1"}).encode()

    ok = rx.accept("demo_proc", body, sign("whsec_demo", body))
    assert ok["accepted"] is True and ok["status"] == 202
    assert q.stats("ingest")["ready"] == 1                     # authenticated event reached the transport

    assert rx.accept("unknown_src", body, sign("whsec_demo", body))["status"] == 401   # unknown source
    assert rx.accept("demo_proc", body, "sha256=deadbeef")["status"] == 401            # bad signature
    assert rx.accept("demo_proc", b"not json", sign("whsec_demo", b"not json"))["status"] == 400  # bad JSON
    bad = _json.dumps({"transaction_id": "wh2", "amount": "N/A"}).encode()             # schema-invalid
    assert rx.accept("demo_proc", bad, sign("whsec_demo", bad))["status"] == 422
    assert q.stats("ingest")["ready"] == 1                     # only the one valid event was published
    q.close()


def test_ingest_schema_normalises_a_valid_event():
    from core.ingest_schema import validate_event
    v = validate_event({"transaction_id": "t1", "amount": "1500.5", "payment_rail": "Faster_Payments",
                        "user_id": "u1", "device_id": "d1"}, source="ingest")
    assert v["valid"] is True
    ev = v["event"]
    assert ev["amount"] == 1500.5                          # coerced + rounded
    assert ev["payment_rail"] == "fps"                     # synonym normalised
    assert ev["device_id"] == "d1"                         # passthrough preserved
    assert ev["_ingest"]["schema_version"] == "1.0"


def test_ingest_schema_rejects_silent_zero_corruption():
    from core.ingest_schema import validate_event
    # the whole point: a non-numeric amount is an error, not a silent 0
    bad = validate_event({"transaction_id": "t2", "amount": "N/A", "user_id": "u1"})
    assert bad["valid"] is False and bad["event"] is None
    assert any(e["field"] == "amount" for e in bad["errors"])
    # negative amount and no-amount-no-features are also rejected
    assert validate_event({"amount": -5, "user_id": "u1"})["valid"] is False
    assert validate_event({"user_id": "u1"})["valid"] is False
    # but precomputed features with no amount is allowed
    assert validate_event({"features": {"amount_zscore": 2.1}, "user_id": "u1"})["valid"] is True


def test_ingest_schema_warns_without_failing():
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 100, "rail": "quantum_rail"})   # unknown rail, no tid, no subject
    assert v["valid"] is True                                     # warnings never block a scoreable event
    fields = {w["field"] for w in v["warnings"]}
    assert "transaction_id" in fields and "payment_rail" in fields and "user_id" in fields
    assert v["event"]["transaction_id"].startswith("txn_")        # generated id


def test_ingest_schema_flags_label_leakage():
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 100, "user_id": "u1", "fraud_typology": "pig_butchering", "is_fraud": 1})
    assert v["valid"] is True
    assert set(v["event"]["_label_fields"]) == {"fraud_typology", "is_fraud"}
    assert any("leakage" in w["message"] for w in v["warnings"])   # do-not-feature warning


def test_ingest_schema_normalises_card_fields_through_the_same_module_authorize_uses():
    """Before this, `core/ingest_schema.py` knew "card" as a rail VALUE and nothing about the
    message: no BIN, entry mode, AVS, CVV, 3DS or token status. Card fields rode through as
    untyped passthrough on the general ingestion surface (POST /ingest, /ingest/batch, the
    stream), even while /authorize had a real normaliser. Two independent notions of "a valid
    card message" is the exact defect this codebase keeps producing."""
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 42.0, "user_id": "u1", "payment_rail": "card",
                        "entry_mode": "chip", "bin": "400000", "merchant_id": "m1",
                        "mcc_code": 5411, "tokenized": True})
    assert v["valid"] is True
    ev = v["event"]
    assert ev["entry_mode"] == "chip"
    assert ev["bin"] == "400000"
    assert ev["tokenized"] == 1
    assert "_card_quality" in ev


def test_ingest_schema_field_aliases_resolve_through_the_shared_normaliser():
    """A processor sending `avs` rather than `avs_result` must not silently produce an empty
    field: alias resolution lives in one place (core/card_message.py) so it cannot drift between
    the ingestion path and the authorization path."""
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 10.0, "user_id": "u1", "payment_rail": "card",
                        "entry_mode": "ecom", "avs": "no_match", "cvv2_result": "no_match"})
    ev = v["event"]
    assert ev["avs_result"] == "no_match"
    assert ev["cvv_result"] == "no_match"


def test_ingest_schema_warns_on_a_card_message_missing_expected_fields():
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 10.0, "user_id": "u1", "payment_rail": "card",
                        "entry_mode": "ecom"})   # no AVS/CVV/3DS on an ecom message
    ev = v["event"]
    assert ev["_card_quality"]["grade"] == "degraded"
    assert "avs_result" in ev["_card_quality"]["decisive_missing"]
    warned = {w["field"] for w in v["warnings"]}
    assert "avs_result" in warned and "cvv_result" in warned


def test_ingest_schema_does_not_flag_avs_missing_on_a_chip_message():
    """A chip transaction cannot carry AVS or CVV; that absence is correct, not a defect, and
    must not generate the same warning a genuine data-quality gap would. Only AVS/CVV/3DS are
    exempt on card-present, so this message still reports OTHER unsupplied fields (channel, MCC,
    ...) as missing; the assertion is specifically that the exempt ones are absent from both the
    missing list and the warnings, not that the message is complete."""
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 10.0, "user_id": "u1", "payment_rail": "card",
                        "entry_mode": "chip"})
    ev = v["event"]
    assert ev["_card_quality"]["grade"] != "degraded", (
        "AVS/CVV/3DS absence on a card-present message must not read as a degraded score")
    for f in ("avs_result", "cvv_result", "three_ds"):
        assert f not in ev["_card_quality"]["missing_expected"]
    warned = {w["field"] for w in v["warnings"]}
    assert "avs_result" not in warned and "cvv_result" not in warned


def test_ingest_schemas_own_amount_validation_is_not_overwritten_by_the_card_normaliser():
    """MUTATION GUARD, on the case that is actually reachable. A negative or non-numeric amount
    was the obvious worry, but it turns out already double-protected: this module short-circuits
    `event` to None whenever `errors` is non-empty, and the amount check runs and appends to
    `errors` BEFORE the card block, so a rejected amount never reaches the response regardless of
    what the card merge does to `ev`. Reverting the exclusion does not fail on that case.

    The real, always-reachable divergence is quieter: this module rounds amount to 2dp
    (`round(amt, 2)`) and card_message.normalise() does not (`max(0.0, float(amount))`). On
    every SUCCESSFUL card event, merging the card row's "amount" back in would silently replace
    the rounded value with the unrounded one - a precision drift with no error to catch it.
    """
    from core.ingest_schema import validate_event
    v = validate_event({"amount": "42.567", "user_id": "u1", "payment_rail": "card",
                        "entry_mode": "chip"})
    assert v["valid"] is True
    assert v["event"]["amount"] == 42.57, (
        f"got {v['event']['amount']!r}; the card normaliser's unrounded amount leaked back "
        "into the validated event")


def test_ingest_schema_leaves_non_card_rails_untouched_by_the_card_normaliser():
    from core.ingest_schema import validate_event
    v = validate_event({"amount": 10.0, "user_id": "u1", "payment_rail": "zelle"})
    assert "_card_quality" not in v["event"]


def test_the_ingest_contract_documents_card_fields():
    from core.ingest_schema import contract
    c = contract()
    assert "card_fields" in c
    assert "avs_result" in c["card_fields"]["categorical"]
    assert "bin" in c["card_fields"]["identifiers"]


def test_motive_is_inconclusive_on_weak_evidence():
    from core.motive import assess_actor
    # a whiff of hesitation must never escalate to enforcement
    r = assess_actor({"hesitation_entropy": 0.2})
    assert r["motive"]["motive"] == "inconclusive"
    assert r["intervention"]["posture"] == "MONITOR"
    assert r["intervention"]["reportable"] is False


def test_telemetry_round_trips_and_derives_reported_tells():
    from core.telemetry import derive_signals
    s = Store(_fresh_db())
    coached = {"call_active": True, "long_reads_before_field": 0.8,
               "remote_access_tool_active": True, "safe_account_narrative": True}
    s.record_telemetry("txA", coached, entity_id=eid("user", "uA"))
    assert s.get_telemetry("txA")["call_active"] is True          # durable round-trip
    assert s.get_telemetry("nope") == {}                          # absence is meaningful

    sig = derive_signals(coached)
    for tell in ("coaching_copresence", "script_reading", "remote_access_active", "safe_account_move"):
        assert sig.get(tell, 0) > 0                               # reported values became tells
    s.close()


def test_telemetry_actor_read_separates_victim_from_automated():
    from core.telemetry import assess_from_telemetry
    coached = assess_from_telemetry({"call_active": True, "long_reads_before_field": 0.8,
                                     "remote_access_tool_active": True, "safe_account_narrative": True})
    assert coached["actor"]["motive"]["motive"] == "coerced_victim"
    assert "VICTIM-PROTECT" in coached["victim"]["intervention"]["posture"]

    automated = assess_from_telemetry({"automation_framework": True, "headless": True,
                                       "action_cadence_regularity": 0.92, "ttfa_seconds": 1,
                                       "nav_path_directness": 0.95, "emulator": True})
    assert automated["actor"]["motive"]["motive"] in ("income_source", "organized_malicious")


def test_telemetry_is_silent_without_behaviour():
    from core.telemetry import assess_from_telemetry
    # no telemetry -> no actor read at all (motive cannot be inferred from nothing)
    empty = assess_from_telemetry({})
    assert empty["telemetry_present"] is False and empty["actor"] is None
    # benign telemetry -> a read, but inconclusive, never an enforcement call
    benign = assess_from_telemetry({"typing_hesitation": 0.1, "action_cadence_regularity": 0.15})
    assert benign["actor"]["motive"]["motive"] == "inconclusive"


def test_holdout_respects_ceiling_cap_and_determinism():
    from core.holdout import holdout_decision
    # protective / allow-like actions are never diverted, even at rate=1.0
    assert holdout_decision("s1", "PROTECT", 100.0, {"rate": 1.0})["release"] is False
    assert holdout_decision("s1", "ALLOW", 100.0, {"rate": 1.0})["release"] is False
    # the liability ceiling: a high-value case is always enforced, never gambled for data
    big = holdout_decision("s2", "BLOCK", 50000.0, {"rate": 1.0, "max_liability": 2000.0})
    assert big["release"] is False and "ceiling" in big["reason"]
    # an eligible low-liability case at rate=1.0 is released and monitored
    rel = holdout_decision("s3", "BLOCK", 100.0, {"rate": 1.0, "max_liability": 2000.0})
    assert rel["release"] is True and rel["enforced_action"] == "ALLOW" and rel["holdout"] is True
    # rate 0 releases nothing
    assert holdout_decision("s3", "BLOCK", 100.0, {"rate": 0.0})["release"] is False
    # deterministic: the same subject always resolves the same way (cannot retry into a release)
    assert holdout_decision("s7", "BLOCK", 100.0)["release"] == holdout_decision("s7", "BLOCK", 100.0)["release"]


def test_holdout_samples_roughly_the_configured_rate():
    from core.holdout import holdout_decision
    n = 2000
    released = sum(1 for i in range(n)
                   if holdout_decision(f"tx{i}", "BLOCK", 100.0, {"rate": 0.05})["release"])
    assert 0.03 <= released / n <= 0.07     # ~5% with sampling slack


def test_graduation_gates_on_gold_and_agreement():
    from core.loop import record_decision
    from core.graduation import evaluate_target
    s = Store(_fresh_db())

    # too little gold -> the gate refuses to graduate
    record_decision(s, "x1", module="motive",
                    heuristic_labels=[{"space": "intent", "key": "motive",
                                       "value": "survival", "confidence": 0.3}])
    s.add_label("intent", "motive", "survival", source="analyst", confidence=0.9, subject_ref="x1")
    assert evaluate_target(s, "intent", "motive")["verdict"] == "not_enough_gold"

    # 60 paired examples at ~80% agreement -> kappa in the useful band -> ready_to_train
    for i in range(60):
        subj = f"g{i}"
        gold = "survival" if i % 2 == 0 else "organized_malicious"
        wrong = i < 12
        heur = gold if not wrong else ("organized_malicious" if gold == "survival" else "survival")
        record_decision(s, subj, module="motive",
                        heuristic_labels=[{"space": "intent", "key": "motive",
                                           "value": heur, "confidence": 0.3}])
        s.add_label("intent", "motive", gold, source="analyst", confidence=0.9, subject_ref=subj)

    rep = evaluate_target(s, "intent", "motive")
    assert rep["gold_labels"] >= 50 and rep["paired_with_heuristic"] >= 50
    assert 0.5 <= rep["cohen_kappa"] <= 0.7          # chance-corrected agreement, not raw accuracy
    assert rep["verdict"] == "ready_to_train"
    s.close()


def test_trainer_beats_a_rule_that_ignores_a_feature():
    from core.seed_substrate import seed_labeled_cohort
    from core.train import train_target
    s = Store(_fresh_db())
    # synthetic cohort: gold depends on two signals, the rule looks at only one
    seed_labeled_cohort(s, n=300, seed=7)
    r = train_target(s, "intent", "motive")
    assert r["trained"] is True
    assert r["heuristic_accuracy"] is not None and r["heuristic_accuracy"] < 0.95   # the rule has a real gap
    assert r["model_accuracy"] >= r["heuristic_accuracy"]                            # the model recovers it
    assert r["beats_heuristic"] is True
    assert "graduate" in r["verdict"]
    s.close()


def test_trainer_refuses_without_enough_data():
    from core.train import train_target
    s = Store(_fresh_db())
    r = train_target(s, "intent", "motive")
    assert r["trained"] is False and "not enough" in r["reason"].lower()
    s.close()


def _seed_correlated_cohort(s, n=400, seed=7, per_copy_noise=0.05, signal_strength=0.7):
    """A cohort where one weakly-predictive signal is exposed as five near-identical copies -
    a velocity family. The label follows the signal only `signal_strength` of the time, so a
    calibrated model should be UNCERTAIN; NB, counting five copies as five votes, will not be."""
    import random
    from core.loop import record_decision
    rng = random.Random(seed)
    with s.batch():
        for i in range(n):
            burst = rng.random() < 0.5
            label = "fraud" if (burst == (rng.random() < signal_strength)) else "legit"
            feats = {}
            for v in ("velocity_1h", "velocity_4h", "velocity_24h", "velocity_7d", "velocity_30d"):
                flip = rng.random() < per_copy_noise
                feats[v] = 0.9 if (burst ^ flip) else 0.1
            record_decision(s, f"cc{i}", module="test", features=feats,
                            heuristic_labels=[{"space": "outcome", "key": "is_fraud",
                                               "value": label, "confidence": 0.3}],
                            decision_id=f"dec:cc{i}")
            s.add_label("outcome", "is_fraud", label, source="analyst", confidence=0.9,
                        decision_id=f"dec:cc{i}", subject_ref=f"cc{i}")


def test_logreg_is_better_calibrated_than_nb_on_correlated_features():
    """The whole reason for the classifier swap, made measurable. On five redundant copies of a
    weakly-predictive signal, accuracy is a near-tie but NB's log-loss is far worse: it counts
    the correlated evidence multiple times and is confidently wrong on the minority outcomes.
    Log-loss is the metric that sees this; accuracy cannot."""
    from core.train import compare_calibration
    s = Store(_fresh_db())
    _seed_correlated_cohort(s)
    cmp = compare_calibration(s, "outcome", "is_fraud", min_rows=40)
    assert cmp.get("naive_bayes") and cmp.get("logreg")
    # accuracy is a near tie (NB may even edge it), so this is not a story accuracy could tell
    assert abs(cmp["naive_bayes"]["accuracy"] - cmp["logreg"]["accuracy"]) < 0.1
    # but logreg's calibration is materially better
    assert cmp["logreg"]["log_loss"] < cmp["naive_bayes"]["log_loss"]
    assert cmp["log_loss_improvement"] > 0.1
    assert "overconfident" in cmp["reading"]
    s.close()


def test_compare_calibration_reports_no_gap_when_features_are_not_correlated():
    """The honesty guard: the comparison must not manufacture a difference. With an independent
    single signal, NB's assumption holds and there is no calibration story to tell."""
    import random
    from core.loop import record_decision
    from core.train import compare_calibration
    s = Store(_fresh_db())
    rng = random.Random(3)
    with s.batch():
        for i in range(300):
            hot = rng.random() < 0.5
            label = "fraud" if (hot == (rng.random() < 0.75)) else "legit"
            record_decision(s, f"ind{i}", module="test", features={"single_signal": 0.9 if hot else 0.1},
                            heuristic_labels=[{"space": "outcome", "key": "is_fraud",
                                               "value": label, "confidence": 0.3}],
                            decision_id=f"dec:ind{i}")
            s.add_label("outcome", "is_fraud", label, source="analyst", confidence=0.9,
                        decision_id=f"dec:ind{i}", subject_ref=f"ind{i}")
    cmp = compare_calibration(s, "outcome", "is_fraud", min_rows=40)
    assert cmp["log_loss_improvement"] < 0.1            # no meaningful gap to claim
    assert "no meaningful calibration gap" in cmp["reading"]
    s.close()


def test_train_target_defaults_to_logreg_and_reports_classifier_and_calibration():
    from core.train import train_target
    s = Store(_fresh_db())
    _seed_correlated_cohort(s)
    default = train_target(s, "outcome", "is_fraud", min_rows=40)
    assert default["classifier"] == "logreg"           # the new default
    assert "model_log_loss" in default                 # calibration is now reported

    nb = train_target(s, "outcome", "is_fraud", model="naive_bayes", min_rows=40)
    assert nb["classifier"] == "naive_bayes"           # opt back in to the old one
    s.close()


def test_graduation_flags_a_rule_that_already_matches_humans():
    from core.loop import record_decision
    from core.graduation import evaluate_target
    s = Store(_fresh_db())
    # near-perfect agreement: a model has little to add yet
    for i in range(60):
        subj = f"h{i}"
        gold = "survival" if i % 2 == 0 else "organized_malicious"
        heur = gold if i >= 2 else ("organized_malicious" if gold == "survival" else "survival")
        record_decision(s, subj, module="motive",
                        heuristic_labels=[{"space": "intent", "key": "motive",
                                           "value": heur, "confidence": 0.3}])
        s.add_label("intent", "motive", gold, source="analyst", confidence=0.9, subject_ref=subj)
    rep = evaluate_target(s, "intent", "motive")
    assert rep["cohen_kappa"] >= 0.9
    assert rep["verdict"] == "rule_already_strong"
    s.close()




# -- active learning: which case should the analyst adjudicate next -------------

def _seed_al_cohort(s, n=120, gold=0, seed=11):
    """n cases the heuristic has scored; `gold` of them already adjudicated."""
    import random
    from core.loop import record_decision
    rng = random.Random(seed)
    classes = ["survival", "opportunistic", "organized_malicious"]
    with s.batch():
        for i in range(n):
            h = classes[i % len(classes)]
            record_decision(s, f"al{i}", module="motive",
                            features={"a": rng.random(), "b": rng.random()},
                            heuristic_labels=[{"space": "intent", "key": "motive",
                                               "value": h, "confidence": 0.3}],
                            decision_id=f"dec:al{i}")
            if i < gold:
                s.add_label("intent", "motive", rng.choice(classes), source="analyst",
                            confidence=0.9, decision_id=f"dec:al{i}", subject_ref=f"al{i}")


def test_active_learning_only_queues_cases_that_can_form_a_pair():
    """The gate needs heuristic/gold PAIRS. A gold label on a case the heuristic never scored
    raises gold_labels and leaves paired untouched, so it cannot move the verdict. Candidates
    must therefore come only from cases carrying a heuristic prediction."""
    from core.active_learning import candidates
    from core.loop import record_decision
    s = Store(_fresh_db())
    # has a heuristic prediction -> queueable
    record_decision(s, "with_h", module="motive", features={"a": 1},
                    heuristic_labels=[{"space": "intent", "key": "motive",
                                       "value": "survival", "confidence": 0.3}],
                    decision_id="dec:with_h")
    # scored but the heuristic said nothing about motive -> not queueable
    record_decision(s, "no_h", module="model", features={"a": 1}, decision_id="dec:no_h")

    subs = {c["subject_ref"] for c in candidates(s, "intent", "motive")}
    assert subs == {"with_h"}
    s.close()


def test_active_learning_excludes_already_adjudicated_cases():
    """Re-asking a question a human has answered is the one purely wasted click.

    Usually add_label supersedes the heuristic label, so an adjudicated case drops out of the
    candidate pool on its own. That is NOT guaranteed: a subject can carry two decisions (this
    codebase uses both `dec:` and `replay:` id namespaces), and gold attached to the second does
    not supersede the heuristic on the first. This test builds exactly that case, so it exercises
    the explicit subject-level exclusion rather than passing for free on supersession.

    Verified by mutation: removing the `sr in gold` guard makes this fail."""
    from core.active_learning import candidates
    from core.loop import record_decision
    s = Store(_fresh_db())

    # heuristic on decision A
    record_decision(s, "sub1", module="motive", features={"a": 1},
                    heuristic_labels=[{"space": "intent", "key": "motive",
                                       "value": "survival", "confidence": 0.3}],
                    decision_id="dec:sub1")
    # a second decision for the SAME subject, and gold attaches to that one
    record_decision(s, "sub1", module="model", features={"a": 1}, decision_id="replay:sub1")
    s.add_label("intent", "motive", "organized_malicious", source="analyst", confidence=0.9,
                decision_id="replay:sub1", subject_ref="sub1")

    # the heuristic label is still CURRENT (supersession missed it) ...
    assert len(s.labels_for_target("intent", "motive", sources=["heuristic"])) == 1
    # ... but the subject must not be queued, because a human already answered
    assert [c["subject_ref"] for c in candidates(s, "intent", "motive")] == []

    # and the ordinary path (supersession) still drops adjudicated cases too
    s2 = Store(_fresh_db())
    _seed_al_cohort(s2, n=10, gold=4)
    subs = {c["subject_ref"] for c in candidates(s2, "intent", "motive")}
    assert len(subs) == 6 and not any(x in subs for x in ("al0", "al1", "al2", "al3"))
    s2.close()
    s.close()


def test_active_learning_cold_start_spreads_across_classes():
    """With too little gold to train, ranking by 'uncertainty' would rank noise. Breadth first:
    you cannot measure agreement on a class you have no examples of."""
    from core.active_learning import rank
    s = Store(_fresh_db())
    _seed_al_cohort(s, n=60, gold=0)
    r = rank(s, "intent", "motive", limit=6)
    assert r["phase"] == "cold_start"
    exploit = [q for q in r["queue"] if q["selection"] == "exploit"]
    # the first few exploit picks should cover distinct classes, not repeat one
    assert len({q["heuristic"] for q in exploit[:3]}) == 3
    assert "class coverage" in exploit[0]["why"]
    s.close()


def test_active_learning_switches_to_uncertainty_once_trainable():
    """With enough gold, rank by the model's entropy: the cases it is least sure about are the
    ones a human answer most changes."""
    from core.active_learning import rank
    s = Store(_fresh_db())
    _seed_al_cohort(s, n=120, gold=40)
    r = rank(s, "intent", "motive", limit=10)
    assert r["phase"] == "uncertainty"
    exploit = [q for q in r["queue"] if q["selection"] == "exploit"]
    assert len(exploit) >= 2
    scores = [q["uncertainty"] for q in exploit]
    assert scores == sorted(scores, reverse=True)          # most uncertain first
    assert all(0.0 <= x <= 1.0 for x in scores)            # normalised entropy
    s.close()


def test_active_learning_reserves_a_representative_slice():
    """Pure uncertainty sampling collapses the labelled set onto the decision boundary, which is
    the same censoring failure the holdout exists to prevent. A fixed share of the queue is
    drawn independently of the model."""
    from core.active_learning import rank
    s = Store(_fresh_db())
    _seed_al_cohort(s, n=120, gold=40)
    r = rank(s, "intent", "motive", limit=12, explore_frac=0.25)
    kinds = {q["selection"] for q in r["queue"]}
    assert "explore" in kinds and "exploit" in kinds
    ex = [q for q in r["queue"] if q["selection"] == "explore"]
    assert all(q["uncertainty"] is None for q in ex)       # not model-selected
    assert "representative" in ex[0]["why"]
    s.close()


def test_active_learning_stops_asking_once_the_gate_is_satisfied():
    from core.active_learning import rank
    s = Store(_fresh_db())
    _seed_al_cohort(s, n=200, gold=120)                    # well past MIN_GOLD / MIN_PAIRED
    r = rank(s, "intent", "motive", limit=5)
    assert r["phase"] == "satisfied"
    assert r["queue"] == []
    assert "cleared the gate" in r["reading"]
    s.close()


def test_next_questions_focuses_the_target_furthest_from_the_line():
    """Partial progress on four targets is worth less than one target actually crossing."""
    from core.active_learning import next_questions
    s = Store(_fresh_db())
    _seed_al_cohort(s, n=60, gold=0)                       # only motive has candidates
    r = next_questions(s, limit=5)
    assert r["focus"] == "intent.motive"
    assert r["queue"]
    assert "furthest" in r["reading"]
    s.close()


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
