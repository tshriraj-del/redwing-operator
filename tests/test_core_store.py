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


def test_liability_prices_irrevocable_rails_higher_than_card():
    from core.liability import expected_liability, reimbursement_rate
    # same probability and amount: a Zelle pig-butchering scam should price far above
    # a card purchase, because the institution eats the APP-scam reimbursement
    zelle = expected_liability(0.9, 10000, "pig_butchering", "zelle")
    card  = expected_liability(0.9, 10000, "card_testing_bot", "card")
    assert zelle > card * 5                    # irrevocable + authorised dominates
    assert reimbursement_rate("pig_butchering", "zelle") > reimbursement_rate("", "card")
    assert expected_liability("bad", None, "x", "y") == 0.0     # never raises


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
