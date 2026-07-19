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
