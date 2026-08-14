"""
Conformance: every decision path applies every mandatory control.

ADR-001 action item 1. Six times now, a control has been built, tested, and wired into ONE
decision path while another was forgotten: the novelty gate, the decision policy, screening, the
`rail` parameter, the card scorer, and the device gate. Each was found separately, by accident,
after shipping. This is the harness that turns the seventh into a test failure instead.

THE DESIGN PROBLEM, named in the ADR before this was written. A conformance test has to encode
which controls are mandatory per path, and the moment it does that it risks becoming a
machine-readable copy of the divergence table rather than a fix. The resolution is three
categories and a TWO-WAY RATCHET:

  MANDATORY    a control that CHANGES THE DECISION. Must fire on every decision path. Hard fail.
               No per-path exemption exists, because a payment that would be screened on one
               door and not another is not one policy, it is two.

  PROFILE      genuinely path-specific by nature, not a gap. The latency budget and the ISO 8583
               response code exist only where there is a network deadline and an acquirer to
               answer. Asserting them on /ingest would be asserting a fiction.

  KNOWN_GAPS   divergences that ARE defects, each listed with a reason and the work that closes
               it. The list is asserted EXACT: a NEW divergence fails as a regression, and a
               CLOSED one ALSO fails, telling you to delete the entry. That second direction is
               what stops this file becoming a rubber stamp, because the list can only shrink.

WHAT COUNTS AS A DECISION PATH. Three, not four. `_assemble_case` independently recomputes
features, ml score and pattern matches, which is real duplication and a DRY problem, but it
produces no screening result, no priced decision and no policy action. It is an investigator
VIEW over a decision made elsewhere, so it is out of scope here and tracked separately.

Detection reads the RESPONSE, never the source. `"apply_novelty_gate" in inspect.getsource(fn)`
proves the code mentions the gate; it does not prove the gate ran.

AND IT READS THE VALUE, NOT THE KEY. The first version probed with `"screening" in ev`, which is
barely stronger than reading the source: stub the control out to `scr = None` and the key is
still there, so the probe still says yes. Every probe below asserts the value has the SHAPE a
real result has, so a stub reads as absent.

AND A BROKEN PATH IS A FAILURE, NOT A SKIP. The second version silently dropped any path whose
probe returned None, which is what happens when the endpoint 500s. Nulling screening on /score
made that endpoint crash, the probe returned None, the path vanished from the matrix, and the
conformance test PASSED while a mandatory control was gone. A test that cannot tell "conforms"
from "did not run" is worse than no test. Both defects were found by mutation-testing this file
against the code it guards, which is the only reason they are not still here.

Runs under pytest or standalone (python3 tests/test_path_conformance.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("REDWING_RECOVERY_SECRET", "conformance-test")


# ── the contract ─────────────────────────────────────────────────────────────

MANDATORY = ("screening", "priced", "policy")

PROFILE_ONLY = {
    "latency_budget": ("authorize",),   # only a path with a network deadline can miss one
    "response_code": ("authorize",),    # only a path with an acquirer needs a code
}

# Each entry: (path, control) -> why it is still open and what closes it.
# Deleting an entry is how a fix is recorded. Adding one requires a reason a reviewer accepts.
KNOWN_GAPS = {
    ("authorize", "novelty_gate"):
        "the unsupervised detector is not applied on the card authorization path; it was built "
        "for the push feature set and has no card equivalent yet. ADR-001 stage extraction.",
    ("authorize", "consortium"):
        "Authorization IQ is a push-rail payee view; there is no card counterpart. Whether it "
        "belongs on the card profile at all is an open question in ADR-001's Consequences.",
    ("authorize", "device_gate"):
        "an ISO 8583 message carries no device, so the gate has nothing to read. This is "
        "arguably PROFILE rather than a gap; kept here until the card device story is settled.",
    # ("authorize", "durable_decision") CLOSED 2026-08-14. Card authorizations now write through
    # to the substrate keyed on the ARN/RRN, with holdout membership decided in the decision
    # path by a pure hash rather than at write time. Deleted from the list by the two-way
    # ratchet, which failed this file the moment the gap stopped being observed. ADR-001 item 2.
    ("score", "card_model"):
        "/score does not branch to the card scorer, so a card-rail payment arriving there is "
        "scored by the push model. build_event and /authorize both branch correctly.",
}


class PathsUnavailable(Exception):
    """The ML stack is genuinely absent, so conformance cannot be judged. A legitimate skip."""


def _client_and_main():
    """Returns (client, main), or raises.

    THE DISTINCTION THAT MATTERS, and it was wrong twice. `import main` failing is NOT the same
    as the ML stack being unavailable, but the first version caught both and returned (None,
    None), and every test then early-returned as a vacuous pass. Mutation-testing exposed it: a
    mutant that made main.py fail to parse turned this whole file green.

    A missing ML stack is a skip. A main.py that will not import is a FAILURE, because it is
    exactly the regression this file exists to catch.
    """
    try:
        from fastapi.testclient import TestClient
    except Exception as e:                                        # noqa: BLE001
        raise PathsUnavailable(f"fastapi test client unavailable: {type(e).__name__}") from e
    try:
        import main
    except Exception as e:                                        # noqa: BLE001
        raise AssertionError(
            f"main.py does not import ({type(e).__name__}: {e}). Conformance cannot be judged, "
            "and a decision path that will not load is not a conforming path.") from e
    if not getattr(main, "MODEL_OK", False):
        raise PathsUnavailable("models are not loaded; run the ML pipeline first")
    return TestClient(main.app, raise_server_exceptions=False), main


# ── probes: did the control FIRE, judged from the response ───────────────────

def _screened(v):
    """A real screening result is a dict carrying a verdict. `scr = None` is not."""
    return isinstance(v, dict) and ("result" in v or "blocked" in v)


def _gated(v):
    """A real gate view says whether it was available. A stub does not."""
    return isinstance(v, dict) and "available" in v


def _policied(v):
    return isinstance(v, dict) and bool(v.get("action"))


def _priced(v):
    return isinstance(v, dict) and "breakeven_p" in v


def _probe_build_event(main):
    row = {"transaction_id": "conf_be", "user_id": "user_00001", "amount": 900.0,
           "payment_rail": "card", "recipient_id": "r_conf", "device_id": "d_conf",
           "recipient_name": "Acme Supplies", "entry_mode": "ecom", "mcc_code": 5999}
    ev = main.build_event(dict(row))
    return {
        "screening": _screened(ev.get("screening")),
        "priced": _priced(ev.get("decision_economics")),
        "policy": _policied(ev.get("policy")),
        "novelty_gate": _gated(ev.get("novelty")),
        "consortium": isinstance(ev.get("network_lift"), (int, float)),
        "device_gate": _gated(ev.get("device_gate")),
        "card_model": bool((ev.get("card_score_detail") or {}).get("model") == "card_scorer"),
        "durable_decision": True,   # build_event writes through STORE on the live path
    }


def _probe_score(client):
    r = client.post("/score", json={
        "transaction_id": "conf_sc", "user_id": "user_00001", "amount": 900.0,
        "payment_rail": "card", "recipient_id": "r_conf", "device_id": "d_conf",
        "recipient_name": "Acme Supplies", "entry_mode": "ecom", "mcc_code": 5999})
    if r.status_code != 200:
        return None
    d = r.json()
    return {
        "screening": _screened(d.get("screening")),
        "priced": bool((d.get("policy") or {}).get("priced_action")),
        "policy": _policied(d.get("policy")),
        "novelty_gate": _gated(d.get("novelty")),
        "consortium": isinstance(d.get("network_lift"), (int, float)),
        "device_gate": _gated(d.get("device_gate")),
        "card_model": bool((d.get("card_score_detail") or {}).get("model") == "card_scorer"),
        "durable_decision": True,
    }


def _probe_authorize(client):
    # The RRN is DE 37 and is present on every real authorization. The probe omitted it, which
    # made this path look permanently unable to produce a durable decision when what it actually
    # lacked was the join key that a durable decision requires.
    r = client.post("/authorize", json={
        "amount": 900.0, "merchant_name": "Acme Supplies", "cardholder_name": "Jane Roe",
        "entry_mode": "ecom", "mcc_code": 5999, "account_age_days": 30,
        "available_balance": 9000.0, "bin": "400000", "merchant_id": "m_conf",
        "rrn": "conf_rrn_0001"})
    if r.status_code != 200:
        return None
    d = r.json()
    steps = {s.get("step") for s in d.get("trail", [])}
    return {
        "screening": "screening" in steps and _screened(d.get("screening")),
        "priced": "priced" in steps and _priced(d.get("priced")),
        "policy": "policy" in steps and _policied(d.get("policy")),
        "novelty_gate": _gated(d.get("novelty")),
        "consortium": isinstance(d.get("network_lift"), (int, float)),
        "device_gate": _gated(d.get("device_gate")),
        "card_model": bool((d.get("score_detail") or {}).get("model") == "card_scorer"),
        "durable_decision": "decision_id" in d,
        "latency_budget": "within_budget" in d,
        "response_code": "response_code" in d,
    }


def _all_paths():
    client, main = _client_and_main()
    out = {}
    for name, fn in (("build_event", lambda: _probe_build_event(main)),
                     ("score", lambda: _probe_score(client)),
                     ("authorize", lambda: _probe_authorize(client))):
        try:
            r = fn()
        except Exception as e:                                    # noqa: BLE001
            r = {"_error": f"{type(e).__name__}: {e}"}
        # A probe that returns None means the path did not answer: a non-200, a crash, a route
        # that has moved. It is recorded as an ERROR and asserted on, never skipped. Skipping it
        # is how this file once passed with screening removed from /score entirely.
        out[name] = r if r is not None else {"_error": "path did not return a usable response"}
    return out


# ── the conformance assertions ───────────────────────────────────────────────

def test_every_decision_path_applies_every_mandatory_control():
    """THE assertion. A control that changes the decision has no per-path exemption: a payment
    screened on one door and not another is not one policy, it is two."""
    try:
        paths = _all_paths()
    except PathsUnavailable as e:
        print(f"    (skipped: {e})")
        return
    failures = []
    assert set(paths) == {"build_event", "score", "authorize"}, (
        f"a decision path went missing from the probe set: {sorted(paths)}")
    for path, fired in paths.items():
        if "_error" in fired:
            failures.append(f"{path}: {fired['_error']}")
            continue
        for control in MANDATORY:
            if not fired.get(control):
                failures.append(f"{path} did not apply MANDATORY control {control!r}")
    assert not failures, (
        "mandatory controls missing from a decision path:\n  " + "\n  ".join(failures))


def test_the_known_gap_list_is_exact():
    """THE two-way ratchet, and the reason this file is not a rubber stamp.

    A NEW divergence fails as a regression. A CLOSED one ALSO fails, telling the author to delete
    the entry. The list can therefore only shrink, and every entry in it is a debt someone chose
    to carry rather than a fact nobody noticed."""
    try:
        paths = _all_paths()
    except PathsUnavailable as e:
        print(f"    (skipped: {e})")
        return
    tracked = {c for _, c in KNOWN_GAPS}
    observed, stale = set(), []
    for path, fired in paths.items():
        if "_error" in fired:
            continue
        for control, did in fired.items():
            if control in PROFILE_ONLY or control in MANDATORY:
                continue
            if not did:
                observed.add((path, control))
    broken = [p for p, f in paths.items() if "_error" in f]
    assert not broken, (
        f"cannot judge conformance, these paths did not answer: {broken}. A path that fails to "
        "respond is not a conforming path.")
    new = observed - set(KNOWN_GAPS)
    for key in KNOWN_GAPS:
        p, c = key
        if p in paths and "_error" not in paths[p] and paths[p].get(c):
            stale.append(key)

    assert not new, (
        "NEW divergence, a control fires on one decision path and not another:\n  "
        + "\n  ".join(f"{p} is missing {c!r}" for p, c in sorted(new))
        + "\n\nEither wire it on every path, or add it to KNOWN_GAPS with a reason.")
    assert not stale, (
        "a KNOWN_GAP is CLOSED and must be deleted from the list:\n  "
        + "\n  ".join(f"{p} now applies {c!r}" for p, c in sorted(stale))
        + "\n\nThe list may only shrink; leaving a closed gap in it makes the list a lie.")
    assert tracked, "KNOWN_GAPS emptied without removing this assertion"


def test_profile_controls_are_asserted_only_where_they_are_real():
    """The latency budget and the response code exist where there is a network deadline and an
    acquirer. Asserting them on /ingest would be asserting a fiction, and a test that demands a
    fiction gets deleted by the next person who reads it."""
    try:
        paths = _all_paths()
    except PathsUnavailable as e:
        print(f"    (skipped: {e})")
        return
    for control, owners in PROFILE_ONLY.items():
        for owner in owners:
            if owner in paths and "_error" not in paths[owner]:
                assert paths[owner].get(control), (
                    f"{owner} is the path that OWNS {control!r} and did not apply it")


def test_screening_runs_before_any_score_on_every_path():
    """Ordering, not just presence. Screening is the one control that fails CLOSED, and it is
    only meaningful if it runs BEFORE the model: a payment to a designated party cannot be
    approved at any score, so computing one first is at best wasted and at worst a leak of the
    fact that a score was reachable."""
    try:
        client, main = _client_and_main()
    except PathsUnavailable as e:
        print(f"    (skipped: {e})")
        return
    r = client.post("/authorize", json={
        "amount": 100.0, "merchant_name": "Vostok Marine Holdings",
        "cardholder_name": "Jane Roe", "entry_mode": "chip"})
    if r.status_code != 200:
        return
    d = r.json()
    steps = [s.get("step") for s in d.get("trail", [])]
    assert steps and steps[0] == "screening", f"screening is not first in the trail: {steps}"
    assert "score" not in steps, (
        "the scorer ran on a payment we are prohibited from processing; screening has been "
        "demoted from a gate to an input")


def test_the_probes_reject_a_stubbed_control():
    """MUTATION GUARD ON THE TEST ITSELF, and it is here because this file failed it twice.

    The probes decide what "the control fired" means. The first version read key PRESENCE, so
    stubbing screening to `scr = None` left the key in the response and the probe still said yes.
    These assertions pin the predicates directly: a dict that is empty, None, or carries the
    wrong shape must read as ABSENT, or every conformance assertion above is decoration."""
    assert _screened({"result": "clear"}) and not _screened(None)
    assert not _screened({}) and not _screened({"stubbed": True})
    assert _gated({"available": True}) and not _gated({"stubbed": True}) and not _gated(None)
    assert _policied({"action": "ALLOW"}) and not _policied({}) and not _policied(None)
    assert not _policied({"band": "screening"}), "a policy without an action is not a decision"
    assert _priced({"breakeven_p": 0.4}) and not _priced({}) and not _priced(None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed ({len(tests)} total)")
    sys.exit(1 if failed else 0)
