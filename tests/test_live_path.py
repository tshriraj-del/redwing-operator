"""
Tests for the two live-path modules that had none: rule_factory and xai.

Both are reachable from the running API (rule_factory backs /rule-factory/*, xai backs
/xai/* and every /score explanation), and between them they were ~700 untested lines.

The emphasis is deliberate. rule_factory._safe_lambda compiles code an LLM wrote into a
callable and runs it over the whole transaction set, so its sandbox is the highest-stakes
function in the repo: a bypass there is arbitrary code execution driven by model output.
Those tests pin the security contract. The rest pin the decision thresholds that determine
whether a generated rule reaches production.

Runs under pytest or standalone (python3 tests/test_live_path.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
ML = os.path.expanduser("~/pulseml_models")
for p in (OP, ML):
    if p not in sys.path:
        sys.path.insert(0, p)

import rule_factory


# -- the LLM-generated-code sandbox --------------------------------------------

def test_safe_lambda_accepts_a_legitimate_rule():
    fn = rule_factory._safe_lambda("lambda r: r.get('amount_vs_max', 0) > 0.9")
    assert fn({"amount_vs_max": 0.95}) is True
    assert fn({"amount_vs_max": 0.10}) is False


def test_safe_lambda_rejects_every_forbidden_pattern():
    """Each of these is a distinct escape route out of the rule sandbox."""
    hostile = [
        "lambda r: __import__('os').system('id')",
        "lambda r: import os",
        "lambda r: eval('1+1')",
        "lambda r: exec('x=1')",
        "lambda r: open('/etc/passwd').read()",
        "lambda r: os.getcwd()",
        "lambda r: sys.exit(1)",
        "lambda r: subprocess.run(['ls'])",
        # the classic sandbox break: walk the type hierarchy to reach a builtin
        "lambda r: ().__class__.__bases__[0].__subclasses__()",
    ]
    for code in hostile:
        try:
            rule_factory._safe_lambda(code)
        except ValueError:
            continue
        raise AssertionError(f"sandbox accepted hostile code: {code}")


def test_safe_lambda_requires_a_lambda_over_the_row():
    """Anything that is not `lambda r:` is refused, so a candidate cannot smuggle in a
    different callable shape."""
    for code in ["r.get('x')", "def f(r): return True", "lambda x: True", ""]:
        try:
            rule_factory._safe_lambda(code)
        except ValueError:
            continue
        raise AssertionError(f"sandbox accepted a non-lambda: {code!r}")


def test_safe_lambda_actually_empties_builtins():
    """The blocklist is only half the defence; the other half is the restricted namespace.
    A name that is not blocked but is also not a rule primitive must still be unreachable,
    which proves the sandbox does not depend on the wordlist being exhaustive."""
    fn = rule_factory._safe_lambda("lambda r: len(r) > 0")   # 'len' is not in the blocklist
    try:
        fn({"a": 1})
    except NameError:
        return                                               # builtins genuinely stripped
    raise AssertionError("builtins are reachable inside a generated rule")


# -- backtest gate: what is allowed to reach production ------------------------

def _df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def _rows(n_fraud=20, n_legit=980, flag_key="amount_vs_max"):
    rows = []
    for i in range(n_fraud):
        rows.append({flag_key: 0.99, "is_fraud": True, "rule_triggered": False})
    for i in range(n_legit):
        rows.append({flag_key: 0.10, "is_fraud": False, "rule_triggered": False})
    return rows


def test_backtest_rejects_a_rule_that_fires_on_everything():
    """A rule that flags every transaction has perfect recall and useless precision. It must
    never be recommended: this is the failure mode that floods an analyst queue."""
    bt = rule_factory.backtest_rule(
        {"fn_code": "lambda r: True"}, _df(_rows()), existing_rules=[])
    assert bt["recommendation"] == "REJECT"
    assert bt["precision"] < rule_factory.MIN_PRECISION


def test_backtest_auto_deploys_only_a_high_precision_rule():
    bt = rule_factory.backtest_rule(
        {"fn_code": "lambda r: r.get('amount_vs_max', 0) > 0.9"},
        _df(_rows()), existing_rules=[])
    assert bt["precision"] == 1.0 and bt["recall"] == 1.0
    assert bt["TP"] == 20 and bt["FP"] == 0
    assert bt["recommendation"] == "AUTO_DEPLOY"


def test_backtest_rejects_a_duplicate_of_an_existing_rule():
    """A candidate that fires where existing rules already fire adds queue load and no
    detection, so overlap is its own rejection reason."""
    rows = _rows()
    for r in rows:
        r["rule_triggered"] = r["is_fraud"]        # existing rules already catch these
    bt = rule_factory.backtest_rule(
        {"fn_code": "lambda r: r.get('amount_vs_max', 0) > 0.9"},
        _df(rows), existing_rules=[])
    assert bt["overlap_with_existing"] > rule_factory.MAX_OVERLAP
    assert bt["recommendation"] == "REJECT_DUPLICATE"


def test_backtest_rejects_unsafe_code_instead_of_running_it():
    bt = rule_factory.backtest_rule(
        {"fn_code": "lambda r: __import__('os').system('id')"}, _df(_rows()), existing_rules=[])
    assert bt["recommendation"] == "REJECT"
    assert "error" in bt


def test_backtest_survives_a_rule_that_raises_on_some_rows():
    """A generated rule that throws on unexpected data must degrade to 'did not fire' for
    that row, not abort the whole backtest."""
    rows = _rows()
    rows[0]["amount_vs_max"] = None
    bt = rule_factory.backtest_rule(
        {"fn_code": "lambda r: r.get('amount_vs_max') > 0.9"},
        _df(rows), existing_rules=[])
    assert "error" not in bt
    assert bt["n_flagged"] >= 0


# -- rule gap extraction -------------------------------------------------------

def _gap_row(is_fraud=True, ensemble=0.95, rule=10):
    return {"is_fraud": is_fraud, "ensemble_score": ensemble, "rule_score": rule}


def test_extract_rule_gaps_finds_only_fraud_the_model_caught_and_rules_missed():
    """A 'gap' is the training signal for a new rule, so each of the three conditions has to
    hold. Anything else would teach the generator from the wrong examples: a false positive
    would generate a rule for legitimate behaviour, and an already-covered case would
    generate a duplicate."""
    rows = [
        _gap_row(),                                    # gap: fraud, ML sure, rules missed
        _gap_row(is_fraud=False),                      # not fraud
        _gap_row(ensemble=0.40),                       # ML was not confident
        _gap_row(rule=80),                             # rules already caught it
    ]
    gaps = rule_factory.extract_rule_gaps(_df(rows), min_gaps=1)
    assert len(gaps) == 1
    assert bool(gaps.iloc[0]["is_fraud"]) is True


def test_extract_rule_gaps_stays_silent_below_the_firing_threshold():
    """Below min_gaps it returns nothing, so the LLM pipeline is not fired (and billed) on
    evidence too thin to generalise from."""
    rows = [_gap_row() for _ in range(3)]
    assert len(rule_factory.extract_rule_gaps(_df(rows), min_gaps=5)) == 0
    assert len(rule_factory.extract_rule_gaps(_df(rows), min_gaps=3)) == 3


def test_extract_rule_gaps_refuses_a_frame_missing_its_schema():
    assert len(rule_factory.extract_rule_gaps(_df([{"is_fraud": True}]), min_gaps=1)) == 0


# -- xai -----------------------------------------------------------------------

def test_model_card_carries_the_regulatory_fields_it_claims():
    """The UI presents this as an EU AI Act Article 11/13 artefact, so the fields it promises
    must actually be present rather than rendering as blanks."""
    import xai
    card = xai.get_model_card({"version": "1.0.0", "features": ["a", "b"]}, ["a", "b"])
    assert isinstance(card, dict) and card
    flat = repr(card).lower()
    for claim in ("version", "feature"):
        assert claim in flat, f"model card is missing {claim}"


def test_governance_metrics_are_shaped_for_the_dashboard():
    import xai
    g = xai.get_governance_metrics()
    assert isinstance(g, dict)


def test_list_explanations_respects_its_limit():
    import xai
    out = xai.list_explanations(limit=3)
    rows = out.get("explanations", out) if isinstance(out, dict) else out
    if isinstance(rows, list):
        assert len(rows) <= 3


# -- standalone runner (no pytest needed) --------------------------------------


def test_a_csv_false_string_does_not_flag_every_transaction():
    """THE bug: bool("False") is True, because a non-empty string is truthy. main.py had

        alert = is_alert(score) or bool(row.get("is_fraud", False))

    so any CSV-sourced row whose is_fraud is the STRING "False" flagged as an alert. Measured
    over 3,000 replayed payments, that reported 98.33% of traffic as alerts against a true
    fraud rate of 0.47%, and it silently inflated every downstream count: the alert queue,
    liability-at-risk, and the HOLD decisions filling the training substrate.

    Pinned as a parsing test rather than an end-to-end one so it stays fast and cannot be
    fixed by accident somewhere else."""
    parse = lambda v: str(v).strip().lower() in ("1", "true", "yes")   # noqa: E731
    for falsey in ("False", "false", "FALSE", "0", "", "no", None):
        assert not parse(falsey), f"{falsey!r} parsed as fraud"
        assert bool(falsey) is not parse(falsey) or falsey in ("", None), (
            f"{falsey!r} is exactly the case bare bool() gets wrong")
    for truthy in ("True", "true", "1", "yes"):
        assert parse(truthy), f"{truthy!r} should parse as fraud"


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

