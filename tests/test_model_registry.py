"""
Tests for the model registry.

Why this file exists. Before it, six `pickle.load` calls sat in main.py with their own glue and
their own idea of what a missing file meant, and `decisions.model_version` was set on 0 of 692
rows. That is survivable with two models and not with five.

Model-risk guidance asks for an inventory, risk tiering, lifecycle monitoring and effective
challenge. The failure named repeatedly in practice is models running OUTSIDE the registry,
where none of that applies, so the tests here are about enforcement rather than bookkeeping:

  a CHALLENGER MUST NOT be able to decide, even when a caller asks for it. Effective challenge
  is worth nothing if the challenger can quietly become the decision.

  a CONTRACT MISMATCH must refuse the load. This generalises the guard that caught an isolation
  forest fitted on 23 features while the model had moved to 32.

  a VERSION must track content. A hand-maintained string drifts on the first retrain somebody
  forgets to bump, and the decision would be attributed to a model that never scored it.

Runs under pytest or standalone (python3 tests/test_model_registry.py).
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import model_registry as M  # noqa: E402


class _Model:
    def __init__(self, n=32):
        self.n_features_in_ = n

    def predict(self, x):
        return 0.5


def _reg():
    return M.Registry()


def _spec(mid="m", state=M.SHADOW, tier=M.TIER_1, feats=None, artifact=None):
    return M.ModelSpec(model_id=mid, purpose="score", tier=tier,
                       features=feats if feats is not None else [f"f{i}" for i in range(32)],
                       state=state, artifact=artifact)


# ---------------------------------------------------------------- effective challenge

def test_a_challenger_cannot_decide_even_when_asked_for_a_decision():
    """THE enforcement. A challenger that can be handed a decision because a caller passed the
    wrong flag is not a challenger, it is a second champion nobody approved."""
    r = _reg()
    r.register(_spec("chal", state=M.CHALLENGER))
    r.load("chal", lambda: _Model())
    assert r.get("chal") is not None, "a challenger should still be usable for scoring"
    assert r.get("chal", for_decision=True) is None, "a challenger decided a case"
    assert r.may_decide("chal") is False


def test_a_champion_may_decide():
    r = _reg()
    r.register(_spec("champ", state=M.CHAMPION))
    r.load("champ", lambda: _Model())
    assert r.get("champ", for_decision=True) is not None
    assert r.may_decide("champ") is True


def test_challengers_still_score_so_there_is_something_to_compare():
    """Effective challenge needs the challenger to see the same traffic. If it were excluded
    from scoring there would be nothing to compare and the lifecycle would be decoration."""
    r = _reg()
    r.register(_spec("champ", state=M.CHAMPION))
    r.register(_spec("chal", state=M.CHALLENGER))
    r.register(_spec("old", state=M.RETIRED))
    for mid in ("champ", "chal", "old"):
        r.load(mid, lambda: _Model())
    scoring = {s.model_id for s in r.scoring_models()}
    assert scoring == {"champ", "chal"}, f"retired models must not score: {scoring}"


def test_promoting_a_challenger_retires_the_incumbent():
    """Two champions for one purpose is the state in which nobody can say which one decided."""
    r = _reg()
    r.register(_spec("champ", state=M.CHAMPION))
    r.register(_spec("chal", state=M.CHALLENGER))
    for mid in ("champ", "chal"):
        r.load(mid, lambda: _Model())
    r.promote("chal", M.CHAMPION)
    assert r.may_decide("chal") is True
    assert r.may_decide("champ") is False
    assert sum(1 for s in r.scoring_models() if s.state == M.CHAMPION) == 1


# --------------------------------------------------------------------- the contract

def test_a_contract_mismatch_refuses_the_load_and_says_why():
    """The generalisation of the isolation-forest bug: an artifact fitted on 23 features while
    the live set is 32 would score a feature space that no longer means what it did, silently,
    on every payment."""
    r = _reg()
    r.register(_spec("stale", feats=[f"f{i}" for i in range(32)]))
    spec = r.load("stale", lambda: _Model(n=23))
    assert spec.loaded is False
    assert "23 features" in spec.error and "32" in spec.error
    assert r.get("stale") is None


def test_a_model_that_does_not_advertise_its_arity_is_allowed():
    """sklearn sets n_features_in_; plenty of custom models do not. Refusing everything
    unlabelled would make the registry unusable for exactly the models it exists to govern."""
    class _Opaque:
        pass
    r = _reg()
    r.register(_spec("custom"))
    assert r.load("custom", lambda: _Opaque()).loaded is True


def test_a_loader_that_raises_is_recorded_not_propagated():
    """This runs at startup on the money path. A missing artifact must cost one model, not the
    process."""
    r = _reg()
    r.register(_spec("broken"))
    spec = r.load("broken", lambda: (_ for _ in ()).throw(FileNotFoundError("gone")))
    assert spec.loaded is False and "FileNotFoundError" in spec.error
    assert r.get("broken") is None


# --------------------------------------------------------------------- versioning

def test_the_version_is_a_hash_of_the_artifact_not_a_label():
    """A hand-maintained string drifts on the first retrain somebody forgets to bump, and the
    decision would be attributed to a model that never scored it."""
    d = tempfile.mkdtemp()
    a = os.path.join(d, "m.pkl")
    open(a, "wb").write(b"weights-v1")
    v1 = M.artifact_version(a)
    open(a, "wb").write(b"weights-v2")
    v2 = M.artifact_version(a)
    assert v1 and v2 and v1 != v2, "retraining did not change the version"
    assert v1.startswith("m@") and len(v1.split("@")[1]) == 12


def test_a_missing_artifact_has_no_version_rather_than_a_fake_one():
    assert M.artifact_version("/nonexistent/nope.pkl") == ""


def test_the_decision_stamp_carries_champions_only():
    """Recording a challenger in model_version would attribute an outcome to a model that had
    no say in it, which is worse than recording nothing."""
    d = tempfile.mkdtemp()
    ca = os.path.join(d, "champ.pkl"); open(ca, "wb").write(b"c")
    ha = os.path.join(d, "chal.pkl"); open(ha, "wb").write(b"h")
    r = _reg()
    r.register(_spec("champ", state=M.CHAMPION, artifact=ca))
    r.register(_spec("chal", state=M.CHALLENGER, artifact=ha))
    for mid in ("champ", "chal"):
        r.load(mid, lambda: _Model())
    stamp = r.decision_versions()
    assert "champ@" in stamp and "chal@" not in stamp


# --------------------------------------------------------------------- inventory

def test_the_inventory_reports_what_is_live_and_what_failed():
    """The artifact model-risk guidance actually asks for. A catalogue that describes loads
    happening elsewhere is the condition under which governance collapses."""
    r = _reg()
    r.register(_spec("champ", state=M.CHAMPION, tier=M.TIER_1))
    r.register(_spec("chal", state=M.CHALLENGER, tier=M.TIER_2))
    r.register(_spec("stale", tier=M.TIER_3, feats=["a", "b"]))
    r.load("champ", lambda: _Model())
    r.load("chal", lambda: _Model())
    r.load("stale", lambda: _Model(n=99))

    inv = r.inventory()
    assert inv["total"] == 3
    assert inv["deciding"] == 1, "only the champion may decide"
    assert inv["failed"] == ["stale"]
    assert inv["by_state"][M.CHAMPION] == 1 and inv["by_state"][M.CHALLENGER] == 1
    assert inv["by_tier"][1] == 1 and inv["by_tier"][3] == 1


def test_a_failed_model_never_reports_itself_as_able_to_decide():
    """The combination that would be worst: registered, tier 1, champion, and broken. It must
    read as unavailable rather than as authoritative."""
    r = _reg()
    r.register(_spec("champ", state=M.CHAMPION, tier=M.TIER_1))
    r.load("champ", lambda: (_ for _ in ()).throw(OSError("corrupt")))
    inv = r.inventory()
    assert inv["deciding"] == 0
    assert inv["models"][0]["may_decide"] is False
    assert r.decision_versions() == ""


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
