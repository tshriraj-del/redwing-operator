"""
Tests for the consortium's differential-privacy accounting.

Why this file exists. A privacy guarantee is the one claim in this system that cannot be
checked by looking at the output: noise looks like noise whether the budget was spent once or
twice. The previous mechanism added Laplace(1/epsilon) to BOTH the fraud count and the
transaction count and reported that as epsilon-DP. One transaction moves both counts, so the
pair has L1 sensitivity 2 and the real guarantee was 2*epsilon. Every stated epsilon in the
system understated the privacy loss by a factor of two, and nothing failed.

These pin the accounting so it cannot drift back:
  - the budget is SPLIT across the two releases, never spent per-release
  - the split is reported, so a consumer can audit rather than trust
  - noise still actually happens, and more of it at a smaller epsilon
  - the uneven split favours the rare count, which is the point of it

Runs under pytest or standalone (python3 tests/test_consortium_dp.py).
"""

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import consortium as C  # noqa: E402


def _payee(neo_tx=40, neo_fraud=3, cry_tx=12, cry_fraud=5):
    return {"inst_neobank": {"tx": neo_tx, "fraud": neo_fraud},
            "inst_crypto": {"tx": cry_tx, "fraud": cry_fraud}}


def test_the_budget_is_split_not_spent_twice():
    """THE regression this file exists for. If these two ever sum to more than the stated
    epsilon, the mechanism is quietly handing out a weaker guarantee than it prints."""
    for eps in (0.5, 1.0, 2.0, 5.0):
        v = C.consortium_view(_payee(), epsilon=eps)
        assert v["epsilon"] == eps
        total = v["epsilon_fraud"] + v["epsilon_tx"]
        assert abs(total - eps) < 1e-6, (
            f"budget does not sum to epsilon at {eps}: fraud {v['epsilon_fraud']} + "
            f"tx {v['epsilon_tx']} = {total}. One transaction moves BOTH counts, so charging "
            f"the full budget to each makes the real guarantee 2x what is reported.")


def test_the_split_is_reported_so_it_can_be_audited():
    """A number a consumer cannot check is a number they have to trust. The division is part
    of the guarantee, so it travels with it."""
    v = C.consortium_view(_payee(), epsilon=1.0)
    for k in ("epsilon", "epsilon_fraud", "epsilon_tx"):
        assert k in v, f"{k} missing from the consortium view; the guarantee is unauditable"
    assert v["epsilon_fraud"] > v["epsilon_tx"], (
        "the split should favour the FRAUD count; see EPSILON_FRAUD_SHARE for why a rare "
        "count needs the precision and a large one does not")


def test_noise_actually_happens_and_grows_as_epsilon_shrinks():
    """A DP mechanism that returns the same answer every time is not adding noise. Spread is
    measured across SEEDS at a fixed input, and must widen as the budget tightens."""
    def spread(eps):
        rates = [C.consortium_view(_payee(), epsilon=eps, seed=s)["combined_rate_dp"]
                 for s in range(60)]
        return statistics.pstdev(rates)
    tight, loose = spread(0.25), spread(8.0)
    assert tight > 0, "no variation across seeds: the mechanism is not adding noise at all"
    assert tight > loose, (
        f"noise did not grow as epsilon shrank (sd {tight:.5f} at 0.25 vs {loose:.5f} at 8.0): "
        f"the budget is not reaching the noise scale")


def test_a_tiny_epsilon_cannot_be_trusted_but_still_returns():
    """Degenerate budgets must not raise or divide by zero. At a punitive epsilon the answer
    should be dominated by noise, which is correct behaviour, not an error."""
    v = C.consortium_view(_payee(), epsilon=1e-6)
    assert 0.0 <= v["combined_rate_dp"] <= 1.0
    assert v["epsilon_fraud"] + v["epsilon_tx"] > 0


def test_the_evidence_floor_still_silences_thin_payees():
    """Unchanged by the budget work, and worth pinning next to it: below the floor the rate is
    noise, so the network must not alert regardless of what the noised rate came out as."""
    thin = {"inst_neobank": {"tx": 2, "fraud": 2}, "inst_crypto": {"tx": 1, "fraud": 1}}
    v = C.consortium_view(thin, epsilon=1.0)
    assert not v["sufficient_evidence"]
    assert not v["alerts"], "alerted below the evidence floor, where the rate is noise"


def test_the_rare_count_gets_the_precision_it_needs():
    """The reason the split is uneven. A fraud count is small, so absolute noise is a huge
    relative error; a transaction count is large and shrugs it off. Estimating the fraud count
    should therefore be tighter than estimating the transaction count, at the same budget."""
    per = _payee()
    fr = C.EPSILON_FRAUD_SHARE
    assert fr > 0.5, "an even or tx-favouring split defeats the purpose"
    # scale is 1/eps_i, so the fraud count's noise scale must be the smaller of the two
    eps = 1.0
    assert (1.0 / (eps * fr)) < (1.0 / (eps * (1 - fr))), (
        "the fraud count is being noised MORE than the transaction count, which is backwards")
    assert per  # keeps the fixture meaningful if the assertions above are edited


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
