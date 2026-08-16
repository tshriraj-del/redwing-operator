"""
Tests for the rate limiter, and for the property that makes a rate limiter safe to add.

WHY THIS FILE EXISTS. CLAUDE.md has mandated "Rate limiting on endpoints" as a pre-commit
security rule since it was written, and the measured implementation count was zero. Several
unauthenticated endpoints do heavy work per request: `/network/graph` reads a 126MB CSV and, with
a large enough `limit_nodes`, disables its own sampler and iterates 914,127 rows; `/alerts` runs
`build_event()` per row, which WRITES to SQLite, so it is a write amplifier.

THE TRAP THIS FILE EXISTS TO CATCH. The obvious limiter is a dict keyed by client IP, and that
is itself a memory-exhaustion DoS: an attacker rotating source addresses (or spoofing a header
the limiter trusts) grows the dict without bound, and the defence becomes the vulnerability. So
the bucket store is capacity-bounded and that bound is asserted here, not assumed.

AND IT MUST NOT TRUST A CLIENT-SUPPLIED IDENTITY. If the key comes from `X-Forwarded-For`, a
caller picks their own bucket, which both evades the limit and multiplies the memory. It is
trusted only when a proxy is explicitly declared.

Pure stdlib, no ML stack, no network. Runs under pytest or standalone.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ratelimit import RateLimiter, client_key  # noqa: E402


# ── the budget ───────────────────────────────────────────────────────────────

def test_requests_under_the_budget_are_allowed():
    rl = RateLimiter(rate=5, per_seconds=60, capacity=1000)
    for i in range(5):
        assert rl.allow("1.2.3.4")[0] is True, f"request {i + 1} of 5 was refused"


def test_the_request_over_the_budget_is_refused():
    rl = RateLimiter(rate=3, per_seconds=60, capacity=1000)
    for _ in range(3):
        rl.allow("1.2.3.4")
    ok, retry_after = rl.allow("1.2.3.4")
    assert ok is False
    assert retry_after > 0, "a refusal must tell the caller when to come back"


def test_budgets_are_per_caller():
    """One noisy caller must not spend everyone else's budget."""
    rl = RateLimiter(rate=2, per_seconds=60, capacity=1000)
    rl.allow("1.1.1.1"); rl.allow("1.1.1.1")
    assert rl.allow("1.1.1.1")[0] is False
    assert rl.allow("2.2.2.2")[0] is True, "a second caller was charged for the first's traffic"


def test_tokens_refill_with_time():
    """MEASURED with an injected clock rather than a sleep, so the test is fast and exact."""
    now = [1000.0]
    rl = RateLimiter(rate=60, per_seconds=60, capacity=1000, clock=lambda: now[0])
    for _ in range(60):
        rl.allow("1.2.3.4")
    assert rl.allow("1.2.3.4")[0] is False
    now[0] += 30.0                                    # half a window -> ~30 tokens back
    assert rl.allow("1.2.3.4")[0] is True


def test_a_bucket_never_refills_past_its_ceiling():
    """Otherwise an idle caller banks unlimited credit and arrives with one enormous burst.

    THIS TEST PASSED FOR THE WRONG REASON UNTIL MUTATION EXPOSED IT. The first version advanced
    the clock BEFORE the bucket existed, and the first allow() creates a bucket at the full rate
    regardless of the clock, so the jump changed nothing and the assertion held with the ceiling
    deleted. The bucket has to be created and DRAINED first, so that the long idle period is
    actually refilling something.
    """
    now = [1000.0]
    rl = RateLimiter(rate=10, per_seconds=60, capacity=1000, clock=lambda: now[0])
    for _ in range(10):                               # create the bucket and drain it
        rl.allow("1.2.3.4")
    assert rl.allow("1.2.3.4")[0] is False, "fixture did not actually drain the bucket"

    now[0] += 100_000.0                               # idle for a very long time
    allowed = sum(1 for _ in range(50) if rl.allow("1.2.3.4")[0])
    assert allowed == 10, f"banked {allowed} tokens against a ceiling of 10"


# ── the limiter must not become the vulnerability ────────────────────────────

def test_the_bucket_store_is_capacity_bounded():
    """THE property that makes this safe to deploy. A dict keyed by client address grows without
    bound under address rotation, so the limiter becomes the memory-exhaustion DoS it exists to
    prevent."""
    rl = RateLimiter(rate=5, per_seconds=60, capacity=100)
    for i in range(10_000):
        rl.allow(f"10.0.{i // 256}.{i % 256}")
    assert rl.size() <= 100, f"the bucket store grew to {rl.size()} against a cap of 100"


def test_eviction_does_not_hand_an_attacker_a_reset():
    """If eviction dropped the BUSIEST bucket, a caller could evict themselves by rotating
    addresses and come back with a full budget. Eviction must prefer idle buckets."""
    now = [1000.0]
    rl = RateLimiter(rate=3, per_seconds=60, capacity=8, clock=lambda: now[0])
    for _ in range(3):
        rl.allow("attacker")
    assert rl.allow("attacker")[0] is False
    now[0] += 1.0
    for i in range(50):                               # flood with fresh keys to force eviction
        rl.allow(f"filler-{i}")
    assert rl.allow("attacker")[0] is False, (
        "the attacker's exhausted bucket was evicted, resetting their budget")


# ── the caller identity may not be chosen by the caller ──────────────────────

class _Req:
    def __init__(self, ip, headers=None):
        self.client = type("C", (), {"host": ip})() if ip else None
        self.headers = headers or {}


def test_the_key_is_the_socket_address_not_a_header():
    """`X-Forwarded-For` is caller-supplied. Trusting it by default lets anyone pick their own
    bucket, which evades the limit AND multiplies memory consumption."""
    r = _Req("9.9.9.9", {"X-Forwarded-For": "1.1.1.1"})
    assert client_key(r, trust_proxy=False) == "9.9.9.9"


def test_not_trusting_the_proxy_header_is_the_DEFAULT():
    """CALLED WITHOUT THE ARGUMENT, which the test above never did. Mutation exposed the gap:
    flipping the signature default to True broke nothing, because every test passed the flag
    explicitly. The default is the security-relevant part, since it is what a future caller who
    omits the argument inherits."""
    r = _Req("9.9.9.9", {"X-Forwarded-For": "1.1.1.1"})
    assert client_key(r) == "9.9.9.9", "the signature default trusts a caller-supplied header"


def test_a_declared_proxy_is_honoured():
    """Behind a real proxy every socket address is the proxy's, so the limiter would see one
    caller. That is opt-in, because it is only correct when a proxy really is in front."""
    r = _Req("10.0.0.1", {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"})
    assert client_key(r, trust_proxy=True) == "1.1.1.1"


def test_a_request_with_no_client_still_yields_a_stable_key():
    """A missing peer address must not become a None key or a fresh key per request; either way
    the limit stops applying."""
    a = client_key(_Req(None), trust_proxy=False)
    b = client_key(_Req(None), trust_proxy=False)
    assert a and a == b


def test_a_forged_forwarded_header_cannot_mint_unlimited_buckets():
    """The end-to-end version of the two tests above, stated as the attack."""
    rl = RateLimiter(rate=2, per_seconds=60, capacity=5000)
    for i in range(200):
        k = client_key(_Req("9.9.9.9", {"X-Forwarded-For": f"1.1.1.{i}"}), trust_proxy=False)
        rl.allow(k)
    assert rl.size() == 1, f"a forged header minted {rl.size()} buckets from one address"


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
