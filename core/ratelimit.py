"""
core/ratelimit.py - a bounded, in-process token-bucket limiter.

WHY THIS IS HAND-ROLLED, given the repo rule to reuse before implementing. The candidates are
slowapi and asgi-ratelimit; both were checked and neither is installed, and both want redis for
anything beyond one process. This deployment is a single uvicorn process, so an in-memory bucket
is the correct shape, and adding a dependency days before an external security assessment is a
supply-chain decision rather than a convenience. `core/` is pure stdlib by design, which is what
lets the bulk of the suite run without loading a model, and this stays inside that boundary.

WHAT THIS IS NOT. It is not a distributed limiter. Run more than one worker and each gets its own
buckets, so the effective budget multiplies by the worker count. That is stated rather than
hidden: if this ever runs multi-process, the counter has to move to shared storage, and a limiter
that silently means something different under gunicorn is worse than none.

THE FAILURE MODE THIS FILE IS SHAPED AROUND. The obvious implementation is `dict[ip] -> bucket`,
and it is a memory-exhaustion DoS wearing the costume of a defence: an attacker rotating source
addresses grows the dict without limit, and one rotating a header the limiter TRUSTS does it from
a single socket. So the store is capacity-bounded, eviction prefers idle buckets over exhausted
ones, and the key comes from the socket rather than from anything the caller can write.
"""

from __future__ import annotations

import time
from collections import OrderedDict

# Default budget. Chosen, not measured: no traffic study exists for this service, so this is an
# ASSUMPTION sized to be generous for a human operator and a console, and restrictive for a
# scripted walk of 96 routes. Tighten per-route where the work per request is large.
DEFAULT_RATE = 120
DEFAULT_WINDOW_SECONDS = 60

# The number of distinct callers tracked at once. Bounds the limiter's own memory: ~100 bytes per
# entry, so 10k entries is ~1MB. A real deployment sees far fewer distinct peers than this; an
# attacker rotating addresses hits the cap and starts evicting, which is the intended behaviour.
DEFAULT_CAPACITY = 10_000


class RateLimiter:
    """Token bucket per caller, with a bounded number of buckets.

    `rate` tokens are granted per `per_seconds`, refilled continuously rather than on a fixed
    window boundary, so a caller cannot spend a full budget at 11:59:59 and another at 12:00:00.
    """

    def __init__(self, rate: int = DEFAULT_RATE, per_seconds: float = DEFAULT_WINDOW_SECONDS,
                 capacity: int = DEFAULT_CAPACITY, clock=time.monotonic):
        self.rate = max(1, int(rate))
        self.per = max(0.001, float(per_seconds))
        self.capacity = max(1, int(capacity))
        self._clock = clock
        # OrderedDict as an LRU: key -> (tokens, last_refill_ts)
        self._buckets: OrderedDict = OrderedDict()

    def size(self) -> int:
        return len(self._buckets)

    # How many of the oldest buckets to consider when making room. Bounded so eviction stays
    # cheap; the scan is over an OrderedDict in insertion order, so these are the stalest keys.
    _EVICT_SAMPLE = 16

    def _evict_if_needed(self) -> None:
        """Make room, preferring an idle bucket that still HAS budget.

        Eviction policy is a security decision, not cache tuning. Plain LRU is wrong here:
        evicting a bucket forgets that its owner was over the limit, so capacity pressure grants
        amnesty. A caller who is being refused would be reset by unrelated churn, and where an
        attacker can influence that churn they reset themselves on demand.

        So among the stalest buckets, drop the one with the MOST tokens left: forgetting a caller
        who had budget to spare costs nothing, because recreating their bucket gives them the
        full allowance they already had. Falling back to the oldest keeps this terminating when
        every sampled bucket is equally exhausted.
        """
        while len(self._buckets) >= self.capacity:
            victim, best = None, -1.0
            for i, (k, (tokens, _ts)) in enumerate(self._buckets.items()):
                if i >= self._EVICT_SAMPLE:
                    break
                if tokens > best:
                    victim, best = k, tokens
            if victim is None:
                self._buckets.popitem(last=False)
            else:
                self._buckets.pop(victim, None)

    def allow(self, key: str) -> tuple:
        """Returns `(allowed, retry_after_seconds)`. Charges one token when allowed."""
        key = str(key or "unknown")
        now = self._clock()
        entry = self._buckets.get(key)
        if entry is None:
            self._evict_if_needed()
            tokens, last = float(self.rate), now
        else:
            tokens, last = entry
            elapsed = max(0.0, now - last)
            # Refill continuously, and NEVER past the ceiling: an idle caller must not bank
            # credit and arrive with one enormous burst.
            tokens = min(float(self.rate), tokens + elapsed * (self.rate / self.per))
            last = now

        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, last)
            self._buckets.move_to_end(key)            # mark recently used for the LRU
            return True, 0.0

        # Refused. Report when one whole token will exist, so the caller can back off correctly
        # instead of hot-looping.
        deficit = 1.0 - tokens
        retry_after = deficit * (self.per / self.rate)
        self._buckets[key] = (tokens, last)
        self._buckets.move_to_end(key)                # a refusal is still activity, see _evict
        return False, round(retry_after, 3)


def client_key(request, trust_proxy: bool = False) -> str:
    """The caller's identity for limiting purposes.

    THE SOCKET, NOT A HEADER, unless a proxy is explicitly declared. `X-Forwarded-For` is written
    by the caller: trusting it by default lets anyone pick their own bucket, which both evades the
    limit and multiplies the limiter's memory from a single connection. It is honoured only when
    the operator states a proxy is in front, because only then is it true.
    """
    if trust_proxy:
        fwd = ""
        try:
            fwd = str(request.headers.get("X-Forwarded-For", "") or "")
        except Exception:                                         # noqa: BLE001
            fwd = ""
        if fwd:
            return fwd.split(",")[0].strip() or "unknown"
    try:
        host = getattr(getattr(request, "client", None), "host", "") or ""
    except Exception:                                             # noqa: BLE001
        host = ""
    # A stable literal, not None and not a fresh value per request: either would silently exempt
    # these requests from the limit entirely.
    return str(host) if host else "unknown"


def clamp(value, low, high, default=None):
    """Bound a caller-supplied numeric parameter, degrading to `default` on garbage.

    Not decoration. `/network/graph` takes `limit_nodes` as an unbounded int and only samples
    when the row count exceeds `limit_nodes * 3`, so a large enough value DISABLES the sampler
    and drops into a 914,127-row iteration over a 126MB frame. The clamp is what makes the
    request cost bounded regardless of what arrives.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default if default is not None else low
    if v != v:                                                    # NaN
        return default if default is not None else low
    v = max(float(low), min(float(high), v))
    return int(v) if isinstance(low, int) and isinstance(high, int) else v


def bounded_text(value, limit: int = 128) -> str:
    """A caller-supplied identifier, length-capped.

    Every column in the decisions schema is bare TEXT with no length constraint and nothing
    between the socket and the INSERT bounds them, so `subject_ref` and `transaction_id` are
    storage-exhaustion vectors against a 1.3GB database. REJECTION is the caller's job; this is
    the last line, and it truncates rather than raising because it sits on decision paths where
    an exception would cost the decision.
    """
    s = str(value if value is not None else "")
    return s[:max(1, int(limit))]
