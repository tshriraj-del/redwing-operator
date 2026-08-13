"""
core/model_registry.py - the ML layer this platform did not have.

WHAT WAS THERE BEFORE. Six `pickle.load(...)` calls scattered through `main.py`, each with its
own glue, its own fallback and its own idea of what a missing file means. No inventory, no
lifecycle, and `decisions.model_version` set on 0 of 692 rows. That is workable with two models.
It is not workable with the five this system is heading for: supervised scorer, novelty gate,
recovery uplift, decline hazard, merchant embedding.

The industry pattern is a three-layer split, feature extraction then MODEL SERVING then
decisioning, and REDWING had the first and third with an ad-hoc middle. This is the middle.

It is also the governance answer. Model-risk guidance (SR 11-7) asks for a model inventory, risk
tiering, lifecycle monitoring and effective challenge, and the failure mode named repeatedly in
practice is models running OUTSIDE the registry, where none of that applies. So the registry is
not a catalogue that describes what happens elsewhere; it is the only load path.

FOUR PROPERTIES, EACH ENFORCED RATHER THAN DOCUMENTED

  CONTRACT     a model declares the feature set it was fitted on, and loading refuses when the
               live set disagrees. This is a generalisation of the check that caught the
               isolation forest trained on 23 features while the model had moved to 32; that
               artifact would have scored a feature space that no longer meant what it did, on
               every payment, silently.

  LIFECYCLE    champion / challenger / shadow / retired. Only a CHAMPION may affect a decision.
               A challenger scores the same traffic and is recorded and compared, and it cannot
               change an outcome even if a caller asks it to. Effective challenge is worth
               nothing if the challenger can quietly become the decision.

  VERSION      a content hash of the artifact bytes, not a hand-maintained string. A string
               drifts the moment someone retrains and forgets to bump it, and a decision would
               then be attributed to a model that never scored it. This is what finally writes
               `decisions.model_version`.

  FAIL-SAFE    a model that will not load, or whose contract does not match, is refused and
               reported. The decision path continues without it. An advisory model that can
               take down the money path is a liability, not a control.

RISK TIER decides how much rigour a model owes. Tier 1 moves money and needs the full
validation chain; tier 3 informs a human and does not. Recording it here means the inventory can
answer "what is in production and how closely is it watched" without a spreadsheet.

Pure stdlib. Loading a pickle is the caller's business; this module governs WHETHER it is loaded
and what it is allowed to do afterwards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ── lifecycle ────────────────────────────────────────────────────────────────
CHAMPION = "champion"     # may affect a decision
CHALLENGER = "challenger" # scores the same traffic, compared, may NEVER affect a decision
SHADOW = "shadow"         # scores, recorded, not compared to anything yet
RETIRED = "retired"       # kept for audit, never scored

DECIDING_STATES = (CHAMPION,)
SCORING_STATES = (CHAMPION, CHALLENGER, SHADOW)

# ── risk tier (SR 11-7 shaped) ───────────────────────────────────────────────
# Tier decides the rigour owed, and is recorded so the inventory can answer "what is live and
# how closely is it watched" without a side spreadsheet that goes stale.
TIER_1 = 1   # drives a money decision directly
TIER_2 = 2   # shapes a money decision alongside others
TIER_3 = 3   # advisory: informs a human, never moves money on its own


class ModelSpec:
    """One model's declaration. Everything the registry needs to govern it."""

    def __init__(self, model_id: str, purpose: str, tier: int,
                 features: list | None = None, state: str = SHADOW,
                 artifact: str | Path | None = None, notes: str = ""):
        self.model_id = model_id
        self.purpose = purpose
        self.tier = int(tier)
        self.features = list(features or [])
        self.state = state
        self.artifact = str(artifact) if artifact else ""
        self.notes = notes
        self.obj = None
        self.version = ""
        self.error = ""

    @property
    def loaded(self) -> bool:
        return self.obj is not None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "purpose": self.purpose, "tier": self.tier,
            "state": self.state, "version": self.version, "loaded": self.loaded,
            "n_features": len(self.features), "artifact": self.artifact,
            "notes": self.notes, "error": self.error or None,
            "may_decide": self.state in DECIDING_STATES and self.loaded,
        }


def artifact_version(path) -> str:
    """A content hash of the artifact itself.

    Not a hand-maintained version string: that drifts the moment somebody retrains and forgets
    to bump it, and a decision would then be attributed to a model that never scored it. Hashing
    the bytes means the version cannot disagree with what is actually loaded.
    """
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{p.stem}@{h.hexdigest()[:12]}"


def contract_ok(obj, features: list) -> tuple:
    """Does this artifact expect the feature set we are about to hand it?

    Generalised from the guard that caught an isolation forest fitted on 23 features while the
    supervised model had moved to 32. Loading it would have scored a feature space that no
    longer meant what it did, on every payment, and nothing would have raised.

    An object that does not advertise its arity passes: sklearn sets `n_features_in_`, plenty of
    other things do not, and refusing everything unlabelled would make the registry unusable for
    exactly the custom models it exists to govern.
    """
    n = getattr(obj, "n_features_in_", None)
    if n is None:
        return True, ""
    if not features:
        return True, ""
    if int(n) != len(features):
        return False, (f"artifact expects {int(n)} features, the live contract has "
                       f"{len(features)}; re-fit it or fix the contract")
    return True, ""


class Registry:
    """The inventory, and the only sanctioned load path."""

    def __init__(self):
        self._specs: dict = {}

    # -- registration ---------------------------------------------------------

    def register(self, spec: ModelSpec) -> ModelSpec:
        self._specs[spec.model_id] = spec
        return spec

    def load(self, model_id: str, loader, features: list | None = None) -> ModelSpec:
        """Load an artifact through the registry.

        `loader` is a callable returning the model object, so this module never needs to know
        whether something is a pickle, a JSON of weights, or pure Python. What it governs is
        whether the result is allowed to be used.
        """
        spec = self._specs.get(model_id)
        if spec is None:
            raise KeyError(f"{model_id!r} is not registered; register a ModelSpec first")
        if features is not None:
            spec.features = list(features)
        try:
            obj = loader()
        except Exception as e:                                   # noqa: BLE001
            spec.obj, spec.error = None, f"{type(e).__name__}: {e}"
            return spec

        ok, why = contract_ok(obj, spec.features)
        if not ok:
            # Refused, loudly. A contract mismatch is not a degraded model, it is a model
            # scoring a different world than the one it was fitted in.
            spec.obj, spec.error = None, why
            return spec

        spec.obj = obj
        spec.error = ""
        spec.version = artifact_version(spec.artifact) if spec.artifact else ""
        return spec

    # -- use ------------------------------------------------------------------

    def get(self, model_id: str, *, for_decision: bool = False):
        """The model object, or None.

        `for_decision=True` returns None for anything that is not a loaded CHAMPION. That is the
        enforcement, not a convention: effective challenge is worth nothing if a challenger can
        quietly become the decision because a caller passed the wrong id.
        """
        spec = self._specs.get(model_id)
        if spec is None or not spec.loaded:
            return None
        if for_decision and spec.state not in DECIDING_STATES:
            return None
        return spec.obj

    def may_decide(self, model_id: str) -> bool:
        spec = self._specs.get(model_id)
        return bool(spec and spec.loaded and spec.state in DECIDING_STATES)

    def scoring_models(self) -> list:
        """Everything that should score this transaction, champion and challengers alike. The
        challengers' scores are recorded for comparison and discarded from the decision."""
        return [s for s in self._specs.values()
                if s.loaded and s.state in SCORING_STATES]

    def promote(self, model_id: str, to_state: str) -> ModelSpec:
        """Move a model through its lifecycle.

        Promoting a challenger to champion DEMOTES the incumbent for the same purpose rather
        than leaving two champions, because two models both entitled to decide is the state in
        which nobody can say which one did.
        """
        spec = self._specs[model_id]
        if to_state == CHAMPION:
            for other in self._specs.values():
                if (other is not spec and other.purpose == spec.purpose
                        and other.state == CHAMPION):
                    other.state = RETIRED
        spec.state = to_state
        return spec

    # -- reporting ------------------------------------------------------------

    def inventory(self) -> dict:
        """What is in production, in what state, at what version, and how closely watched.

        This is the artifact model-risk guidance actually asks for, and the reason the registry
        is the load path rather than a catalogue describing loads that happen elsewhere.
        """
        specs = [s.to_dict() for s in self._specs.values()]
        return {
            "models": sorted(specs, key=lambda d: (d["tier"], d["model_id"])),
            "total": len(specs),
            "deciding": sum(1 for d in specs if d["may_decide"]),
            "failed": [d["model_id"] for d in specs if d["error"]],
            "by_state": {st: sum(1 for d in specs if d["state"] == st)
                         for st in (CHAMPION, CHALLENGER, SHADOW, RETIRED)},
            "by_tier": {t: sum(1 for d in specs if d["tier"] == t) for t in (1, 2, 3)},
        }

    def version_of(self, model_id: str) -> str:
        """The content-hash version of one model, or "" if it is not loaded.

        `get()` deliberately returns the model OBJECT, so callers wanting the declaration had no
        public way to reach it and were writing `REGISTRY.get(id).version`, which is an
        AttributeError on an XGBClassifier rather than a missing-model error. A scorer stamping
        its own output with its version is a routine need and deserves an accessor.
        """
        spec = self._specs.get(model_id)
        return (spec.version or "") if spec and spec.loaded else ""

    def decision_versions(self) -> str:
        """The version stamp for `decisions.model_version`.

        Every model that could affect this decision, in one stable string. Champions only:
        recording a challenger here would attribute an outcome to a model that had no say in it.
        """
        parts = sorted(s.version or s.model_id for s in self._specs.values()
                       if s.loaded and s.state in DECIDING_STATES)
        return ";".join(parts)


# A process-wide registry, so the inventory is one thing rather than one per importer.
REGISTRY = Registry()


def describe() -> str:
    return json.dumps(REGISTRY.inventory(), indent=2, sort_keys=True)
