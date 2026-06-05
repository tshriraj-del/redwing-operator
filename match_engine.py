# Match Engine — scores a transaction against the pattern library.
# Returns soft confidence scores (0.0–1.0) per pattern using weighted signal matching.
# Supports partial credit for near-miss thresholds so borderline cases surface.

from patterns import PATTERNS
from typing import Any, Dict, List


def _signal_match(value: float, op: str, threshold: float) -> float:
    """
    Returns 0.0–1.0 match strength for a single signal.
    Hard match at threshold; linear decay for near-misses up to 2x the gap.
    """
    if op == "lt":
        if value <= threshold:
            return 1.0
        decay = (value - threshold) / max(threshold, 0.01)
        return max(0.0, 1.0 - decay)

    elif op == "gt":
        if value >= threshold:
            return 1.0
        decay = (threshold - value) / max(threshold, 0.01)
        return max(0.0, 1.0 - decay)

    elif op == "eq":
        return 1.0 if abs(value - threshold) < 0.05 else 0.0

    return 0.0


def score_transaction(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Score a feature dict against all patterns.

    Args:
        features: dict mapping feature name → float value
                  (same 10 features used by the XGBoost model)

    Returns:
        List of pattern match results sorted descending by confidence.
        Each result includes pattern metadata, confidence, and matched signals.
    """
    results = []

    for pattern in PATTERNS:
        total_weight = sum(s["weight"] for s in pattern["signals"])
        weighted_score = 0.0
        matched_signals = []
        missed_signals = []

        for sig in pattern["signals"]:
            value = float(features.get(sig["feature"], 0.0))
            strength = _signal_match(value, sig["op"], sig["threshold"])
            weighted_score += sig["weight"] * strength

            if strength >= 0.5:
                matched_signals.append({
                    "label": sig["label"],
                    "feature": sig["feature"],
                    "value": round(value, 4),
                    "strength": round(strength, 2),
                })
            else:
                missed_signals.append({
                    "label": sig["label"],
                    "feature": sig["feature"],
                    "value": round(value, 4),
                })

        confidence = weighted_score / total_weight if total_weight > 0 else 0.0

        results.append({
            "pattern_id":      pattern["id"],
            "pattern_name":    pattern["name"],
            "icon":            pattern["icon"],
            "risk":            pattern["risk"],
            "color":           pattern["color"],
            "confidence":      round(confidence, 4),
            "matched_signals": matched_signals,
            "missed_signals":  missed_signals,
            "signal_hit_rate": round(len(matched_signals) / len(pattern["signals"]), 2),
        })

    return sorted(results, key=lambda x: x["confidence"], reverse=True)


def combined_score(ml_score: float, pattern_confidence: float, ml_weight: float = 0.60) -> float:
    """
    Blend XGBoost probability with pattern-match confidence.
    Default: 60% ML + 40% pattern (pattern adds context ML can't see).
    """
    pattern_weight = 1.0 - ml_weight
    return round(ml_score * ml_weight + pattern_confidence * pattern_weight, 4)


def is_alert(score: float, threshold: float = 0.65) -> bool:
    return score >= threshold
