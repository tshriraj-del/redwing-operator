"""
fraud_env.py - a resettable agent-evaluation environment for financial-crime
investigation.

Most "AI fraud" demos grade a one-shot answer. This grades a *trajectory*: an agent
starts from a known case state, chooses from a bounded action space (gather specific
evidence, or commit a disposition), and is scored by two verifiers -

  • OUTCOME verifier  - was the final disposition correct vs. ground truth, weighted
                        by real fraud economics (clearing a fraud hurts far more than
                        a false alarm; blocking a good customer also costs).
  • PROCESS verifier  - did the agent inspect the DECISIVE evidence for *this* case
                        before deciding, without guessing or flailing? This rewards
                        correct investigative reasoning, not a lucky label.

Built on case_file.assemble(): every case already carries the full evidence set, the
ground-truth label, and a gold disposition. The environment simply *redacts* the
evidence behind inspection actions and reveals it as the agent works - turning the
investigator workbench into something an agent can be trained and evaluated against.

The environment is agent-agnostic. Drive it statelessly via step(case, history,
action), or run a reference policy end-to-end via run_episode(). Cases are synthetic
(enriched deterministically); the ground-truth labels are the synthetic is_fraud.
"""

from __future__ import annotations

# ── Action space ──────────────────────────────────────────────────────────────
INSPECT_ACTIONS = {
    "inspect_customer":       "customer",          # Customer 360 / CDD + risk profile
    "inspect_instrument":     "instrument",        # card / payment-instrument detail
    "inspect_card_signals":   "card_fraud_signals",# card-usage fraud playbook hits
    "inspect_dispute":        "dispute",           # dispute / chargeback evidence study
    "inspect_device_network": "device_network",    # device + graph / ring context
    "inspect_timeline":       "timeline",          # account activity timeline
}
TERMINAL_ACTIONS = [
    "confirm_fraud", "clear_false_positive", "deny_dispute_first_party",
    "escalate_stepup", "block_instrument", "place_hold",
]

STEP_COST = 0.02            # small cost per inspection → rewards efficient evidence use
W_OUTCOME = 0.6
W_PROCESS = 0.4


def env_spec() -> dict:
    """The environment contract - observation schema, action space, reward design."""
    return {
        "name": "redwing-financial-crime-investigation-v1",
        "task": "Investigate an escalated transaction and commit the correct disposition.",
        "observation": {
            "always_visible": ["case_id", "priority", "queue", "alert (scores/typology, "
                               "NO ground-truth label)", "transaction basics"],
            "revealed_by_inspection": list(INSPECT_ACTIONS.values()),
        },
        "actions": {
            "inspect": list(INSPECT_ACTIONS.keys()),
            "terminal": TERMINAL_ACTIONS,
        },
        "reward": {
            "outcome_weight": W_OUTCOME, "process_weight": W_PROCESS,
            "step_cost": STEP_COST,
            "outcome": "correct disposition vs ground truth, cost-sensitive "
                       "(clearing a fraud = -1.0; false-positive on a good customer = -0.6)",
            "process": "fraction of the case's DECISIVE evidence inspected before "
                       "deciding, minus penalties for guessing (deciding blind) and flailing",
        },
        "ground_truth": "synthetic is_fraud label + gold disposition per case",
    }


# ── Ground truth & verifiers ──────────────────────────────────────────────────

def _is_fraud(case: dict) -> bool:
    return case.get("alert", {}).get("ground_truth_label") == "fraud"


def gold_disposition(case: dict) -> str:
    d = case.get("dispute", {}) or {}
    if d.get("active") and (d.get("first_party_fraud_risk") or 0) >= 0.6:
        return "deny_dispute_first_party"
    return "confirm_fraud" if _is_fraud(case) else "clear_false_positive"


def outcome_reward(action: str, case: dict) -> float:
    """Cost-sensitive correctness of the terminal disposition."""
    gold = gold_disposition(case)
    if action == gold:
        return 1.0
    if _is_fraud(case):
        if action in ("confirm_fraud", "block_instrument"):
            return 1.0                              # fraud caught (just a different lever)
        if action in ("escalate_stepup", "place_hold"):
            return 0.4                              # held for review - not cleared
        if action == "deny_dispute_first_party":
            return -0.5                             # denied a real victim's dispute
        return -1.0                                 # cleared a real fraud - the costly miss
    # legitimate customer
    if action in ("escalate_stepup", "place_hold"):
        return -0.2                                 # needless friction on a good customer
    if action in ("confirm_fraud", "block_instrument", "deny_dispute_first_party"):
        return -0.6                                 # false positive - customer harm
    return -0.3


def decisive_evidence(case: dict) -> set:
    """Which evidence an analyst MUST look at to defend a decision on this case."""
    keys = set()
    if any(s.get("severity") == "high" for s in case.get("card_fraud_signals", [])):
        keys.add("inspect_card_signals")
    if (case.get("dispute", {}) or {}).get("active"):
        keys.add("inspect_dispute")
    if (case.get("device_network", {}) or {}).get("ring_flag"):
        keys.add("inspect_device_network")
    if (case.get("customer", {}) or {}).get("risk_rating") in ("High", "Medium"):
        keys.add("inspect_customer")
    if not keys:                                    # quiet case - at least check the card usage
        keys.add("inspect_card_signals")
    return keys


def process_reward(actions: list, case: dict) -> dict:
    """Reward correct investigative reasoning over the action sequence."""
    decisive = decisive_evidence(case)
    inspects = [a for a in actions if a in INSPECT_ACTIONS]
    inspected = set(inspects)
    covered = decisive & inspected
    coverage = len(covered) / len(decisive) if decisive else 1.0
    redundant = len(inspects) - len(inspected)                       # repeated inspections
    over = max(0, len(inspected) - len(decisive) - 1)               # flailing beyond need
    guessed = 1.0 if not inspects else 0.0                          # decided blind
    score = coverage - 0.10 * redundant - 0.05 * over - 0.5 * guessed
    return {
        "score": round(max(-1.0, min(1.0, score)), 3),
        "decisive": sorted(decisive),
        "covered": sorted(covered),
        "coverage": round(coverage, 3),
        "guessed": bool(guessed),
        "redundant": redundant,
    }


# ── Stateless step (any agent can drive it) ───────────────────────────────────

def _observation(case: dict, revealed: list) -> dict:
    a = case.get("alert", {})
    obs = {
        "case_id": case.get("case_id"), "priority": case.get("priority"),
        "queue": case.get("queue"),
        "alert": {k: a.get(k) for k in
                  ("trigger_label", "model_score", "combined_score", "verdict",
                   "fraud_typology", "matched_signals")},   # ground_truth_label withheld
        "transaction": case.get("transaction", {}),
        "revealed": {},
        "available_actions": list(INSPECT_ACTIONS) + TERMINAL_ACTIONS,
    }
    for act in revealed:
        sect = INSPECT_ACTIONS.get(act)
        if sect:
            obs["revealed"][sect] = case.get(sect)
    return obs


def step(case: dict, history: list, action: str) -> dict:
    """Apply `action` given the prior `history` of actions. Returns the resulting
    observation, reward, done flag, and info. Stateless: pass the full history each call."""
    history = list(history or [])
    if action in INSPECT_ACTIONS:
        revealed = [a for a in history if a in INSPECT_ACTIONS] + [action]
        return {
            "observation": _observation(case, revealed),
            "reward": -STEP_COST, "done": False,
            "info": {"type": "inspect", "revealed_section": INSPECT_ACTIONS[action]},
        }
    if action in TERMINAL_ACTIONS:
        o = outcome_reward(action, case)
        p = process_reward(history + [action], case)
        total = round(W_OUTCOME * o + W_PROCESS * p["score"]
                      - STEP_COST * len([a for a in history if a in INSPECT_ACTIONS]), 3)
        return {
            "observation": _observation(case, [a for a in history if a in INSPECT_ACTIONS]),
            "reward": o, "done": True,
            "info": {"type": "decide", "terminal_action": action,
                     "gold_disposition": gold_disposition(case),
                     "correct": action == gold_disposition(case),
                     "outcome_reward": round(o, 3), "process": p, "total_reward": total},
        }
    return {"observation": _observation(case, [a for a in history if a in INSPECT_ACTIONS]),
            "reward": -0.1, "done": False, "info": {"type": "invalid", "action": action}}


# ── Reference policies (deterministic - the env works with no LLM) ─────────────

def _policy_investigator(case, inspected):
    """Disciplined: gather the decisive evidence, then decide from what it sees."""
    todo = decisive_evidence(case) - set(inspected)
    if todo:
        return sorted(todo)[0]
    d = case.get("dispute", {}) or {}
    score = case.get("alert", {}).get("combined_score", 0) or 0
    high = any(s.get("severity") == "high" for s in case.get("card_fraud_signals", []))
    if d.get("active") and (d.get("first_party_fraud_risk") or 0) >= 0.6:
        return "deny_dispute_first_party"
    if high or score >= 0.6:
        return "confirm_fraud"
    if score < 0.35 and not high:
        return "clear_false_positive"
    return "escalate_stepup"


def _policy_trigger_happy(case, inspected):
    """Blocks first, asks never - no investigation."""
    return "block_instrument"


def _policy_cautious(case, inspected):
    """Escalates everything without looking - safe-seeming but useless."""
    return "escalate_stepup"


POLICIES = {
    "investigator":   _policy_investigator,
    "trigger_happy":  _policy_trigger_happy,
    "cautious":       _policy_cautious,
}


def run_episode(case: dict, agent: str = "investigator", max_steps: int = 10) -> dict:
    """Run a reference policy to a terminal action; return the full trajectory + scorecard."""
    policy = POLICIES.get(agent, _policy_investigator)
    history, trajectory, cumulative = [], [], 0.0
    for i in range(max_steps):
        inspected = [a for a in history if a in INSPECT_ACTIONS]
        action = policy(case, inspected)
        res = step(case, history, action)
        cumulative += res["reward"]
        entry = {"step": i + 1, "action": action, "type": res["info"]["type"],
                 "reward": round(res["reward"], 3), "cumulative": round(cumulative, 3)}
        if res["info"]["type"] == "inspect":
            entry["revealed"] = res["info"]["revealed_section"]
        trajectory.append(entry)
        history.append(action)
        if res["done"]:
            info = res["info"]
            return {
                "agent": agent, "transaction_id": case.get("transaction_id"),
                "case_id": case.get("case_id"),
                "ground_truth_label": case.get("alert", {}).get("ground_truth_label"),
                "gold_disposition": info["gold_disposition"],
                "trajectory": trajectory,
                "scorecard": {
                    "terminal_action": info["terminal_action"],
                    "correct": info["correct"],
                    "outcome_reward": info["outcome_reward"],
                    "process_reward": info["process"]["score"],
                    "process_detail": info["process"],
                    "total_reward": info["total_reward"],
                    "n_inspections": len([a for a in history if a in INSPECT_ACTIONS]),
                },
            }
    # ran out of steps without deciding - force a "no decision" penalty
    return {"agent": agent, "transaction_id": case.get("transaction_id"),
            "trajectory": trajectory,
            "scorecard": {"terminal_action": None, "correct": False,
                          "outcome_reward": -1.0, "process_reward": 0.0, "total_reward": -1.0,
                          "n_inspections": len([a for a in history if a in INSPECT_ACTIONS])}}
