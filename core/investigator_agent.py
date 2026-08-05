"""
core/investigator_agent.py - an LLM investigator that is GRADED, not asserted.

WHY THIS IS DIFFERENT FROM AN "AI FRAUD AGENT" DEMO. The usual shape is: hand a model a case,
let it write a confident paragraph, and present the paragraph as the result. Nothing in that
loop can tell a genuine investigation from a plausible-sounding guess that happened to land on
the right label, and on a skewed target guessing lands on the right label most of the time.

`fraud_env.py` was built first and deliberately: it redacts the case behind inspection actions
and scores a TRAJECTORY with two verifiers, one for the disposition and one for whether the
decisive evidence was actually looked at before deciding. This module simply drives that
environment with a language model instead of a hand-written policy, which means the agent is
measured by machinery that predates it and cannot be talked around.

The comparison is the artifact. Three reference policies already exist and the agent is run
against the same cases, the same verifiers and the same scorecard:

    investigator    disciplined: gathers the decisive evidence, then decides. The BASELINE TO
                    BEAT, and it is genuinely strong, because on most cases the right process
                    is not subtle.
    trigger_happy   blocks immediately, investigates never. Scores well on fraud, badly on
                    everybody else, and exists to catch an agent that has learned to always
                    block.
    cautious        escalates everything without looking. Never catastrophically wrong, never
                    useful, and exists to catch an agent that has learned to always escalate.

An agent that beats `investigator` on outcome while losing on process has not investigated
better; it has guessed luckier. Reporting both is the point, and `compare()` puts them in one
table so the difference cannot be hidden in an average.

NO KEY, NO PRETENDING. Without an API key this returns `available: false` and a reason. It does
not silently fall back to a scripted policy and report the result as an agent run, because a
demo that fabricates its headline is worse than one that says it needs a key.

Pure stdlib plus urllib for the API call, so it stays importable and testable with no ML stack
and no network.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import fraud_env

MODEL = "claude-sonnet-4-5-20250929"
API_URL = "https://api.anthropic.com/v1/messages"
MAX_STEPS = 8            # the env's action space is small; more than this is flailing
TIMEOUT = 45


def _api_key() -> str:
    """Same resolution order the rest of the operator uses."""
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VITE_ANTHROPIC_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith(("ANTHROPIC_API_KEY", "VITE_ANTHROPIC_API_KEY")):
                return line.split("=", 1)[-1].strip()
    return ""


SYSTEM = """You are a financial-crime investigator working one escalated case at a time.

You see a redacted case. Evidence is hidden until you inspect it. Each turn you choose exactly
one action.

INSPECT ACTIONS (reveal evidence, no decision made):
  inspect_customer        Customer 360, CDD, risk rating
  inspect_instrument      the payment instrument's detail
  inspect_card_signals    card-usage fraud playbook hits
  inspect_dispute         dispute / chargeback evidence
  inspect_device_network  device fingerprint and ring context
  inspect_timeline        account activity timeline

TERMINAL ACTIONS (end the case):
  confirm_fraud               it is fraud
  clear_false_positive        it is legitimate
  deny_dispute_first_party    the customer is disputing their own genuine transaction
  escalate_stepup             not resolvable here, step the customer up
  block_instrument            block the card / instrument
  place_hold                  hold the funds pending review

You are graded on two things, separately:
  1. whether the disposition is right, weighted by real cost. Clearing a genuine fraud is the
     worst outcome. Blocking a genuine customer is also expensive.
  2. whether you INSPECTED THE DECISIVE EVIDENCE before deciding. Deciding without inspecting
     anything is penalised heavily even when the answer happens to be right, and repeatedly
     inspecting the same thing or inspecting everything indiscriminately is penalised too.

Investigate what this case actually calls for, then decide. Do not inspect everything by
reflex, and do not decide blind.

Reply with ONLY a JSON object, no prose around it:
{"reasoning": "<one sentence>", "action": "<exactly one action name>"}"""


def _post(key: str, messages: list) -> str:
    body = json.dumps({
        "model": MODEL, "max_tokens": 300, "system": SYSTEM, "messages": messages,
    }).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        payload = json.loads(r.read())
    parts = payload.get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _parse_action(text: str, valid: set) -> tuple:
    """(action, reasoning). Tolerant: models wrap JSON in prose or fences often enough that
    failing on it would measure formatting rather than investigation."""
    reasoning = ""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            reasoning = str(obj.get("reasoning", ""))[:300]
            act = str(obj.get("action", "")).strip()
            if act in valid:
                return act, reasoning
        except (ValueError, TypeError):
            pass
    # Fall back to the first valid action name mentioned anywhere in the reply.
    for a in sorted(valid, key=len, reverse=True):
        if a in (text or ""):
            return a, reasoning or "(action recovered from unstructured reply)"
    return "", reasoning


def run_episode(case: dict, max_steps: int = MAX_STEPS) -> dict:
    """Drive fraud_env with the LLM and return the trajectory plus the env's own scorecard.

    Deliberately drives `fraud_env.step`, the same stateless function the reference policies go
    through, and assembles the scorecard from the env's own `info` block in the same shape
    `fraud_env.run_episode` produces. Scoring the agent with anything of its own would let the
    agent's author choose the grader, which is the failure this environment exists to prevent.
    """
    key = _api_key()
    if not key:
        return {"available": False,
                "reason": ("no ANTHROPIC_API_KEY in the environment or .env, so no agent run "
                           "happened. Nothing is substituted: a scripted policy reported as an "
                           "agent result would make the whole comparison meaningless.")}

    valid = set(fraud_env.INSPECT_ACTIONS) | set(fraud_env.TERMINAL_ACTIONS)
    history, transcript, cumulative = [], [], 0.0
    # The opening observation. step() is stateless and returns the redacted view for an empty
    # history when handed an action it does not recognise, so this costs nothing and avoids
    # reaching into the module's internals for a view it already exposes.
    obs = fraud_env.step(case, [], "")["observation"]

    for _ in range(max_steps):
        prompt = (f"Case state (evidence you have not inspected is absent):\n"
                  f"{json.dumps(obs, indent=2, default=str)[:6000]}\n\n"
                  f"Actions already taken: {history or 'none'}\n"
                  f"Choose one action.")
        try:
            raw = _post(key, [{"role": "user", "content": prompt}])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            return {"available": False,
                    "reason": f"the model call failed ({type(e).__name__}), so there is no "
                              f"trajectory to score", "history": history}

        action, reasoning = _parse_action(raw, valid)
        if not action:
            transcript.append({"action": None, "reasoning": reasoning,
                               "note": "no valid action parsed; episode abandoned"})
            break

        res = fraud_env.step(case, history, action)
        history.append(action)
        cumulative += res["reward"]
        entry = {"step": len(history), "action": action, "type": res["info"]["type"],
                 "reasoning": reasoning, "reward": round(res["reward"], 3),
                 "cumulative": round(cumulative, 3)}
        if res["info"]["type"] == "inspect":
            entry["revealed"] = res["info"]["revealed_section"]
        transcript.append(entry)
        obs = res["observation"]

        if res["done"]:
            # The scorecard is assembled from the env's OWN info block, in the same shape
            # run_episode() produces for the reference policies. Computing it any other way
            # here would let the agent's author pick the grader, which is the failure this
            # whole environment exists to prevent.
            info = res["info"]
            return {
                "available": True, "model": MODEL, "agent": "llm_agent",
                "transaction_id": case.get("transaction_id"), "case_id": case.get("case_id"),
                "ground_truth_label": case.get("alert", {}).get("ground_truth_label"),
                "gold_disposition": info["gold_disposition"],
                "trajectory": transcript,
                "scorecard": {
                    "terminal_action": info["terminal_action"],
                    "correct": info["correct"],
                    "outcome_reward": info["outcome_reward"],
                    "process_reward": info["process"]["score"],
                    "process_detail": info["process"],
                    "total_reward": info["total_reward"],
                    "n_inspections": len([a for a in history
                                          if a in fraud_env.INSPECT_ACTIONS]),
                },
            }

    # Never committed. Scored exactly as run_episode scores the same failure, rather than
    # quietly omitted: an agent that investigates forever and decides nothing is a real
    # failure mode and the one most likely to look fine in a transcript.
    return {"available": True, "model": MODEL, "agent": "llm_agent",
            "transaction_id": case.get("transaction_id"), "trajectory": transcript,
            "scorecard": {"terminal_action": None, "correct": False,
                          "outcome_reward": -1.0, "process_reward": 0.0,
                          "total_reward": -1.0,
                          "n_inspections": len([a for a in history
                                                if a in fraud_env.INSPECT_ACTIONS])}}


def compare(case: dict, policies=("investigator", "trigger_happy", "cautious")) -> dict:
    """The agent and the reference policies on ONE case, same verifiers, one table.

    The reference policies are the honest part. `investigator` is a strong baseline and an
    agent that does not beat it should be reported as not beating it; `trigger_happy` and
    `cautious` exist so a degenerate strategy cannot look good by scoring well on the half of
    the distribution it happens to suit.
    """
    out = {"case_id": case.get("case_id") or case.get("transaction_id", ""), "runs": {}}
    for p in policies:
        try:
            out["runs"][p] = fraud_env.run_episode(case, agent=p)
        except Exception as e:                                       # noqa: BLE001
            out["runs"][p] = {"error": f"{type(e).__name__}: {e}"}
    agent = run_episode(case)
    out["runs"]["llm_agent"] = agent
    if agent.get("available"):
        ranked = [(k, (v.get("scorecard") or {}).get("total_reward"))
                  for k, v in out["runs"].items()
                  if isinstance(v, dict)
                  and (v.get("scorecard") or {}).get("total_reward") is not None]
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        out["ranking"] = ranked
        best_ref = next((k for k, _ in ranked if k != "llm_agent"), None)
        out["beats_best_reference"] = bool(ranked and ranked[0][0] == "llm_agent")
        out["best_reference"] = best_ref
    return out
