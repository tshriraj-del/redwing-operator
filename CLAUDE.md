# CLAUDE.md

Governing standards for this repository. Adopted from **ECC** (`affaan-m/everything-claude-code`),
reconciled against REDWING's own hard-won conventions where the two disagree. Where a rule is
DEVIATED from, the reason is stated. A standard nobody can see the reasoning for is a standard
that gets quietly dropped.

## Project

REDWING operator: the FastAPI decision engine for a payment risk platform. Sits alongside
`~/pulseml_models` (redwing-ml, model training) and `~/redwing-fraud-os` (React console). The
repos deploy independently and do not import each other's application code, though this one
loads the ML repo's trained artifacts and shares its feature functions so train and serve
cannot diverge.

---

## Non-negotiables (REDWING-specific, these outrank everything below)

These came from measured failures in this codebase, not from a style guide.

1. **NEVER use em dashes** anywhere in output. Commas, periods, colons, or plain hyphens.
2. **Measure, do not assert.** Any claim about behaviour is backed by a run, not by reading the
   code. "The tests pass" is not "the endpoint returns the right answer": boot it and check.
3. **A guard is verified by mutation.** A test that has never failed proves nothing. Reintroduce
   the defect, confirm the test breaks, restore. Every guard protecting a real bug gets this.
4. **State what is real, modelled, and absent.** Synthetic data is fine and is the current stage;
   presenting it as production data is not. Sourced constants carry a citation; chosen ones say
   ASSUMPTION. The distinction is never blurred.
5. **Escalate-only layers may raise a score, never lower one.** Applies to the novelty gate, the
   consortium view, and the device gate. `max()`, never assignment.
6. **A control goes on every decision path or none.** See `docs/adr/ADR-001`. Six controls have
   now been wired to one path and forgotten on another. If a change touches `build_event()`,
   check `/score`, `_assemble_case()`, and `core/authorization.authorize()`.

---

## From ECC: adopted

### Workflow (`rules/common/development-workflow.md`)

- **Research and reuse BEFORE implementing.** `gh search repos` / `gh search code` for an existing
  implementation, then primary vendor docs, then broader web. Check PyPI before hand-rolling a
  utility. Prefer porting a proven approach to writing net-new.
- **Plan before executing.** Complex changes get broken into deliberate phases first.
- **Code review immediately after writing.** Address CRITICAL and HIGH; fix MEDIUM where possible.
- **Pre-review checks:** CI green, conflicts resolved, branch current with target, THEN request review.

### Security (`rules/common/security.md`): checked before every commit

- No hardcoded secrets. Environment variables only. Validate required secrets are present at startup.
- Validate all input at system boundaries. Never trust external data.
- Parameterised queries only.
- Rate limiting on endpoints.
- Error messages must not leak sensitive data.
- On finding a security issue: STOP, fix CRITICAL before continuing, then sweep the codebase for
  the same class.

### Git (`rules/common/git-workflow.md`)

- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`
  with an optional scope, as in `feat(card): ...`. The body stays as detailed as it has been;
  only the prefix changes.
- PRs: analyse the FULL history via `git diff <base>...HEAD`, not just the last commit. Include a
  test plan. Push new branches with `-u`.
- Verify a merge by CONTENT, not by ancestry. Squash-merges produce a new SHA, so
  `--is-ancestor` gives false alarms.

### Code quality (`rules/common/coding-style.md` + `rules/python/coding-style.md`)

- **PEP 8**, and PEP 8 wins over the common pack's `camelCase` rule, which is a JS/TS convention
  the language pack explicitly overrides.
- **Type annotations on function signatures.** `core/` is at 87%; new code is 100%.
- KISS, DRY, YAGNI. Extract when repetition is real, not speculative.
- **Functions under 50 lines. Files under 800.**
- No deep nesting past 4 levels; prefer early returns.
- No magic numbers. Named constants with the reasoning attached.
- Never silently swallow an error. Advisory layers may degrade, but they say so loudly.

### Testing (`rules/common/testing.md`): split by task type

Which discipline applies is decided by whether the contract is knowable before the work starts.

**TDD, strictly test-first, where the contract IS specifiable in advance:**
endpoints, validators, schema rules, response-code mappings, message normalisers, anything with a
stated input/output shape. Write the test, RUN IT AND WATCH IT FAIL, implement minimally, refactor.
A test that has never been seen red is not evidence.

**Measure-then-guard where the design EMERGES from the data:**
models, calibration, feature work, generator calibration, threshold selection. Test-first is not
reachable here because the measurement decides the design. The device gate is the worked example:
fan-out looked obviously right and measured 0.86x baseline, BELOW population. No test written in
advance could have encoded that, because the correct behaviour was not known until it was measured.

**Mutation verification is mandatory in BOTH modes.** Every guard protecting a real defect gets
the defect reintroduced, the failure confirmed, and the code restored. This is the part that is
not optional, and it is what makes the second mode as rigorous as the first.

Also:
- Descriptive test names stating the behaviour under test, not the function name.
- Unit AND integration coverage; the endpoint gets exercised, not only the function.
- Fix the implementation, not the test, unless the test is provably wrong.

### Agent delegation (`SOUL.md` principle 1): ADOPTED

Route work to specialists early rather than doing everything inline.

- **planner** before a multi-phase implementation.
- **code-reviewer** after writing a significant change, in a FRESH context. This is the specific
  thing ECC is right about: reviewing my own PRs in the same context found 5 real defects, but a
  cold reader would have reached the client-trust gap in the fingerprint layer faster, because I
  was anchored on having just written it.
- **security-reviewer** on anything touching auth, crypto, secrets, or untrusted input. The
  fingerprint collector qualifies; every value in it is attacker-controlled.
- **tdd-guide** for the specifiable-contract half of the testing split above.

Pass the relevant conventions from this file into the agent's prompt. An agent that has not been
told about the escalate-only rule or the mutation-verification requirement will not apply them.

---

## From ECC: deviated, with reasons

**Immutability as CRITICAL / never mutate.** ECC mandates new objects over in-place changes.
REDWING's scoring path builds one `event` dict incrementally across roughly a dozen stages, and
converting it to frozen dataclasses would be a rewrite of a working hot path measured at 4.1ms
p50. ECC's own YAGNI rule argues against it. **Adopted in spirit, and verified rather than
assumed**: no function mutates a caller's argument. Checked by deep-copying a row, calling
`build_event`, and comparing: the dict comes back unchanged. Shared module state is not written
at request time either. **Not adopted**: frozen dataclasses throughout.

**A single 80% coverage floor across everything.** Now measured, and the blanket form is the wrong
target. A per-surface floor is adopted instead; the numbers and reasoning are below.

---

## Coverage: measured, and the floor is per-surface

Measured 2026-08-12 with `coverage` 7.15.4, branch coverage on. Config is `.coveragerc`.

> **STALE as of 2026-08-15 and NOT re-measured.** The table below predates the dispute rail, the
> card sequence gate, the card durable record, and the `/score` rail branch. `main.py` has grown
> 348 lines and the suite has grown from 362 to 460 tests since. Do not quote these percentages
> until the command below has been re-run. They are left in place rather than deleted so the
> re-measurement has a baseline to move against.

**Read the config comment before quoting any number.** The first run listed
`source = core,main.py,...` and coverage SILENTLY IGNORED the `main.py` entry, because `source`
takes package and directory names, not file paths. That produced a clean-looking **88%** which
excluded the 3,667-line file holding `build_event`, `/score` and `/authorize`. A coverage figure
that omits the hot path is worse than none, because it is quotable. `source = .` now.

Headline, statements plus branches: **operator 69%, redwing-ml 52%.** Both below ECC's 80%.
Decomposed by statements only, which is where the structure shows:

| Surface | Statements | Covered | Floor |
|---|---|---|---|
| `core/` (decision logic, stdlib) | 3,715 | **89%** | **85%, enforced** |
| `main.py` (API surface, the 4 pipelines) | 1,450 | **36%** | **60% target** |
| `integrations/` (14 UNCONFIGURED connectors) | 503 | 66% | none |
| everything else (agent, rule_factory, xai) | 1,207 | 67% | none |
| redwing-ml, libraries imported at serve time | 1,304 | 71% | 75% target |
| redwing-ml, one-shot pipeline scripts | 746 | 23% | none |

**Why not a blanket 80%.** It would push effort toward testing one-shot training scripts, which
run under human supervision and fail loudly, and away from `main.py`, which serves live decisions.
ECC's own KISS and YAGNI rules argue for the floor that changes behaviour rather than the floor
that is easiest to state.

**`main.py` at 36% is the real finding here**, and it is the same file ADR-001 is about. The
decision logic is well covered because it lives in `core/` where it is testable without the ML
stack; the pipelines that CALL that logic are not, because they are inlined in a 3,667-line
module behind a model load. Extracting the decision core raises coverage as a side effect rather
than as a separate campaign.

```bash
rm -f .coverage .coverage.*
for f in tests/test_*.py; do .venv/bin/python -m coverage run "$f" >/dev/null 2>&1; done
.venv/bin/python -m coverage combine && .venv/bin/python -m coverage report --sort=cover
```

---

## The silent-degradation class

Measured 2026-08-15. This is now the repo's most repeated defect and it deserves its own rule:
**the code cannot distinguish "I could not look" from "I looked and there is nothing."**

Three confirmed instances, all shipped, all found in one day:

1. `core/card_history.sequence_view` swallows a read failure and returns the no-history view,
   while `apply_sequence_gate` sets `available: True` unconditionally. A gate that saw nothing
   reports as one that looked. Measured: 8 verified prior authorizations, gate reported
   `burst_24h: 0.0, card_known: false, available: true`.
2. `core/record.row_from_backbone` returns `None` on any read failure and `main.py` `get_case()`
   turns that into `HTTPException(404)`. An investigator pulling a live case cannot tell
   "never ingested" from "the store was briefly unreadable". This one lands on a human.
3. `get_network_graph` and `get_typologies` return empty node/link sets on a CSV read failure,
   which reads to an analyst as "no fraud network exists".

**The rule:** an advisory layer may degrade, but its OUTPUT must carry the degradation. Absence
of evidence is never rendered as evidence of absence. Concretely: return a `read_ok` / `degraded`
flag beside the value, set `available` from that flag and never unconditionally, and bind the
exception (`except Exception as e:`) with a log line.

The correct pattern already exists in this repo, twice: `core/screening.py` stores `self.error`
and surfaces "screening is unavailable" through both `status()` and the blocked reason, and
`integrations/hub.py` binds and logs. Copy those, do not invent a third shape.

## Known violations of the adopted rules

Recorded rather than hidden, so they read as debt with a plan and not as standards nobody follows.

| Rule | Violation | Plan |
|---|---|---|
| Files < 800 lines | `main.py` **3,667**; `core/store.py` **1,142**; `tests/test_core_store.py` 1,680 | ADR-001 extracts the decision core out of `main.py` |
| Functions < 50 lines | 53+ exceed it; `build_event` is 314, `score` **147**, `get_network_graph` **124**, `store.add_label` **~90** | Same |
| Errors never silently swallowed | **24 bare `except Exception:` in `main.py`, and `main.py` imports no logger at all** | See "The silent-degradation class" below |
| Type annotations | `core/` at 87% return-annotated | New code 100%; backfill opportunistically |
| `core/` >= 85% | 89%, holds | keep it there |
| `main.py` >= 60% | **36%** | ADR-001 extraction raises it as a side effect |

---

## Commands

```bash
# operator tests (stdlib runner, no pytest needed)
for f in tests/test_*.py; do .venv/bin/python "$f"; done

# one suite
.venv/bin/python tests/test_fingerprint.py

# the ML repo shares this venv
cd ~/pulseml_models && ~/redwing-operator/.venv/bin/python tests/test_device_graph.py
```

## Architecture

- `core/`: pure-stdlib decision logic, testable without the ML stack. This boundary is
  load-bearing: it is what lets the bulk of the 460-test suite run without loading a model.
- `main.py`: FastAPI surface, model loading, the scoring pipelines.
- `integrations/`: external connector interfaces. 15 registered; an unconfigured one now
  returns `DERIVED` synthetic signals via `hub._safe_enrich`, not a bare `UNCONFIGURED`.
- `docs/adr/`: architecture decision records. `docs/` is gitignored (candid, local only).

## Reference

ECC clone for consultation: `scratchpad/ecc`. Read `rules/common/` and `rules/python/` before
inventing a convention. Skills catalogue is in `skills/`, agents in `agents/`.
