# agent_supervisor.planning

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Adaptive and formal planning: compile, validate, replan, metrics, and rollout helpers (non-daemon).

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, `objectives` contracts, proof contracts as needed |
| **Typical dependents** | todo_daemon, validation, proof-carrying flows |

## Modules

| Module | Path |
| --- | --- |
| `adaptive_planner` | `planning/adaptive_planner.py` |
| `campaign_prompt_handoff` | `planning/campaign_prompt_handoff.py` |
| `formal_plan_compiler` | `planning/formal_plan_compiler.py` |
| `formal_plan_conformance` | `planning/formal_plan_conformance.py` |
| `formal_plan_context` | `planning/formal_plan_context.py` |
| `formal_plan_validator` | `planning/formal_plan_validator.py` |
| `formal_planning_adversarial` | `planning/formal_planning_adversarial.py` |
| `formal_planning_contracts` | `planning/formal_planning_contracts.py` |
| `formal_planning_metrics` | `planning/formal_planning_metrics.py` |
| `formal_planning_rollout` | `planning/formal_planning_rollout.py` |
| `formal_replanner` | `planning/formal_replanner.py` |
| `ir_learning_campaign_planner` | `planning/ir_learning_campaign_planner.py` |
| `intent_requirement_adapter` | `planning/intent_requirement_adapter.py` |
| `intent_requirement_repair` | `planning/intent_requirement_repair.py` |
| `intent_symbolic_planning` | `planning/intent_symbolic_planning.py` |
| `plan_evaluator` | `planning/plan_evaluator.py` |
| `plan_failure_memory` | `planning/plan_failure_memory.py` |
| `proof_carrying_planner` | `planning/proof_carrying_planner.py` |
| `repository_planning_context` | `planning/repository_planning_context.py` |
| `repository_planning_evidence` | `planning/repository_planning_evidence.py` |
| `repository_planning_artifacts` | `planning/repository_planning_artifacts.py` |
| `task_proposal_router` | `planning/task_proposal_router.py` |
| `task_quality` | `planning/task_quality.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.planning.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

`intent_requirement_adapter` checks reviewed native atom matchers and exact
signed task operations, then supplies administrative coverage predicates and
producer rules to existing symbolic stages. `intent_symbolic_planning` joins
obligation compilation, candidate selection, critique, exact task projection,
coverage and formal output checking. Both produce proposals; callers verify
source and manifest authority independently. They supply no observed source
facts and grant no semantic, execution or completion authority.

`intent_requirement_repair` consumes a bounded, identified public-check
observation and nominates validation or repairs using `ResidualLlmPacket`.
Repairs retain signed output paths, validation commands and intent revision;
dependent tasks are nominated for revalidation. No nomination dispatches work,
admits a task, changes source interpretation or grants write/completion authority.

## Repository context in the task proposal consumer

`TaskProposalRouterConfig.repository_context` accepts a
`RepositoryPlanningContext` with a caller-owned `RepositorySourceSession`, its
exact generation, a reviewed native Intent requirement contract, a retrieval
query and an unresolved question. For an existing config:

```python
from dataclasses import replace
from ipfs_accelerate_py.agent_supervisor.planning.repository_planning_context import RepositoryPlanningContext
from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import run_task_proposal_router

configured = replace(config, artifact_dir=external_artifacts,
    repository_context=RepositoryPlanningContext(
        session=session, generation=session.generation,
        intent_contract=reviewed_contract, query="answer",
        unresolved_question="Which source implements the requested behavior?"))
preflight = run_task_proposal_router(configured, generate=False)
```

The session workspace must equal `repo_root`; the task board, roadmap and Intent
source must belong to its snapshot. Artifacts must be outside both the workspace
and private session state. Source inputs use exact original bytes even when the
session offers a compact view. The complete validated Intent contract and
explicit Security/UI status are required evidence. Optional roadmap chunks and
native capsule handles use the existing budget selector. The bounded profile
refuses excess populations or required-context overflow instead of truncating
obligations: at most 32 Intent ledger rows, 60 KiB per domain body, 32 roadmap
references and 32 query candidates, within a 256 KiB compiler ceiling and the
existing 60,000-character provider prompt limit.

Optional paired `security_scope` / `security_analyses` inputs replay the native
scoped SecurityIR owner against current sources and retain its full compilation
report, including unsupported and unmodeled scope. This is declaration status,
not full security-policy expansion or source satisfaction. UI/UX is explicitly
unavailable here. Learned CodebaseIR training and automatic intent extraction
are not supplied by this adapter.

An explicit `expansion_reference_ids` tuple can request native capsule bodies
for the bound question. The existing consumer is stateless: each call sends the
complete reconstructed context, including all required evidence. Compiler
deltas are retained as receipts, not substituted for a full provider prompt.
`prompt_bytes` and, when a tokenizer is supplied, `prompt_tokens` measure exactly
that rendered input. `context_input_tokens` is conservative budget accounting.
Neither field includes provider-specific envelopes or establishes billed usage.

Shared resource admission, provider usage routing and batch backoff remain in
their existing owners. Source freshness is checked inside the consumer dispatch
callback, after its response, and inside artifact-write callbacks. Queueing or
provider admission can occur after the first check; a changed source can still
reach physical dispatch, but its response is refused before publication. These
cooperative checks are not a transaction or hard memory sandbox. Artifacts use
bounded, no-follow, atomic writes to prevent a child symlink from overwriting
source or session state. No raw prompts or rejected provider responses are
persisted in context receipts.

The repository binding is explicit; callers without one retain their existing
router behavior. This host-side integration does not establish deployment or
live Terminal Bench worker adoption. The authored
`terminal_repository_planning_experiment` measures the actual consumer with
stub providers and preserves separate input costs for every stateless call.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/planning.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
