# agent_supervisor.runtime

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Multi-lane execution fabric: multi-supervisor runners, event log, CAS, and resource/batch schedulers.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, merge/rescue coordination surfaces |
| **Typical dependents** | todo_daemon, ops scripts |

## Modules

| Module | Path |
| --- | --- |
| `artifact_store` | `runtime/artifact_store.py` |
| `configured_board_scheduler` | `runtime/configured_board_scheduler.py` |
| `context_selection_observation` | `runtime/context_selection_observation.py` |
| `event_log` | `runtime/event_log.py` |
| `intent_requirement_context` | `runtime/intent_requirement_context.py` |
| `intent_requirement_observation` | `runtime/intent_requirement_observation.py` |
| `learning_checkpoint` | `runtime/learning_checkpoint.py` |
| `local_planning_admission` | `runtime/local_planning_admission.py` |
| `multi_supervisor_runner` | `runtime/multi_supervisor_runner.py` |
| `operational_campaign` | `runtime/operational_campaign.py` |
| `provider_batch_scheduler` | `runtime/provider_batch_scheduler.py` |
| `resource_scheduler` | `runtime/resource_scheduler.py` |
| `role_context_selection` | `runtime/role_context_selection.py` |
| `router_public_instruction` | `runtime/router_public_instruction.py` |
| `runtime_cas` | `runtime/runtime_cas.py` |
| `runtime_temporal_monitor` | `runtime/runtime_temporal_monitor.py` |
| `scheduler_metrics` | `runtime/scheduler_metrics.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.runtime.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

`intent_requirement_context` projects exact accepted source and task bindings for
worker delivery. `router_public_instruction` verifies their signed public
admission without owner keys and includes them after semantic encoding.
`intent_requirement_observation` reads actual native owner validation records,
checks event order and source/revision/attempt freshness, and reports public
checks and output presence separately. These projections do not prove source
meaning, expand task scope, change canonical state or grant completion authority.

`semantic_context_runtime.prepare_semantic_context` records the final capsule
selection after byte-budget repacking in `context-selection.json`. The sidecar
binds the unchanged worker payload, source handles, native capsule/admission
identities, declared dependencies and obligation references. Its reference is
stored in the existing world observation and metadata catalog, with the existing
DuckLake projection. It is host metadata and adds no text to the worker prompt.
The observation distinguishes structural Codebase IR evidence from modality
advisors that did not run at this preparation stage. It does not establish
complete dependencies, current source at a later dispatch, provider delivery,
proof, or token savings. `verify_context_selection_observation` replays the
observation against a caller-supplied trusted producer payload; callers still
use the native semantic loader for task and source freshness checks.
If the observation exceeds its own resource bounds, preparation records a fixed
unavailable status and publishes no partial sidecar. The native context and its
acceptance requirements remain unchanged. Oversized worker payloads skip
observation decoding.

`role_context_selection` prepares separate, immutable host artifacts for an
explicit planning, coding or review profile. A profile binds its policy revision,
query, capsule limit, byte limit and optional verified retrieval artifact. It
uses the existing lexical or independently replayed retrieval selection; it is
not a learned role ranker. Required raw sources and native admission checks stay
in effect. Changing only the owner-declared observation role leaves worker bytes
unchanged. The role appears in the existing selection observation and its
DuckDB/DuckLake metadata reference; `role-context.json` additionally binds the
profile and exact worker/observation artifacts outside the worker prompt.

Use separate fresh output directories, and retain the returned reference and
the independently chosen profile:

```python
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.runtime.role_context_selection import (
    RoleContextProfile, prepare_role_semantic_context, load_role_semantic_context,
)

repository = Path("/path/to/repository").resolve()
profile = RoleContextProfile(
    role="planning", query="public contract", capsule_limit=2,
    policy_id="repair-context", policy_revision=1,
)
reference = prepare_role_semantic_context(
    profile=profile, repository=repository,
    paths=["code.py", "instruction.md"], program_paths=["code.py"],
    required_raw_paths=["instruction.md"],
    objective="Repair the program while preserving its public contract.",
    task_id="repair-001", output=repository / ".semantic" / "planning-v1",
)
worker_bytes = load_role_semantic_context(
    repository=repository, artifact=reference["artifact"],
    expected_sha256=reference["sha256"], task_id="repair-001", profile=profile,
)
```

The loader checks the expected profile, native producer reconstruction, current
sources, selection controls and available observation bytes. Stale profiles
require explicit preparation into another fresh directory; the existing native
refresh API does not carry this host profile. An unavailable observation remains
explicitly unavailable. Neither a role label nor a successful load establishes
provider delivery, semantic relevance, modality coverage, proof or token savings.
This API does not change the benchmark's current planning/coding artifact reuse
or add a review provider route. The peer repository context compiler and planning
consumer remain the intended integration point for mandatory domain evidence,
question-bound expansion and retry deltas.

`ContextCompiler` enforces the provider byte ceiling on both transmitted retry
deltas and reconstructed full inputs, including consumer reverification. These
checks supplement the existing token, invariant-core and required-coverage gates.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/runtime.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
