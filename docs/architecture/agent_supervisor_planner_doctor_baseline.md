# Planner/Doctor Mainline Capability Baseline

Status: frozen PDR-001 implementation baseline

Interface: `PlannerDoctorCapabilityInventory@1`

This document is the human projection of the provider-free capability
inventory in
`ipfs_accelerate_py.agent_supervisor.analysis.planner_doctor_capability_inventory`.
The canonical runtime record is content-addressed and replay-checked; this
Markdown is not proof, completion, mutation, or promotion authority.

## Identity boundary

The baseline and the PDR control inputs are intentionally separate:

| Observation | Commit | Tree | Meaning |
| --- | --- | --- | --- |
| Audited implementation | `f25e5719cb738a50fb96bac4bea3f66ebca9800b` | `cf6f6450bba0126a9846ae88bf7764f3e1ccb7c3` | Exact `origin/main` implementation audited for shipped and default-wired behavior |
| PDR seed plan | `d56ded212b6c5ecc50e60aec89b8d8bb9c8df552` | `8a0d69b96e14313d43e2d426169c8d937c5bb6f0` | Plan, objective heap, seed taskboard, and scheduler inputs only |
| Live checkout | Observed at inventory time | Observed at inventory time | Current HEAD plus a body-free dirty-overlay CID and current recursive Gitlink frontier |

The inventory requires an explicit `audited_ref`; it never labels worktree
HEAD as the audited mainline implicitly. It reads audited source with
`git cat-file <commit>:<path>`, so the PDR implementation files and dirty
worktree cannot participate in their own audited source identity.

The audited superproject tree contains ten Gitlinks:

| Gitlink | Recorded commit |
| --- | --- |
| `docs/fastmcp` | `1d932cc778a24cc0bf46fc4baad8306d4fed9c4b` |
| `docs/mcp-python-sdk` | `0da9a074d09267a927d72faa58c26d828f0f8edb` |
| `ipfs_accelerate_py/mcplusplus` | `15c1816d6c63a2b11edd505704f6a04a9abc6167` |
| `ipfs_datasets_py` | `7415adc5100192ee35676778f1018f6b072378f9` |
| `ipfs_kit_py` | `f6a574375febbcf9a46fcd24bbc7bc5cfb551de5` |
| `ipfs_model_manager_py` | `f6151d2113f42e75ea7d83a1b2362fc97e55e44d` |
| `ipfs_transformers_py` | `b397988ed9e3e656475c1cf4417b84efdb95daf3` |
| `test/doc-builder` | `6108e850ae1cf2f71bb0815a600bcd50c39abfa7` |
| `test/huggingface_doc_builder` | `6108e850ae1cf2f71bb0815a600bcd50c39abfa7` |
| `test/huggingface_transformers` | `44752c8dd99f3fb0da23006dc4fde4a07d9c417f` |

Every known Gitlink path and commit remains bound even when its checkout is
unavailable. Available child repositories are traversed at the recorded
commit. An unavailable child or nested frontier sets
`gitlink_closure_complete=false`; it is never silently treated as an empty
closure.

## Shipped versus default-wired inventory

`shipped` means the audited file/interface exists. It does not mean the
normal service constructor reaches it. `partial` and `missing` describe
implementation presence; the separate wiring column describes the default
runtime path.

| Capability ID | Availability | Default wiring | Mainline evidence and residual |
| --- | --- | --- | --- |
| `prompt.repository_analysis` | shipped | unwired | `PromptSupervisorService.optional_analysis` defaults to `None`; normal prompt scans do not receive the reasoning registry |
| `prompt.independent_plan_admission` | shipped | unwired | `admission_request_factory` defaults to `None`; compiled plans reach the independent-IR-required gate without a request |
| `planner.create_steer_revision` | missing | unwired | No `PlanRevision`, `PlanDelta`, create, or steer contract module exists |
| `planner.parallel_execution_plan` | missing | unwired | No `ParallelPlanCompiler` or `ParallelExecutionPlan` exists |
| `doctor.production_stage_backends` | shipped | unwired | `DeterministicDoctorService(backends=None)` constructs an empty `DoctorStageBackends`; normal diagnose/plan/transaction stages abstain |
| `doctor.snapshot_contract_bridge` | partial | incompatible | Repository diagnostics and deterministic Doctor use different snapshot/finding schemas and no checked adapter joins them |
| `doctor.pinned_proof_authority` | shipped | unwired | Hammer/proof components exist, but the default Doctor path does not require a pinned native solver and independent theorem reconstruction |
| `doctor.live_transaction` | partial | unsafe stub | Default applicator reports planned paths without changing bytes and default restore returns `True` without rereading bytes |
| `doctor.live_fixed_point` | partial | unsafe stub | Validator consumes supplied stage evidence and default restore returns `True`; no live reparse/static/security/replan/reprove runner exists |
| `benchmark.live_paired_runner` | partial | synthetic only | Deterministic Doctor and generation-2 fixture producers exist; no paired runner invokes real Planner and Doctor services |
| `self_improvement.derived_refill` | partial | disabled | PDR scheduler has `derived_refill.enabled_at_bootstrap=false`; runtime refill remains gated until PDR-081 |
| `runtime.cold_import_hygiene` | partial | incompatible | At the audited commit, package-root import eagerly reaches `hf_space_inference`, which imports `requests`; skip-core and normal profiles are therefore distinct |

The inventory records every observation with an audited blob CID, compact
detector result, interface ID, test paths, configuration paths, and typed
reason codes. It stores no source bodies, prompts, proof transcripts, raw
tool logs, credentials, timestamps, absolute checkout paths, or provider
claims.

## Task, goal, test, and configuration status

At the committed PDR seed:

- The taskboard contains 43 unique `PDR-###` tasks.
- `PDR-000` is the only completed task; the remaining 42 are pending.
- The objective heap contains 11 `PDR-G###` goals rooted at `PDR-G000`.
- The scheduler is report-only/shadow: Doctor execution and mutation are
  disabled, automatic rollout is disabled, remote/model/network use is
  disabled for deterministic Doctor, and derived refill is disabled.

Runtime inventory reads the taskboard, objective heap, and selected JSON
configuration twice from the live checkout. It records their raw blob CIDs
and a closed safe projection of status fields. Changes to a task status,
goal status, configuration, test blob, source blob, HEAD, tree, dirty
overlay, or Gitlink change the inventory CID.

Tests are observations, not completion authority. Relevant existing tests
and missing PDR successor tests are recorded separately as `present` or
`missing`; the mere existence of a test file never upgrades a capability to
`wired`.

## Optional tool health

The inventory imports and executes no optional provider. Callers may inject
metadata-only probes for theorem provers, AST/program-analysis providers,
knowledge graphs, vector indexes, and ZKP tooling. Each result must state
that it used no network and started no process. Missing probes become
`not_probed`, not `available`.

Tool health remains a routing observation. It cannot:

- prove a theorem or security property;
- make an incomplete repository/Gitlink scan complete;
- upgrade a shipped-but-unwired runtime path;
- authorize mutation, promotion, or task/goal completion; or
- replace a current-tree independent validation.

## Canonical use

```python
from ipfs_accelerate_py.agent_supervisor.analysis.planner_doctor_capability_inventory import (
    PDR_AUDITED_BASELINE_COMMIT,
    build_planner_doctor_capability_inventory,
    replay_planner_doctor_capability_inventory,
)

inventory = build_planner_doctor_capability_inventory(
    repository_root,
    audited_ref=PDR_AUDITED_BASELINE_COMMIT,
    tool_probes=metadata_only_probes,
)
record = inventory.to_record()
replayed = replay_planner_doctor_capability_inventory(record)
assert replayed.content_id == inventory.content_id
```

The canonical record uses CIDv1 DAG-JSON/sha2-256 identities compatible with
the formal-verification contracts. Arbitrary file bodies use raw CIDv1
digests and are never embedded. Unknown fields, floats, private configuration
keys, unsafe paths, malformed object IDs, mixed live observations, forged
nested revision IDs, forged gap projections, and forged top-level CIDs fail
closed.

## PDR-001 validation

```bash
python -m pytest \
  test/api/test_agent_supervisor_planner_doctor_capability_inventory.py -q
```

The focused fixture constructs a real nested Git submodule, freezes an
audited commit, adds a dirty live overlay, changes task/config state, injects
metadata-only tool health, and verifies deterministic identity, body-free
serialization, cold import, strict replay, and tamper rejection.
