# Logic Platform Canonicalization Plan

**Status:** Active campaign  
**Prefix:** `LPC`  
**Root goal:** `LPC-G000`  
**Branch:** `agent/logic-platform-canonicalization`  
**Scheduler:** `config/agent_supervisor_logic_platform_canonicalization_scheduler.json`

This plan is the human-readable companion to the supervisor objective heap,
todo board, and `FormalWorkPlan@1`. It does **not** create another theorem
prover, agent supervisor, logic-family registry, receipt format, or MCP++
profile.

## 1. Ownership

| Authority | Owner |
| --- | --- |
| Family, property, profile, notation, evidence, translation, verdict, receipt, cache identity, formalization meaning | `ipfs_datasets_py.logic` |
| Scheduling, isolation, resources, model routing, worktrees, leases, cancellation, workflow | `ipfs_accelerate_py.agent_supervisor` |

The supervisor may decide **when and where** work executes. It must not
redefine **what the work means**.

## 2. Inspected revisions

| Repository | Reviewed baseline | Current head (implementation authority) | Delta |
| --- | --- | --- | --- |
| `endomorphosis/ipfs_datasets_py` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` at `/home/barberb/lift_coding/external/ipfs_datasets` | Exact match |
| `endomorphosis/ipfs_accelerate_py` | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` | `ea11293bb996f052d620eae989f5377a956764b1` at this worktree | Current head is **1,245 commits behind** the reviewed baseline; merge-base is current HEAD |

Use the current checked-out sources as implementation authority. Record
intervening changes in `LPC-001` before any production edit.

Existing campaigns already running on this host (do not join or steal their
state): incremental-proof-sealer-v1 and state-laws-reindex.

## 3. Canonical pipeline

```
SourceDocument
  → syntax_core parse artifact
  → TypedExpression
  → ElaborationArtifact
  → FormalizationArtifact@3
  → DomainLogicSlice@2
  → LogicObligation@2
  → BackendRequest@2
  → provider execution under supervisor resources
  → typed EvidenceArtifact
  → translation validation / reconstruction / kernel checking
  → TrustedProofReceipt or typed non-proof outcome
  → supervisor admission, validation, merge policy
```

Every transition preserves source, expression, namespace, assumptions, finite
bounds, translation guarantees, provider, environment, evidence kind,
authority ceiling, cache provenance, and receipt identity.

No stage may silently convert a candidate into a proof, a model into a
theorem, provider success into semantic success, a registry entry into
executability, an importable package into a working provider, a structurally
valid receipt into authenticated evidence, a timeout into a pass, a heuristic
translation into an exact one, or a free-form family string into a canonical
family.

## 4. Preserve existing strong contracts

Datasets: `syntax_core`, `families.models/namespaces/registry/registry_v3`,
`profile_catalog_v3`, `provider_matrix_v2`, `formalization.artifacts_v3`,
`backends.requests_v2`, `backends.provider`, `software_verification`, domain
slices (legal, security, intent, crypto, software, UI/UX),
`verification_api`, tactician and counterexample facilities.

Supervisor: `proof.canonical_logic_adapter`, `logic_provider_contract`,
`formal_verification_provider`, `formal_verification_cache`,
`multi_prover_router`, `prover_matrix_registry`,
`integrations.ipfs_datasets_logic_provider`, plus scheduler, resource
manager, admissibility, cancellation, lease, fencing, evidence, and
attestation.

## 5. Goal / subgoal / task DAG

Supervisor-consumable artifacts:

- Goals: `docs/architecture/logic_platform_canonicalization.objectives.md`
- Tasks: `docs/architecture/logic_platform_canonicalization.todo.md`
- FormalWorkPlan: `data/agent_supervisor/logic_platform_canonicalization/formal_work_plan.json`

```
LPC-G000  root
  LPC-G010  inventory          [blocks all refactors]
    LPC-001..007  parallel inventory slices
    LPC-008       compose index
  LPC-G020  catalog            ─┐ after G010, parallel with G030
  LPC-G030  status axes        ─┘
  LPC-G040  formalization      ─┐ after G020+G030, parallel with G050
  LPC-G050  provider protocol  ─┘
  LPC-G060  API split             after G050
  LPC-G070  tactician          ─┐ after G040+G050
  LPC-G080  cache              ─┘ after G030+G050
  LPC-G090  supervisor maps    ─┐ after G020+G030
  LPC-G100  package manifest   ─┘ after G020
  LPC-G110  supervisor client     after G050+G090+G100
  LPC-G120  hammer adapter        after G020+G050+G090
  LPC-G130  CLI/MCP parity        after G060+G110
  LPC-G140  tests                 after G040+G050+G080+G110
  LPC-G150  packaging/CI          after G140
  LPC-G160  documentation         after G150
  LPC-G170  final report          after G160
```

Parallel lanes after inventory:

| Lane | Owns | First ready tasks |
| --- | --- | --- |
| `lpc-inventory-*` | inventory artifacts only | LPC-001..007 |
| `lpc-catalog` | datasets `logic/families` snapshot | LPC-020 |
| `lpc-axes` | datasets `logic/ir_core/axes` | LPC-030 |
| `lpc-formalization` / `lpc-domain-*` | artifacts_v3 + domain slices | LPC-040..043 |
| `lpc-provider` | `logic/backends/protocol_v2` | LPC-050 |
| `lpc-api` | `logic/platform/*` facades | LPC-060 |
| `lpc-tactician` | `logic/tactician` | LPC-070 |
| `lpc-cache` | `logic/common` cache key | LPC-080 |
| `lpc-supervisor-*` | adapter, client, admission | LPC-090, LPC-110 |
| `lpc-hammer` | `integrations/ipfs_datasets_logic_provider.py` | LPC-120 |
| `lpc-parity` / `lpc-tests*` / `lpc-packaging` / `lpc-docs` / `lpc-report` | later waves | after gates |

Do not allow two agents to edit the same canonical contract file.

## 6. Epics (A–O)

These map 1:1 onto goals G020–G160.

**A. Canonical catalog.** Compose, do not flatten. v2 = taxonomy, v3 =
lifecycle, snapshot = composition. No registry v4 rename.

**B. Orthogonal axes.** Operation status ≠ semantic verdict ≠ availability ≠
evidence kind ≠ evidence authority ≠ boundedness ≠ translation preservation.

**C. Typed new-write path.** Source → syntax_core → TypedExpression →
Elaboration → FormalizationArtifact@3 → DomainLogicSlice@2. Keep TDFOL, DCEC,
and frame logic distinct. Keep software/security families distinct. No new
families.

**D. LogicProviderProtocol@2.** Operation-specific requests. v1 generic
payloads cannot bypass BackendRequest@2. Finite positive bounds on executable
ops.

**E. Split verification_api.** Internal `logic/platform/*`. Public imports
stay as facades. Pure-data imports are side-effect free.

**F. One proof-plan model.** Advisors propose; they do not prove.

**G. One cache-key contract.** Datasets owns semantics. Supervisor owns
placement and single-flight.

**H. Generated supervisor maps.** Keep the lazy adapter. Stop hand lists.

**I. LogicPlatformManifest@1.** Wheels, separate envs, no Git, no siblings.

**J. SupervisorLogicPlatformClient.** One handshake + typed invocation +
ten-point receipt admission.

**K. Hammer adapter cleanup.** Candidate-producing until reconstruction.

**L. Channel parity.** Python = CLI = MCP. Installation is not verify.

**M. Test matrix.** Hermetic required vs installed-provider optional vs
network/heavy opt-in. Mocks cannot satisfy real-provider gates.

**N. Packaging and CI.** Independent installs. Required lanes fail on
failure. No `continue-on-error` / `|| true`.

**O. Documentation.** Generated tables. Label maturity. No invented
availability or production-readiness.

## 7. Execution constraints

For every task:

1. Identify owned files.
2. Run pre-change focused tests if they exist.
3. Implement the smallest coherent change.
4. Run focused tests and affected integration tests.
5. Record result and unresolved risks.

Inventory before refactor. Fail closed on timeout, unavailable provider,
unknown verdict, unknown label, stale cache, simulated evidence, and
authority overclaim.

## 8. Receipt admission (supervisor)

A proof result may influence completion or merge only when all hold:

1. Receipt is structurally valid.
2. Content identity is valid.
3. Source/tree/environment/policy bindings match.
4. Translation chain is valid.
5. Evidence kind supports the claimed verdict.
6. Authority ceiling is adequate.
7. Required reconstruction or kernel checks passed.
8. It is not stale.
9. It is not simulated.
10. Supervisor policy admits that authority for the requested operation.

## 9. Launch

From this worktree:

```bash
python scripts/validate_logic_platform_canonicalization_board.py --check-all
python scripts/ops/agent_supervisor/configured_board_scheduler.py \
  --repo-root . \
  --config config/agent_supervisor_logic_platform_canonicalization_scheduler.json \
  preflight
python scripts/ops/agent_supervisor/configured_board_scheduler.py \
  --repo-root . \
  --config config/agent_supervisor_logic_platform_canonicalization_scheduler.json \
  launch --implement
```

Provider is selected at launch (`grok_cli` / `grok-4.5` with quota-only
Codex fallback). Provider choice is not encoded in goal text.

## 10. Required final claim

Do not claim the whole logic platform is production-ready because the
refactor and tests complete. The only acceptable closing claim is:

> ipfs_datasets_py.logic now provides the canonical typed semantic,
> formalization, provider, evidence, and verification contracts. The
> ipfs_accelerate_py agent supervisor consumes those contracts through one
> lazy, version-negotiated boundary while retaining operational ownership of
> scheduling, isolation, resources, cancellation, leases, model routing and
> workflow state. Direct and supervisor-mediated verification are qualified
> against the same catalog, typed requests, evidence semantics and receipt
> identities for the tested providers and logic slices.

## 11. Out of scope

- New logic families
- New theorem provers
- Another agent framework or GUI
- Another MCP++ profile
- Automatic provider installation
- Default network access
- LLM output granting proof authority
- Duplicate semantic authorities in the supervisor
- Replacing typed contracts with free-form dictionaries
- Deleting compatibility APIs without adapters
- Silent normalization of unknown labels
- Reporting simulations as live verification
- Weakening fail-closed admission
- Unrelated portfolio cleanup
