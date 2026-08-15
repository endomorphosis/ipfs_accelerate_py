# Verified GUI Optimizer

**Status:** Current
**Audience:** Developers, operators, and agents running or extending
VerifiedGuiOptimizer
**Scope:** Cross-repository architecture, security, journaled improvement
loop, CLI, evidence taxonomy, standalone dependency boundary, and exact
prerequisites for one additional application
**Non-goals:** Rewriting benchmark outcomes; auto-releasing to users;
broadening optimization to every SwissKnife application; presenting a
forbidden prior subsystem as an implementation dependency; claiming
aesthetic optimality, full WCAG, or complete security
**Interfaces:** `VerifiedGuiOptimizerArchitecture@1`,
`VerifiedGuiOptimizer@1`, `GuiApplicationAdapter@1`,
`GuiEvidenceAuthorityMatrix@1`
**Selected screen:** Agent Supervisor console
**IDs:** `app:agent-supervisor`, `screen:agent-supervisor`,
`route:agent-supervisor`, CLI target `agent-supervisor`
**Primary source:** `swissknife/web/js/apps/agent-supervisor.js`
**Companions:** `swissknife/docs/gui-optimizer/ARCHITECTURE.md`,
`external/ipfs_datasets/docs/gui_optimizer_contracts.md`

VerifiedGuiOptimizer is a standalone, evidence-driven GUI improvement
subsystem. Formal machinery is used only for bounded structural and
transition properties.

## 1. Purpose and selected screen

For one bounded screen and one or a few declared objectives per iteration
the subsystem:

1. statically derives versioned component, state, action, layout,
   accessibility, route, policy, style, localization, test, and screenshot
   facts;
2. builds a typed component/state graph and compact `UiSemanticCapsule@1`
   records;
3. calculates an explicit `UiInvalidationPlan@1`;
4. creates a token-accounted `UiContextPack@1`;
5. accepts a bounded `GuiImprovementProposal@1` or obtains one through the
   provider-neutral `GuiPatchProposer@1` (caller-selected route; not a
   model router);
6. applies the proposal only in an isolated Git worktree;
7. runs affected checks first and expands when extraction confidence or
   impact closure is incomplete;
8. compares state, interactions, accessibility, screenshots, and metrics
   against a content-addressed baseline;
9. rejects regressions in required interaction, accessibility,
   confirmation, policy, security, and scope invariants;
10. emits a content-addressed `GuiImprovementReceipt@1`.

The selected screen is the SwissKnife **Agent Supervisor** console.

## 2. Package boundaries

```text
swissknife/src/services/gui-optimizer/*     observation + evaluation
        ^  closed JSON (schema + interface + evidence labels)
ipfs_datasets_py/logic/gui_optimizer/*      contracts + formal/structural
        ^  closed JSON
ipfs_accelerate_py/agent_supervisor/
    gui_optimizer/*                         execution + security + CLI
```

| Package | Current modules |
| --- | --- |
| Datasets | `schema.py`, `models.py`, `identity.py`, `formal_adapter.py`, `invariants.py`, `receipts.py` |
| SwissKnife | `models.ts`, `scanner.ts`, `identity.ts`, `component-graph.ts`, `ui-capsule.ts`, `state-machine.ts`, `scenario-catalog.ts`, `policy-validator.ts`, `invalidation.ts`, `context-pack.ts`, `accessibility.ts`, `visual-regression.ts`, `interaction-runner.ts`, `baseline.ts`, `evaluator.ts`, `cli.ts`, `targets/agent-supervisor.ts` |
| Accelerator | `authority.py`, `patch_scope.py`, `proposal.py`, `worktree_executor.py`, `check_plan.py`, `run_journal.py`, `artifact_store.py`, `improvement_loop.py`, `cli.py`, `benchmark.py` |

A task that changes more than one repository commits each nested repository
first and hands off one atomic superproject candidate containing every
changed gitlink.

## 3. Current modules and interfaces

Accelerator interfaces currently in tree:

| Module | Interfaces |
| --- | --- |
| `authority.py` | `GuiPatchAuthority@1`, `GuiHostBoundaryPolicy@1`, `GuiAcceptanceAuthority@1` |
| `patch_scope.py` | `GuiPatchScopeGate@1`, `GuiPatchScopeDecision@1` |
| `proposal.py` | `GuiPatchProposer@1`, `DeterministicGuiTransformation@1`, `HumanGuiReviewRequest@1` |
| `worktree_executor.py` | `GuiIsolatedWorktreeExecutor@1`, `GuiPatchApplicationReceipt@1` |
| `check_plan.py` | `GuiAffectedCheckPlanner@1`, `GuiCheckPlan@1`, `GuiCheckExecutionReceipt@1` |
| `run_journal.py` | `GuiRunJournal@1`, `GuiRunCheckpoint@1`, `GuiResumeDecision@1` |
| `artifact_store.py` | `GuiEvidenceArtifactStore@1`, `GuiEvidenceArtifactManifest@1` |
| `improvement_loop.py` | `VerifiedGuiOptimizer@1`, `GuiImprovementRun@1`, `GuiImprovementDecision@1` |
| `cli.py` | `GuiOptimizerCli@1`, `gui-opt scan@1` … `gui-opt report@1` |
| `benchmark.py` | `GuiOptimizationBenchmark@1`, `GuiBenchmarkTask@1`, `GuiBenchmarkResult@1` |

`GuiApplicationAdapter@1` is the documented extension seam. The current
implementation is `AgentSupervisorTarget@1` in
`swissknife/src/services/gui-optimizer/targets/agent-supervisor.ts`.

## 4. Security model

`GuiOptimizerSecurityAuthority` fail-closes on:

- paths outside `DEFAULT_ALLOWED_ROOTS` or containing forbidden segments;
- browser payloads that select host paths, process commands, or credentials;
- UI state presented as authorization;
- sensitive change kinds without contract verification or human review;
- stale policy decisions used to authorize the current action;
- undeclared files, deleted tests, arbitrary HTML execution, or unrelated
  applications.

Confirmation remains bound to the exact action and arguments. Disabled or
prohibited presentation must not retain a hidden dispatch path.

The artifact store is a narrow immutable content-addressed evidence CAS. It
is not a semantic-index and not a proof cache. Reuse is gated by exact
repository, component, scenario, extractor, and checker identities.

## 5. Improvement loop

Journaled phases (`PHASE_ORDER` in `improvement_loop.py`):

```text
baseline -> select_objective -> impact -> context_pack
  -> proposal -> isolated_worktree -> rescan -> invalidation
  -> affected_checks -> fallback -> compare -> decision -> receipt
```

Tied tests:

- `external/ipfs_accelerate/test/api/test_gui_optimizer_improvement_loop.py`
- `external/ipfs_accelerate/test/api/test_gui_optimizer_authority.py`
- `external/ipfs_accelerate/test/api/test_gui_optimizer_patch_scope.py`
- `external/ipfs_accelerate/test/api/test_gui_optimizer_acceptance_adversarial.py`
- `external/ipfs_accelerate/test/api/test_gui_optimizer_benchmark_catalog.py`
- `external/ipfs_accelerate/test/api/test_gui_optimizer_architecture_docs.py`

A rejected patch remains only in the disposable worktree. Interrupted runs
resume from content-addressed journal state and revalidate current Git
identities. Process exit is not completion.

## 6. Evidence authority matrix

`GuiEvidenceAuthorityMatrix@1`:

| Claim class | Status / level | What it means | Automatic accept? |
| --- | --- | --- | --- |
| Formally verified | `verified` + `proved_bounded_property` | Supported bounded obligation discharged with bound premises/tools | Yes, only for that obligation |
| Structurally validated | `structurally_valid` / `structural` | Finite-graph structural conclusion | No |
| Integrity valid | `integrity_valid` / `integrity` | Bytes match a content identity | Only as integrity of already-classified evidence |
| Simulated | `simulated` | Fixture or synthetic observation | No |
| Heuristic | `heuristic` | Inferred visual or metric description | No; cannot override hard gates |
| Human-reviewed | `human_reviewed` | Operator judgment | Required for subjective kinds |

Analysis classification (`exact`, `conservative`, `heuristic`, `opaque`) is
independent from verification status. Content identities and receipts do
not prove truth.

What is formally verified: the closed `SUPPORTED_PROPERTY_KINDS` in
`formal_adapter.py`: `defined_transition_targets`, `failure_recovery`,
`async_effect_completeness`, `event_outcome_coverage`,
`reachable_required_action`, `single_initial_state`,
`no_duplicate_state_ids`, `confirmation_bound_action`,
`form_accessible_names`, `modal_focus_lifecycle`, and
`policy_not_browser_authoritative`.

What is structurally validated: exact finite-graph results that are not
elevated to theorem-prover authority.

What is heuristic and human-reviewed: visual hierarchy, density,
consistency, clarity, whitespace, polish, primary-action prominence, and
unperformed screen-reader review. Automated accessibility success is not
WCAG certification.

## 7. Commands

```text
gui-opt scan agent-supervisor
gui-opt baseline agent-supervisor
gui-opt impact <path-or-component>
gui-opt evaluate agent-supervisor [--benchmark ID] [--expected-tasks N]
gui-opt pack-context agent-supervisor --objective <objective>
gui-opt verify <worktree-or-patch-or-alias> [--receipt PATH] [--full]
gui-opt improve agent-supervisor --objective <objective> [--isolated]
gui-opt report <run-id-or-alias> [--require-complete] --verify-receipts
```

Root launcher: `scripts/gui-opt` →
`python3 -m ipfs_accelerate_py.agent_supervisor.gui_optimizer.cli`.
JSON receipts use `GuiOptimizerCliReceipt@1`. Command interfaces are
`gui-opt scan@1`, `gui-opt baseline@1`, `gui-opt impact@1`,
`gui-opt evaluate@1`, `gui-opt pack-context@1`, `gui-opt verify@1`,
`gui-opt improve@1`, and `gui-opt report@1`. The only registered
application target is `agent-supervisor`.

## 8. Application-extension checklist

Exact additions required for **one** other application
(`GuiApplicationAdapter@1`):

### Manifest

- `swissknife/src/services/apps/virtual-desktop-app-manifest.ts` — new
  verified application definition
- `swissknife/web/js/main-simple.js` — matching runtime registration, with
  recorded divergence if the maps disagree

### Target

- `swissknife/src/services/gui-optimizer/targets/<app-id>.ts`
- `TARGET_REGISTRY` + `COMPONENT_REGISTRY` in
  `swissknife/src/services/gui-optimizer/cli.ts`
- `TARGET_REGISTRY` + `COMPONENT_REGISTRY` in
  `external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/gui_optimizer/cli.py`
- optional `VERIFY_ALIAS_REGISTRY` / `REPORT_ALIAS_REGISTRY` aliases

### Scenario

- `swissknife/src/services/gui-optimizer/scenario-catalog.ts`
- `swissknife/test/fixtures/gui-optimizer/scenarios/<app-id>-scenarios.json`
- `swissknife/test/fixtures/gui-optimizer/<app-id>/fixture-host.html`
- `swissknife/test/fixtures/gui-optimizer/<app-id>/fixture-services.js`
- `swissknife/test/fixtures/gui-optimizer/<app-id>/fixture-scenarios.json`

### Action and policy

- `swissknife/src/services/apps/all-app-executable-backend-contract.ts`
- `swissknife/src/services/apps/all-app-live-tool-bindings.ts`
- `swissknife/src/services/apps/app-capability-policy.ts`
- `swissknife/src/services/apps/mcp-deontic-ui-manifest.ts`
- `UiActionBinding@1` / `UiConfirmationBinding@1` for every displayed action
- host re-evaluation remains
  `swissknife/src/services/mcp/mcp-control-surface-mediator.ts` and
  `swissknife/src/services/mcp/all-app-tool-gateway.ts`

### Tests, screenshots, acceptance

- `swissknife/test/unit/services/gui-optimizer/<app-id>-fixtures.test.ts`
- `swissknife/test/unit/services/gui-optimizer/<app-id>-baseline.test.ts`
- `swissknife/test/browser/verified-gui-optimizer-<app-id>-boundary.test.ts`
- `swissknife/test/e2e/verified-gui-optimizer-<app-id>-baseline.spec.ts`
- `swissknife/test/e2e/verified-gui-optimizer-<app-id>-regression.spec.ts`
- screenshot artifacts via `GuiEvidenceArtifactStore@1` and manifests under
  `implementation_plan/evidence/verified_gui_optimizer/`
- proposal acceptance criteria plus
  `VisualRegressionReceipt@1`, `AccessibilityReceipt@1`,
  `InteractionReceipt@1`, `UiConstraintReceipt@1`,
  `GuiImprovementReceipt@1`

This program does not implement those additions for a second application.

## 9. Exclusions and non-goals

Standalone dependency boundary — must not import, call, require, or derive
authority from:

- a semantic-index module created by an earlier task
- a semantic-capsule module created by an earlier task
- a proof-cache or formal-verification-cache module
- a model-routing or provider-routing module
- the untracked datasets UI/UX IR implementation
  (`ipfs_datasets_py/logic/ui_ux_ir`)

`UiSemanticCapsule@1` is the new closed GUI record, not the excluded prior
capsule package. `GuiPatchProposer@1` exposes caller-selected routes and
does not choose or route models. No content-addressed artifact cache is a
proof cache.

Also excluded as runtime authority:

- `swissknife/web/legacy-archive`, `emergency-archive`, `cleanup-archive`
- `virtual-desktop-live-gateway.ts` as authorization
- production credentials, production services, production MCP tools, user
  data, remote scripts, arbitrary filesystem paths, and arbitrary
  subprocess commands in fixtures

Non-goals: beauty proofs; complete accessibility; complete security;
unbounded correctness; proving the GUI optimal; optimizing every
application; rewriting failed results or deleting runtime history.

## 10. Narrow final claim

The selected GUI workflow was incrementally analyzed and improved against declared interaction, accessibility, policy, and visual-regression criteria, with content-addressed evidence for the evaluated scenarios.

Documentation states what is formally verified, structurally validated,
heuristic, and human-reviewed. It does not claim that the GUI is proved optimal.

## Diagrams tied to tests

```text
gui-opt improve agent-supervisor --objective …
        |  test_gui_optimizer_authority.py
        v
  GuiPatchAuthority@1 + GuiHostBoundaryPolicy@1
        |  test_gui_optimizer_patch_scope.py
        v
  GuiPatchScopeGate@1
        |  test_gui_optimizer_improvement_loop.py
        v
  isolated worktree -> affected checks -> receipt
        |  test_gui_optimizer_acceptance_adversarial.py
        |  test_gui_optimizer_benchmark_catalog.py
        v
  accept | reject | human_review
        |  test_gui_optimizer_architecture_docs.py
        v
  architecture + contracts remain consistent with current modules
```
