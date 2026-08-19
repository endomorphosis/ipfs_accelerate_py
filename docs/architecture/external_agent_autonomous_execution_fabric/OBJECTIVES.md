# ExternalAgentAutonomousExecutionFabric goals

## EAAEF-G000 Implement and qualify ExternalAgentAutonomousExecutionFabric

- Status: active
- Parent:
- Depends on:
- Completion contract: All mandatory epic goals reach accepted terminal evidence against one source/semantic/plan generation; no blocking invalidation, mutable claim or merge remains; the terminal report and seal verify.
- Desired postconditions: ["All required A-R postconditions are independently accepted against one current source, semantic and plan generation, and the terminal seal verifies."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"ROOT","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks:

## EAAEF-G010 A — Unmerged-work reconciliation and release baseline

- Status: active
- Parent: EAAEF-G000
- Depends on:
- Completion contract: All relevant refs and dirty overlays are classified, reviewed integration roots are immutable, and StackCompatibilityManifest@1 binds the exact cross-package stack.
- Desired postconditions: ["All relevant refs and dirty overlays are classified, reviewed integration roots are immutable, and StackCompatibilityManifest@1 binds the exact cross-package stack."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"A","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-000, EAAEF-001, EAAEF-002, EAAEF-003, EAAEF-004, EAAEF-005, EAAEF-006, EAAEF-007, EAAEF-008, EAAEF-009

## EAAEF-G020 B — External agent-session handoff protocol

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G010
- Completion contract: Raw client exports and a bounded normalized event stream are durably preserved with separate identities, provenance, trust labels, privacy and retention.
- Desired postconditions: ["Raw client exports and a bounded normalized event stream are durably preserved with separate identities, provenance, trust labels, privacy and retention."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"B","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-010, EAAEF-011, EAAEF-012, EAAEF-013, EAAEF-014, EAAEF-015

## EAAEF-G030 C — Complete Git repository transfer

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G020
- Completion contract: Every admitted repository state is reconstructed in quarantine and verified across Git objects, dirty overlays, submodules, LFS, modes, symlinks and transfer bounds.
- Desired postconditions: ["Every admitted repository state is reconstructed in quarantine and verified across Git objects, dirty overlays, submodules, LFS, modes, symlinks and transfer bounds."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"C","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-020, EAAEF-021, EAAEF-022, EAAEF-023, EAAEF-024

## EAAEF-G040 D — Caller identity, capability and disclosure policy

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G030
- Completion contract: Effect-bound authenticated authority is distinct from prompts, CIDs, transport identity and imported history; disclosure and approvals bind exact inputs.
- Desired postconditions: ["Effect-bound authenticated authority is distinct from prompts, CIDs, transport identity and imported history; disclosure and approvals bind exact inputs."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"D","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-030, EAAEF-031, EAAEF-032, EAAEF-033, EAAEF-034

## EAAEF-G050 E — Project onboarding and codebase classification

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G040
- Completion contract: Every ordinary Git repository receives a typed assessment; autonomous mutation is admitted only through a qualified ProjectAdapter and known validation profile.
- Desired postconditions: ["Every ordinary Git repository receives a typed assessment; autonomous mutation is admitted only through a qualified ProjectAdapter and known validation profile."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"E","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-040, EAAEF-041, EAAEF-042, EAAEF-043, EAAEF-044

## EAAEF-G060 F — OCI container execution fabric

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G050
- Completion contract: Workers execute only leased tasks in isolated containers with bounded resources, no Docker socket, default-deny network and restart-safe checkpoints; the engine is rootless where supported, otherwise an independently approved rootful-host-daemon/nonroot-worker fallback is required.
- Desired postconditions: ["Workers execute only leased tasks in isolated containers with bounded resources, no Docker socket, default-deny network and restart-safe checkpoints; the engine is rootless where supported, otherwise an independently approved rootful-host-daemon/nonroot-worker fallback is required."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"F","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-050, EAAEF-051, EAAEF-052, EAAEF-053, EAAEF-054, EAAEF-055

## EAAEF-G070 G — Handoff context and federated retrieval

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G060
- Completion contract: Repository truth, imported claims, receipts, legal corpora and hypotheses remain distinct while AST, capsules, BM25, vector, GraphRAG and knowledge graphs compose through one provenance-preserving retrieval plan.
- Desired postconditions: ["Repository truth, imported claims, receipts, legal corpora and hypotheses remain distinct while AST, capsules, BM25, vector, GraphRAG and knowledge graphs compose through one provenance-preserving retrieval plan."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"G","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-060, EAAEF-061, EAAEF-062, EAAEF-063, EAAEF-064

## EAAEF-G080 H — Logic-governed goal and task compilation

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G070
- Completion contract: The existing logic platform admits only covered, acyclic, bounded, feasible goal/task plans with explicit conflicts, proof obligations and completion contracts.
- Desired postconditions: ["The existing logic platform admits only covered, acyclic, bounded, feasible goal/task plans with explicit conflicts, proof obligations and completion contracts."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"H","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-070, EAAEF-071, EAAEF-072, EAAEF-073, EAAEF-074

## EAAEF-G090 I — Conflict-free multi-agent parallel execution

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G080
- Completion contract: The existing semantic work fabric selects fenced conflict-free frontiers; multiple attempts are allowed but one logical result alone may be accepted.
- Desired postconditions: ["The existing semantic work fabric selects fenced conflict-free frontiers; multiple attempts are allowed but one logical result alone may be accepted."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"I","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-080, EAAEF-081, EAAEF-082, EAAEF-083, EAAEF-084, EAAEF-085

## EAAEF-G100 J — Production DuckDB, Quack and DuckLake plane

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G090
- Completion contract: DuckDB plus one fenced authenticated Quack owner form the sole mutable coordination plane; DuckLake and immutable artifacts provide non-authoritative history, analytics, lineage and recovery.
- Desired postconditions: ["DuckDB plus one fenced authenticated Quack owner form the sole mutable coordination plane; DuckLake and immutable artifacts provide non-authoritative history, analytics, lineage and recovery."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"J","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-090, EAAEF-091, EAAEF-092, EAAEF-093, EAAEF-094, EAAEF-095, EAAEF-096, EAAEF-097

## EAAEF-G110 K — Closed-loop execution and adaptive replanning

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G100
- Completion contract: Every accepted result refreshes source and semantic state, invalidates stale evidence, revises the immutable plan and converges to a bounded fixed point.
- Desired postconditions: ["Every accepted result refreshes source and semantic state, invalidates stale evidence, revises the immutable plan and converges to a bounded fixed point."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"K","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-100, EAAEF-101, EAAEF-102, EAAEF-103, EAAEF-104

## EAAEF-G120 L — Python, CLI, MCP and MCP++ surfaces

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G110
- Completion contract: All transports expose one semantic operation set and canonical identity, support detach/reconnect/cursors, and use only existing MCP++ profiles where a shared contract is required.
- Desired postconditions: ["All transports expose one semantic operation set and canonical identity, support detach/reconnect/cursors, and use only existing MCP++ profiles where a shared contract is required."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"L","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-110, EAAEF-111, EAAEF-112, EAAEF-113, EAAEF-114, EAAEF-115

## EAAEF-G130 M — Security hardening

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G120
- Completion contract: Hostile repositories and histories cannot widen policy, escape containers, forge verification, expose secrets or acquire mutation authority.
- Desired postconditions: ["Hostile repositories and histories cannot widen policy, escape containers, forge verification, expose secrets or acquire mutation authority."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"M","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-120, EAAEF-121, EAAEF-122, EAAEF-123, EAAEF-124, EAAEF-125

## EAAEF-G140 N — Observability and accounting

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G130
- Completion contract: Typed, privacy-safe events and resource/cost metrics make every run observable, explainable, steerable and auditable without publishing sensitive bodies.
- Desired postconditions: ["Typed, privacy-safe events and resource/cost metrics make every run observable, explainable, steerable and auditable without publishing sensitive bodies."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"N","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-130, EAAEF-131, EAAEF-132, EAAEF-133

## EAAEF-G150 O — End-to-end and fault qualification

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G140
- Completion contract: Real client, supervisor, worker, Quack, DuckDB, DuckLake, network and crash fixtures demonstrate safe recovery and evidence-backed terminal outcomes.
- Desired postconditions: ["Real client, supervisor, worker, Quack, DuckDB, DuckLake, network and crash fixtures demonstrate safe recovery and evidence-backed terminal outcomes."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"O","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-140, EAAEF-141, EAAEF-142, EAAEF-143, EAAEF-144, EAAEF-145

## EAAEF-G160 P — Performance and parallelism benchmark

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G150
- Completion contract: Configurations A through D are measured honestly; missed targets remain reported and historical or simulated results never count as current qualification.
- Desired postconditions: ["Configurations A through D are measured honestly; missed targets remain reported and historical or simulated results never count as current qualification."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"P","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-150, EAAEF-151, EAAEF-152, EAAEF-153

## EAAEF-G170 Q — Packaging and external deployment

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G160
- Completion contract: Clean wheels and digest-pinned OCI images install without sibling checkouts or editable installs and ship locks, SBOMs, migrations, backup, restore and rollback.
- Desired postconditions: ["Clean wheels and digest-pinned OCI images install without sibling checkouts or editable installs and ship locks, SBOMs, migrations, backup, restore and rollback."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"Q","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-160, EAAEF-161, EAAEF-162, EAAEF-163, EAAEF-164

## EAAEF-G180 R — Blocking CI and qualification release

- Status: active
- Parent: EAAEF-G000
- Depends on: EAAEF-G170
- Completion contract: Every required lane is blocking and current; the release emits a narrow evidence-backed qualification level and explicit go/no-go recommendation.
- Desired postconditions: ["Every required lane is blocking and current; the release emits a narrow evidence-backed qualification level and explicit go/no-go recommendation."]
- Prohibited outcomes: ["universal autonomous-mutation support claim","duplicate accepted work or overlapping accepted effects","stale-fence acceptance","authority, disclosure, resource or mutation scope wider than the admitted parent policy","worker or model self-acceptance","stale source, semantic, plan, lease, fence, test or proof evidence","unverified imported history satisfying completion","DuckLake or a replica granting current coordination authority"]
- Scope: {"epic":"R","mutation":"only task-owned files/effects in admitted isolated worktrees and containers","repositories":["ipfs_accelerate_py","ipfs_datasets_py","ipfs_kit_py","Mcp-Plus-Plus only for an existing shared protocol contract"]}
- Resource budget: {"network":"deny unless an exact effect-bound approval names the action and inputs","policy":"bounded by the sum of admitted child-task reservations and the parent run ceiling","unbounded_refill":false}
- Authority ceiling: ["no protected-branch push or automatic production deployment","merge or reviewed-patch delivery only through independent admission","no worker, model, prompt, repository file, CID or run ID may widen authority"]
- Verification requirements: ["current pre-change, focused and affected-integration receipts","zero required skip, xfail, xpass or failure","independent verifier acceptance against exact source and plan roots"]
- Proof requirements: ["content identities and provenance for inputs, outputs and receipts","dependency coverage plus read/write/effect conflict admission","current proof obligations or a typed independently reviewed not-applicable decision"]
- Human review requirements: ["authenticated review for merge, push, new dependency/network/secret access, wider disclosure, destructive cleanup or publication","explicit review for unresolved critical security or compatibility findings"]
- Completion evidence: ["all required child task outcomes and independent acceptance receipts","current source/semantic/plan roots with no blocking invalidation","settled merge queue and no live mutating claims","content-addressed terminal report or typed no-go decision"]
- Gap tasks: EAAEF-170, EAAEF-171, EAAEF-172, EAAEF-173, EAAEF-174, EAAEF-175, EAAEF-176
