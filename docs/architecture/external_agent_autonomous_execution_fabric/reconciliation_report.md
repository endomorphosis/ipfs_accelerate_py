# External Agent Fabric source-reconciliation report

Status: reviewed planning-time input  
Snapshot time: 2026-08-18  
Machine-readable authority: [`source_reconciliation_manifest.json`](source_reconciliation_manifest.json)

This report records the source selection performed before implementation of the
ExternalAgentAutonomousExecutionFabric. It is a human-readable projection of the
machine manifest, not a mutable task output and not a claim that every historical
branch is qualified. Branch names, old test messages, generated artifacts and
worker assertions are evidence only. Full object IDs, changed-path set hashes,
schema and API classifications, dependencies, tests, supersession decisions and
preservation counters are in the machine manifest.

## Baselines and selected integration roots

| Repository | Verified `origin/main` at planning time | Reviewed integration input | Decision |
|---|---|---|---|
| `ipfs_accelerate_py` | `12c4e8387de4986d38d69534f3d74864e7bb15c1` (tree `96504e7744d741784bc8076456f4b169dd665ab5`) | `0085dc719686bf4cd077c8099170bdd55fa2cf99` (tree `4298f4b06fa753a60ff8f95ffead39be9a83092c`) | Start from `origin/main` and retain only the narrow, provenance-recorded forward repairs for checkout lease-state classification and `DatabaseProgramConfig` propagation. |
| `ipfs_datasets_py` | `480a1666f144ad606fcb3cacb66e59775f28d0d1` (tree `6fdc81fabb04a86683d0f26200636fa8f61fd25c`) | `41533721c5559ad68cecfe226fa6ba5f76f8a15d` (tree `d88b10b706d91c37e4be346366ff06bb58d1e8a3`) | Preserve ancestry with a real two-parent UI/UX-IR merge; retain current LPC API and registry behavior in the two reviewed conflicts. |
| `ipfs_kit_py` | `2564aea1ae35061f2165872aff91e8a40801ab7e` (tree `98ab8d00f79ec542032dbbb21a1ea416b983a845`) | same commit and tree | Existing semantic-root, proof-sealer, artifact-store and MCP++ adapter lineages are already contained. |
| `Mcp-Plus-Plus` | `5ac0ab162f420264fd224073a5df3f2d7c054ae3` (tree `19d41c4838e9ed3616bb4883d71b7f788f66a1c9`) | `5bf87beba3acf18d705c5c8ee3174e5e16ab5e04` (tree `9459e5a6695771e284142577da00aac07370fde8`) | Use only the existing shared profiles and wire contracts. The additional commit clarifies backend roles without changing schema, conformance, API or CID identities. |

The planning-time four-repository source-forest root is
`sha256:ed543c10f6aa90e093c8ae8b8866934e0cc1614e1be49ddcdc5dd7a2ce8565fa`.
Implementation commits must supersede this input through a reviewed compatibility
manifest and must never rewrite it in place.

The current EAAEF bootstrap implementation is a reviewed, task-owned working-tree
overlay on the exact accelerator input commit and tree above; it is not represented
as a nonexistent clean commit. `EAAEF-000` owns every restored or newly implemented
path, while the compatibility manifest freezes the schema/interface ceiling. A later
clean-source materialization receipt must record the actual implementation commit,
tree and file identities before any execution admission. Until that receipt exists,
the selected integration root remains the exact planning input and the overlay grants
no supervisor, merge or production authority.

## Inventory and reconciliation decisions

### `ipfs_accelerate_py`

The inventory covered 4,773 local branches, 155 remote refs, 992 commits reachable
from an observed ref but not `origin/main`, and 266 registered worktrees. Twenty-five
worktrees were dirty and were preserved. The original partial-merge regressions were
repaired forward rather than by reverting later work:

- checkout mutation lease-state classification was introduced at
  `a8a4a71d85cfa8ce2280f4d5895e2b6ba0acd43a`, retained through
  `ea9b2af0c1e772ba445ab1589b123050484c217a`, accidentally deleted at
  `d73b66e80497b78c6b209af9b4cf9bdb33ced8aa`; an unsealed in-session run
  reported 24 focused tests plus a supervisor/CLI import check;
- `DatabaseProgramConfig` propagation from the `b6dc155c3`/`299b8633c` lineage
  was forward-restored at `0085dc719686bf4cd077c8099170bdd55fa2cf99`; an
  unsealed in-session run reported 16 authority-forwarding tests plus compilation
  and import checks. These observations establish repair diagnostics, not current
  promotion evidence.

The final pre-admission source-freeze audit found another exact merge deletion.
Commit `38cd50092d300b61327a9225e7f10cfe8acefb4f` (tree
`005f885e270bbc7573710686a78fc2f740ee5b9f`) captured a still-referenced semantic
contract-assurance runtime closure. The same 25 blobs remained byte-identical at
`ea9b2af0c1e772ba445ab1589b123050484c217a^1`, exact commit
`56ef4fa6479eee78cacfe7372df67a5eda329060` (tree
`baa136198bb182f2b9be7c932f87fab46f94d5fa`), but the merge result
`ea9b2af0c1e772ba445ab1589b123050484c217a` deleted them. The forward repair
restores, without squashing or rewriting history:

- twelve production modules: `contract_mismatch_analyzer.py`,
  `contract_vulnerability_rules.py`, `mcp_contract_catalog.py`,
  `mcp_invocation_trace.py`, `parser_failure_triage.py`, `polyglot_ast_health.py`,
  `polyglot_ast_provider.py`,
  `python_mcp_surface_extractor.py`, `runtime_component_catalog.py`,
  `runtime_contract_evidence_compiler.py`, `swissknife_contract_extractor.py`, and
  `proof/mcp_contract_proof_cache.py`;
- their twelve exact paired `test/api/test_agent_supervisor_<stem>.py` tests; and
- the exact `scripts/extract_typescript_ast.mjs` runtime helper required by the
  polyglot AST provider.

The machine manifest records every restored path and Git blob identity. These
historical identities establish provenance only; current focused qualification is
still an EAAEF-000 requirement and is not inferred from an old test report.

The same audit separated line-level domain-move repairs from byte-exact restoration.
The `ab984496fb1109700ec37ed91bda5b29d7214ecb` canonical-logic and
repository-forest import fixes were present at the first parent of `ea9b2af0` and
regressed in its merge result. They were reapplied narrowly to the current
`integrations/ipfs_datasets_logic_provider.py` and
`validation/proof_cached_test_validation.py` rather than overwriting their later
content. Direct imports were also corrected in `objectives/objective_graph.py`,
`entrypoints/inference_runtime.py`, and the two proof-reuse identity modules so the
sealed bootstrap does not depend on mutable package-level compatibility aliases.
The lazy merge-proof gate was likewise redirected from the old merge-local policy
path to `proof.formal_verification_policy`, with a direct receipt-type regression
test; its original import and domain-move revisions are recorded in the manifest.
The machine manifest records each old and canonical import target; their current
tests remain part of the manual EAAEF-000 gate.

The final lifecycle audit forward-reconciled the controlled-restart contract from
`9e39c6c9edb0b756f99f9857a89e70642ef1321c` (tree
`ea321ea749103ece6a175c4e984372e42ac204bd`) into the current
`merge/worktree_lifecycle.py` and its direct test. Cleanup now receives the caller's
exact lane state directory; only a provably dead same-lane owner may be reclaimed
before lease expiry, while peer lanes remain expiry-gated and a successful restart
advances the fence. An in-session run reported 48 passing lifecycle tests. It is
unsealed diagnostic evidence and remains subject to the EAAEF-000 gate.

Shared-dependency linking was then reconciled from the intent of
`a4413463c6e9d356b0143750db98486c0689bb0a` (tree
`f205cbf01c9677549e25ae733a7ad2fdf8610b3d`) with a necessary containment
correction: a sibling path is eligible only when Git registers it as a worktree of
the exact repository. Arbitrary siblings remain outside the daemon's mutation
boundary. The paired positive registered-sibling and negative outside-sibling
diagnostics were part of an unsealed eight-test in-session run; they do not promote
the daemon or authorize a live EAAEF launch.

The final configured-board load audit then exposed a duplicated-contract drift:
the tracked scheduler configuration and materializer already required
`eaaef-signed-command-fabric-profile@2` with the exact EAAEF board namespace and
`control-shard-0`, while the downstream capsule/admission field sets and one runner
fixture still described profile `@1`. The consumers were reconciled forward to the
existing `@2` contract. Capsule validation now joins board and shard to independent
canonical expectations, bootstrap admission joins the board identity to the
materializer's validation projection, and both database programs require the exact
EAAEF operational schema revision `@2`. The `@2` revision also triggers the EAAEF
live-seal child gate without changing the legacy datasets-authoritative `@1`
marker. A 36-case focused in-session diagnostic passed; it is unsealed evidence and
does not admit a supervisor launch.

The exact isolated materializer `launch-plan` then exposed a separate dependency
projection drift. The outer launcher had verified one absolute
`approved_import_root`, but its fresh `-I -S` board-validator child did not receive
that closed root and stopped while importing the validator's cryptography
dependency. The child bootstrap now accepts only the already verified runtime
binding, requires the approved root to exist and equal its strict canonical
resolution, and prepends only the reviewed repository root followed by that
approved dependency root. It retains `-I`, `-S` and `-B` and never inherits
`PYTHONPATH` or user-site discovery. Independent source review reported no P0/P1
finding; 20 focused cases passed, and the actual isolated launch plan advanced to
the expected dirty-source guard. These are unsealed diagnostics, not launch or
promotion evidence, and no supervisor process was started.

Router reconciliation used three additional exact source revisions. The missing
ASE3-028 router-owned provider-decision block was restored byte-for-byte from
`757471e4a7dcb5b9f01800e8dd01ea2b89de3c1f` (block SHA-256
`aa7f9459e168add0a67b26dd2f309a8591d4a92b9fd32f1e76587222fc0cf17f`) because
current capability consumers import that contract and otherwise silently degrade
the leaf evidence to unavailable. Exact-provider fallback/cache authority was
ported narrowly from the uncontained
`5a6a0859500e44094b6b11b467555855149c88ba`; unrelated historical default-model
changes were not replayed. The VGO authorization/root/witness scope was reconciled
from ancestor `dc6dbae8252206150210733ba31a5d1271617775` into the current split route
module while retaining separate legacy-V3 Grok-4.5 and EAAEF Grok-4.6 namespaces.
Focused in-session diagnostics exercise these repairs, but they are not immutable
promotion receipts and do not change the EAAEF-000 no-go disposition.

The DCR linked-Git authority branch `d7aac67ee8ca9d06994333323ecd81fbae87e1c7`,
self-hosting branch `ef2ed37d87d07e7f7b845d67cc948bdbdaa09e14`, semantic-runtime branch
`e10694ce3fbb4129c69553168239c89961f9efb7`, and governed-contract branch
`e7fe9afc6eb9963b8fab715418c1bb0abae2a936` were not wholesale merged. Their
residual contracts and tests require current-API semantic comparison; the manifest
marks every one unsafe to cherry-pick.

Committed PID, WAL, log and TLS key/certificate-shaped artifacts were recorded as
source-hygiene risks. In particular `state/tls/mcpplusplus.key` is treated as
potentially compromised and must not be exposed. Classification, fixture relocation
or rotation is a forward repair; history is not rewritten without explicit approval.
The implementation integration tree therefore forward-removes all eight listed
runtime artifacts and relies on the existing ignore policy and ephemeral certificate
generator. Their historical blobs remain preserved; any credential ever derived from
the key-shaped blob still requires rotation rather than trust in deletion alone.

### `ipfs_datasets_py`

The inventory covered 4,275 local refs, 901 remote refs, 2,388 commits reachable
outside `origin/main`, and 75 worktrees. The important UI/UX-IR lineage at
`9d558ad706e83a944bbf3b66508f969041cc9518` had 111 changed paths: 106 were
content-identical on main, two required current-LPC conflict resolution, and three
documentation paths were absent. It was integrated by the provenance-preserving
merge `41533721c5559ad68cecfe226fa6ba5f76f8a15d` rather than copied or squashed.

That merge's focused run recorded 359 passes and seven failures whose code and test
paths were byte-identical to its first parent. Those failures remain current release
debt; they are not represented as a green gate. The stale proof-reuse restoration at
`382e0d7e0ba4a774e15573cea27607e8ae617ab5`, isolated semantic-contract candidate
at `d087dd9c781f9e953c148ee9fd29395c5b4f0e1c`, and conflict-heavy snapshot at
`0691203550c0f316852c74d293d8fc3c4ce130a6` were preserved but not blindly
cherry-picked. DuckDB/Quack/DuckLake, incremental semantic indexes, semantic-state
contracts, logic-platform canonicalization and patent/legal intelligence were already
present and must be reused rather than reimplemented.

### `ipfs_kit_py`

The inventory covered 3,853 local branches, 2,882 remote refs, 2,957 commits
reachable outside main, and 228 registered worktrees. The stale proof restoration
`cc66cda6982860796d505f7602ad5410bf6d213a` is incompatible with the current proof
API, and the Copilot subprocess-wrapper branch
`7d6791eb6cd0e7a14aac2127fcb719d1677f759e` is the wrong abstraction for an
external-session handoff. Neither was admitted.

One dirty partial CAS worktree at
`implementation/ksr-002-ce41dfca5e3d-attempt-1-1786430953` was explicitly
preserved. Its uncommitted `coordination_storage.py` delta is superseded and
unqualified; it was not overwritten or merged. The reviewed baseline already contains
the semantic-root KSR, incremental proof sealer, AAE artifact store, GraphRAG roots
and MCP++ G/E/confidential adapters.

### `Mcp-Plus-Plus`

All observed branches were merged at its planning baseline. The fabric reuses Profiles
A/B/C/D/F/G and the existing execution, encrypted-artifact, obligation, durable-
executor and state-reference contracts; it does not invent Profile H or another A2A
profile. An unsealed in-session run for the documentation clarification reported 123
focused tests and no schema or conformance-tree change; it is diagnostic evidence,
not current qualification.

Existing MCP++ prose or demonstrations do not qualify backend authority. Profile G's
in-process coordinator remains a validator/demo. For this fabric, DuckDB holds the
transactional state, one authenticated and fenced Quack owner is the only mutable
multi-client boundary, and DuckLake/Parquet/IPLD/CAR/IPFS are immutable history and
analytics. Replication never grants a lease, fence, claim or merge authority.

### Final working-overlay source reconciliation

The frozen working overlay now contains the source-verified R1 runtime factory and
the distinct source-complete R2 process-remote owner seam. R1 binds an independently
signed native-dependency admission, the V2 lane/verifier/merge chain, exact-envelope
journaling, qualified-input-only Quack and dispatcher factories, and exact per-birth
supervisor/daemon wiring. R2 binds only the three-operation prepare/apply/observe
remote capability, canonical wire, owner service, gateway and durable client journal.
Direct proxy construction, R1/R2 authority crossover, raw paths, tokens, callbacks
and database handles remain prohibited.

Frozen source evidence reported 13/13 canonical parallel cases, 109/109 scheduler,
567/567 todo-daemon, 121/121 boundary, 56/56 validation, 32/32 incremental runtime
and 26/26 R1/R2/birth-wiring cases. The native admission/pin slice passed 50 and
skipped one environment-only aarch64 reviewed-fixture case. Ordinary Plan-R2/state
execution passed 10 and explicitly skipped three cases requiring DuckDB 1.5.5 plus
`QUACK_155_EXTENSION_PATH`; the same selection passed 13/13 under a temporary,
unsealed exact DuckDB/Quack diagnostic environment. Volta's independent final source
review found no remaining P0/P1 issue and `git diff --check` was clean.

This closes source ownership and focused regression reconciliation, not production
qualification. No actual independently signed native/lane/Quack/dispatcher/Plan-R2
artifact, deployed signed command-authorizer/Quack/dispatcher endpoint, qualified
extension, or real admitted Docker/container/image/provider/network authority is
present. No live EAAEF supervisor, provider, container, network, Quack mutation,
Plan-R2 transition or host-merge effect was launched.

## Preservation and release disposition

The reconciliation operation deleted no refs or worktrees, overwrote no dirty
overlay, performed no force-push or history rewrite, and pushed nothing remotely.
The selected roots are reviewed implementation inputs, not a release claim.

At reconciliation time, embedded one-writer materialization remained conditional on
a new immutable board seal; continuous Quack coordination and live DuckLake remained
typed no-go pending exact runtime, gateway, failover, security and recovery evidence.
Any task that discovers a better residual implementation must submit a proposal and
qualification receipt against these exact identities. It may not silently replace
this report, the machine manifest, a preserved ref, or an accepted task revision.
