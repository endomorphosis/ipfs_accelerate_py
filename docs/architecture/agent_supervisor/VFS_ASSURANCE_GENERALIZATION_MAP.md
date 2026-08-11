# VFS assurance generalization map

**Status:** active for LPR-021 through LPR-028  
**Source lock:** [`config/agent_supervisor_vfs_generalization_sources.lock.json`](../../../config/agent_supervisor_vfs_generalization_sources.lock.json)  
**Pinned revision:** `0cc04ebb640c4c981cf4650016e096a73ab0e8c0`  
**Local ref:** `refs/agent-supervisor/source-locks/vfs-generalization/0cc04ebb640c4c981cf4650016e096a73ab0e8c0`

This map is the content-addressed old-to-new API, schema, caller, and package
ownership guide for extracting reusable assurance engines from the seven root
VFS assurance modules. Workers may **read exact Git blobs** from the lock; they
must **not** merge, cherry-pick, or copy root `vfs_*.py` implementations into the
target tree.

## Presence states (non-conflating)

| State | Applies to | Meaning |
| --- | --- | --- |
| `target_present` | any path coordinate | Regular file exists in the current target tree |
| `source_only` | **source path / source test only** | Locked blob is available; source path is absent from the target tree |
| `planned_only` | **planned path / planned test only** | Destination is planned here but not yet present in the target tree |

Rules:

1. `source_path_state` and `planned_path_state` are **independent**.
2. A module may be `source_only` on the source path and `planned_only` (or later
   `target_present`) on the planned path at the same time.
3. Do not treat “source revision has file X” as “target tree has planned path Y”.
4. Lock evaluation records states at write time; subsequent generalization tasks
   flip planned coordinates to `target_present` without rewriting source blob
   identities.

## Module map

| Source module (blob) | Planned destination | Package | Source path state | Planned path state | Source test blob |
| --- | --- | --- | --- | --- | --- |
| `vfs_surface_inventory.py` (`76f34e1b…`) | `analysis/repository_surface_inventory.py` | analysis | source_only | target_present (LPR-021; test also target_present) | `4c49d3d7…` |
| `vfs_contract_pack.py` (`9acc4ceb…`) | `analysis/program_contract_profile.py` | analysis | source_only | planned_only | `f0115f4a…` |
| `vfs_differential_harness.py` (`8a6c8af6…`) | `validation/differential_contract_harness.py` | validation | source_only | planned_only | `c428d285…` |
| `vfs_mcp_contract_checker.py` (`26144a7b…`) | `analysis/interface_contract_parity.py` | analysis | source_only | planned_only | `50310fad…` |
| `vfs_symbolic_benchmark.py` (`90023a09…`) | `validation/symbolic_efficiency_benchmark.py` | validation | source_only | planned_only | `28e65a8e…` |
| `vfs_symbolic_pilot.py` (`483ecaf6…`) | `runtime/symbolic_assurance_pilot.py` | runtime | source_only | planned_only | `1cb18acb…` |
| `vfs_symbolic_rollout.py` (`6a1ef7b8…`) | `control/symbolic_assurance_rollout.py` | control | source_only | planned_only | `854f5198…` |

Full blob identities, public export lists, and schema constants live only in the
lock file (no repository bodies embedded).

## Public export → generic surface (inventory)

Source: `vfs_surface_inventory` → destination:
`analysis.repository_surface_inventory`.

| Source export / concept | Generic destination | Notes |
| --- | --- | --- |
| `VfsSurfaceInventoryError` | `SurfaceInventoryError` | Same reason-code pattern |
| `SurfaceClassification` | `SurfaceClassification` | Unchanged closed taxonomy |
| `SurfaceKind` enum | string kinds via `SurfaceKindSpec` | Kinds are policy data, not module enums |
| `Definition` | `Definition` | Signature-comparable static definition |
| `SurfaceEvidence` / `EvidenceKind` | same names | Observed syntax only |
| `SurfaceContradiction` | `SurfaceContradiction` | Inconclusive by default; not a defect |
| `InventoryDiagnostic` | `InventoryDiagnostic` | Completeness / unknown reporting |
| `VfsSurface` | `SurfaceRecord` | Path-keyed surface observation |
| `InventoryCompleteness` | `InventoryCompleteness` | Bounded completeness ledger |
| `VfsSurfaceInventory` | `RepositorySurfaceInventory` | Carries immutable policy identity |
| `VARIANT_SUFFIXES` | `SurfaceInventoryPolicy.variant_suffixes` | Policy-supplied |
| `discover_vfs_surface_paths` | `discover_surface_paths(root, policy)` | Policy required |
| `inventory_vfs_surfaces` | `inventory_repository_surfaces(root, policy)` | Policy required |
| `assert_inventory_complete` | `assert_inventory_complete` | Fail-closed completeness |
| `publish_vfs_surface_inventory` | `publish_surface_inventory` | Atomic JSON publish |
| `VFS_SURFACE_INVENTORY_SCHEMA` | `REPOSITORY_SURFACE_INVENTORY_SCHEMA` or policy.schema | Profile may override |
| `VFS_SURFACE_INVENTORY_GOAL_ID` | **removed from generic code** | Board/goal ids stay in job profile only |

### Inventory behavioural contract (locked source)

The generic engine preserves the locked source contract when driven by a
VFS-equivalent profile:

- Bounded byte and static-AST scans only; **no import of scanned code**.
- Historical suffixes are discovery signals; `variant_presence_is_defect` is
  always false.
- Completeness and unexplained surfaces are reported; incomplete inventories
  fail `assert_inventory_complete`.
- Output is deterministic under reordered filesystem inputs (sorted
  enumeration, sorted records, content-addressed `content_id`).
- Authority flags remain non-authoritative: not completion evidence, not
  correctness evidence, does not authorize repair.

## Remaining module maps (seeds for LPR-022+)

| Source | Planned generic | Primary schemas (source) | Extraction focus |
| --- | --- | --- | --- |
| `vfs_contract_pack` | `program_contract_profile` | `vfs-contract-pack@1`, `canonical-operation-matrix@1`, `drift-inventory@1` | Immutable operation/invariant profile + drift inventory without domain constants |
| `vfs_differential_harness` | `differential_contract_harness` | `differential-contract-witness@1`, `canonical-operation-trace@1` | Hermetic multi-surface differential runner |
| `vfs_mcp_contract_checker` | `interface_contract_parity` | `vfs-mcp-parity-report@1` family | Interface/tool parity without MCP/VFS literals in the engine |
| `vfs_symbolic_benchmark` | `symbolic_efficiency_benchmark` | `symbolic-efficiency-benchmark@1` family | Cache/scan efficiency gates over profiled stages |
| `vfs_symbolic_pilot` | `symbolic_assurance_pilot` | pilot manifest/coverage/stage schemas | Bounded pilot orchestration; domain path filters in profile |
| `vfs_symbolic_rollout` | `symbolic_assurance_rollout` | rollout decision + control + adversarial gate schemas | Control surface + adversarial gates; binding via integration |

## Package ownership

| Destination package | Owns |
| --- | --- |
| `analysis/` | inventory, program contract profile, interface contract parity |
| `validation/` | differential harness, symbolic efficiency benchmark |
| `runtime/` | symbolic assurance pilot |
| `control/` | symbolic assurance rollout control surface |
| `integrations/` | `ipfs_kit_vfs_assurance` job assembly only (later task) |
| `scripts/ops/agent_supervisor/` | thin CLI facade only (later task) |

Dependency direction follows
[PACKAGE_MAP.md](PACKAGE_MAP.md). Generic engines must not import optional
providers implicitly.

## Callers and entry points (source lineage)

| Source caller pattern | Post-generalization disposition |
| --- | --- |
| Direct import of `agent_supervisor.vfs_*` | Forbidden after cutover; migrate to package modules |
| Pilot / rollout CLI | Thin facade → integration → generic engines |
| Tests under `test/api/test_*vfs*` | Replaced or dual-run via generic profile tests |
| Root `vfs_*.py` shims | Forbidden; no compatibility re-export modules |

## Non-VFS parameterization proof

LPR-021 requires a hermetic non-VFS fixture profile that traverses the same
inventory engine. That profile supplies its own content/path signals and kind
specs. Matching domain results under a VFS-equivalent profile plus distinct
results under a non-VFS profile proves the engine is parameterized rather than
hard-coded.

## Worker constraints

- Read only exact source-lock Git blobs and this map / lock.
- Never merge or cherry-pick revision `0cc04ebb…` as a broad snapshot.
- Never copy a root `vfs_*` module into the target tree.
- Never import scanned repository code during inventory.
- Never embed repository bodies in the lock or this map.
- Generic modules contain no VFS / IPFS / fsspec / SwissKnife literals, fixed
  repository aliases, board IDs, or implicit provider imports.

## Related

- Plan §4.12 in `AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md`
- Task `LPR-021` in the tactician-hammer logic repair board
- Generic inventory:
  `ipfs_accelerate_py/agent_supervisor/analysis/repository_surface_inventory.py`
