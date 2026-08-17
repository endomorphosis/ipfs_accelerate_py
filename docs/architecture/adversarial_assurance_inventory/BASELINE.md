# AAE-005 focused baseline and authority join

**Task:** AAE-005  
**Interface:** `AssuranceAuthorityMatrix@1`  
**Controller head at join:** `bdfff1240b6e707a2d2106210d546d25fde3064d`  
**Protected runner:** `docs/architecture/adversarial_assurance_inventory/run_focused_baselines.py`

This note is the human-readable twin of `authority_matrix.json`. It does not
replace the matrix, the runner, or the closed receipts. Workers may not weaken
the recorded RED, stale, or unavailable findings.

## Repository forest

| Authority | Path | Commit | Mode |
| --- | --- | --- | --- |
| Accelerate controller | `.` on `agent/adversarial-assurance-engine-v1` | `bdfff1240b6e707a2d2106210d546d25fde3064d` | exact current-tree head |
| Accelerate planning ancestor | same repository | `7c9f3fa3d2ac14c7b5bfa5036e2fe6fb59f0afda` | required ancestor |
| Datasets | `ipfs_datasets_py` | `fbd1ba9f70803de157622bb20e22595ef09d606f` | exact gitlink |
| Kit | `ipfs_kit_py` | `c7e5feeb24582ab68c1f5ca626366b665a82ad61` | exact gitlink |
| MCP++ | `ipfs_accelerate_py/mcplusplus` | `dc3164653a48d059ae9812078359daeafb451c07` | exact gitlink |
| IVP public-API freeze | `ipfs_accelerate_py/agent_supervisor/verification` | `8c7800cedc5e1b848367db9952f912428466f8cc` | historical freeze |

AAE-001 inventoried accelerate at `3256d8bb`. Later AAE-002/003/004 merges
advanced the controller head. Nested gitlinks did not change.

## Reuse and non-reimplementation

| Owner | Reuse | Forbidden |
| --- | --- | --- |
| `ipfs_datasets_py` | CID profile, incremental semantic index, functional `SemanticCapsuleCompiler@1`, claim/property/vacuity surfaces that already exist | second CID, second scanner/compiler, invented general vacuity |
| `ipfs_accelerate_py` | isolated worktrees, resource admission, IVP, receipt cache, model routing, `ContextPacker`, execution and evaluation | new scheduler, new context packer, new receipt format, local proof system |
| `ipfs_kit_py` | durable blocks, root CAS, recovery, campaign-history patterns | computing canonical CIDs, owning semantic index |
| MCP++ | existing Profiles A/B/F/G, vectors, cross-language conformance | a new profile |

Missing `IncrementalProofSealer`, `FullCheckpointSeal`, and `DeltaSeal` public
APIs remain `typed_unavailable`. An IVP Merkle commitment is not a seal.

## Known RED / stale / unavailable

- **RED / typed blocker:** frozen SCG authority matrix still expects datasets
  `1330038f...` while the live AAE gitlink is `fbd1ba9f...`. The check stays.
  This is not AAE success and is not permission to weaken the matrix.
- **RED non-authoritative:** IVP benchmark remains red (seeded false negative,
  false positives, unavailable real provers).
- **Unavailable:** planned `ipfs_accelerate_py.agent_supervisor.adversarial_assurance`
  package; released sealer checkpoint/delta APIs; integrated mutation campaign,
  expected-detection, and four-family vacuity engines.

## Protected runner

The runner accepts only:

- `--current-tree` — execute the four inventoried suites and write closed
  receipts under `baseline_receipts/`
- `--verify-bundle --output-dir <root>` — recompute path/CID bindings from
  existing receipts; do not trust declared pass counts

Reviewed output roots:

- `docs/architecture/adversarial_assurance_inventory/baseline_receipts`
- `docs/architecture/adversarial_assurance_inventory/prerequisite_evidence`

Each receipt binds repository identity, repository-state root, exact argv,
return code, actual counts, bounded log digest, canonical UTC interval,
environment and dependency-lock CIDs, disabled network, and absent production
credentials.

Planning-time observations (not current-tree authority):

- datasets semantic index/capsule/identity: 152 passed
- accelerate verification and semantic-governor selection: 283 passed, 1 failed / 284
- kit durable roots and governor store: 196 passed
- MCP++ CID, event DAG, conformance, Profile G: 58 passed

Current-tree receipts at `bdfff1240`:

| Suite | Passed | Failed | Status |
| --- | ---: | ---: | --- |
| datasets | 503 | 0 | passed |
| accelerate | 215 | 1 | failed: SCG authority matrix still expects datasets `1330038f` |
| ipfs_kit_py | 230 | 0 | passed |
| mcp_plus_plus | 72 | 1 | failed: `test_dispatch_artifact_explicit_false_overrides_enabled_default_policy` still emits `event_dag` |

The SCG matrix failure is the planned typed blocker. The MCP++ emit=false
override failure is a new current-tree finding. Neither is hidden and neither
is treated as AAE success.

## Operator gate

`AAE-006` stays blocked. Workers may not complete it. Runtime and sealing
tasks stay unauthorized until a signed prerequisite receipt proves terminal
SCG, released sealer APIs, exact clean pins, and fresh green baselines at
those pins.
