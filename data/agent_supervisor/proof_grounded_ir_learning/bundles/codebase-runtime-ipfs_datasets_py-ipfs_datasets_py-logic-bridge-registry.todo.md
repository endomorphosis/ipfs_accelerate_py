# Codebase Bundle: codebase/runtime/ipfs_datasets_py-ipfs_datasets_py-logic-bridge-registry

Source todo: docs/architecture/proof_grounded_ir_learning.todo.md
Purpose: group generated codebase findings by source file and AST locality.
Conflict policy: serialize edits to one file; allow independent file bundles to run concurrently.

## PGIR-113 Replace placeholder runtime path in ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py:251

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on: 
- Outputs: /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery, ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py
- Validation: python3 -m py_compile ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py
- Board namespace: proof_grounded_ir_learning.todo.md
- Graph parents: PGIR-G000
- Graph depth: 1
- Goal id: PGIR-G040
- Goal lineage: PGIR-G040, PGIR-G000
- Goal registration: existing
- Canonical task key: task/v1/b4b510c6ad8ba07a515c0bd05bd59a2112f717449d24efd67f44a5602e1cca63
- Canonical task CID: baguqeeraws2rbrvnroqhuuk4bpifxvm2eejpof2etuso7vt7isswalq4zjrq
- Semantic identity: b4b510c6ad8ba07a515c0bd05bd59a2112f717449d24efd67f44a5602e1cca63
- Acceptance subset: Resolve placeholder_runtime_path at ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py:251
- Preconditions: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py exists and the scan evidence remains applicable
- Effects: resolve placeholder_runtime_path in ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py and pass focused validation
- Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py:251, /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-16-pgir-113-codebase-scan-2049598c9447.md
- Resource class: cpu-small
- Token class: small
- Resources: python, focused validation runner
- Merge fate: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py
- Rejection reasons: none
- Missing evidence: Replace placeholder runtime path in ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py:251
- Candidate kind: codebase_scan
- Todo vector key: 2049598c94470b41
- Bundle: codebase/runtime/ipfs_datasets_py-ipfs_datasets_py-logic-bridge-registry
- Bundle shard: data/agent_supervisor/proof_grounded_ir_learning/bundles/codebase-runtime-ipfs_datasets_py-ipfs_datasets_py-logic-bridge-registry.todo.md
- Bundle strategy: codebase_file_ast
- Parallel lane: codebase/runtime/ipfs_datasets_py-ipfs_datasets_py-logic-bridge-registry
- Conflict policy: serialize findings for the same file; allow independent file bundles to run concurrently
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py
- AST symbols: __future__, __future__.annotations, bridge name for component, bridge_name_for_component, dataclasses, dataclasses dataclass, dataclasses.dataclass, future, future annotations, importlib, load logic bridge adapter, load_logic_bridge_adapter, logic bridge manifest, logic bridge spec, logic bridge specs, logic_bridge_manifest, logic_bridge_spec, logic_bridge_specs, logicbridgespec, logicbridgespec to dict, logicbridgespec.to_dict, to dict, to_dict, typing, typing any, typing mapping, typing optional, typing.any, typing.mapping, typing.optional
- AST symbol scope: file
- Merge key: codebase/runtime/ipfs_datasets_py-ipfs_datasets_py-logic-bridge-registry
- Merge family: ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py
- Merge role: codebase_scan
- Work item count: 1
- Work scope: codebase_file_ast
- Acceptance: Goal-scoped refill admitted this finding from ipfs_datasets_py/ipfs_datasets_py/logic/bridge/registry.py:251 for PGIR-G040. Use evidence in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-16-pgir-113-codebase-scan-2049598c9447.md, make only the smallest change required by that goal lineage, add or update focused validation when appropriate, and do not expand into adjacent cleanup.
