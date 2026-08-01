# IPFS Kit VFS Symbolic Assurance — Goal Packet Evidence

**Primary task:** VFS-062  
**Primary goal:** VFS-G134  
**Primary evidence term:** `vfs/contract-assurance-root@1`  
**Related goal packets:** `13f52635926c` (VFS-055), `7414e86c1a7c` (VFS-052), `ab6d1ed417d3` (VFS-056)  
**Bundle:** `vfs-assurance/root`  
**Track:** vfs-symbolic-assurance  

This directory is the declared durable output for the VFS symbolic-assurance
root evidence package. It binds the VFS-G000 root evidence terms to live
producers, tests, and a frozen multi-repository forest fixture without editing
operator-protected architecture files.

VFS-062 materializes the typed root envelope receipt that proves
`vfs/contract-assurance-root@1` while retaining sibling leaf receipts for the
shared package terms under VFS-G132, VFS-G133, and VFS-G135.

## Covered evidence terms

| Evidence term | Goal | Receipt |
| --- | --- | --- |
| `vfs/contract-assurance-root@1` | VFS-G134 | [`receipts/contract_assurance_root.receipt.json`](./receipts/contract_assurance_root.receipt.json) |
| `vfs/repository-forest-receipt@1` | VFS-G132 | [`receipts/repository_forest_receipt.receipt.json`](./receipts/repository_forest_receipt.receipt.json) |
| `vfs/exhaustive-index-receipt@1` | VFS-G133 | [`receipts/exhaustive_index_receipt.receipt.json`](./receipts/exhaustive_index_receipt.receipt.json) |
| `vfs/autonomous-refill-exhaustion@1` | VFS-G135 | [`receipts/autonomous_refill_exhaustion.receipt.json`](./receipts/autonomous_refill_exhaustion.receipt.json) |

### Packet aggregates and narratives

| Artifact | Role |
| --- | --- |
| [`evidence_coverage_vfs_g134.md`](./evidence_coverage_vfs_g134.md) | VFS-062 / VFS-G134 primary coverage |
| [`goal_packet_13f52635926c_evidence.json`](./goal_packet_13f52635926c_evidence.json) | Packet aggregate (VFS-055) |
| [`goal_packet_ab6d1ed417d3_evidence.json`](./goal_packet_ab6d1ed417d3_evidence.json) | Packet aggregate (VFS-056) |
| [`goal_packet_7414e86c1a7c_evidence.json`](./goal_packet_7414e86c1a7c_evidence.json) | Packet aggregate (VFS-052) |
| [`evidence_coverage_packet_13f52635926c.md`](./evidence_coverage_packet_13f52635926c.md) | Packet narrative |
| [`evidence_coverage_packet_ab6d1ed417d3.md`](./evidence_coverage_packet_ab6d1ed417d3.md) | Packet narrative |
| [`evidence_coverage_packet_7414e86c1a7c.md`](./evidence_coverage_packet_7414e86c1a7c.md) | Packet narrative |

Term index: [`covered_evidence_terms.json`](./covered_evidence_terms.json)  
Gap resolution: [`objective_gap.json`](./objective_gap.json)  
Seed manifest: [`seed_manifest.json`](./seed_manifest.json)

## Exact-term bindings (discovery keys)

The following strings are first-class evidence identifiers for this package:

- `vfs/contract-assurance-root@1`
- `vfs/repository-forest-receipt@1`
- `vfs/exhaustive-index-receipt@1`
- `vfs/autonomous-refill-exhaustion@1`

## Producer map (no parallel authority)

| Term | Primary producers | Validation |
| --- | --- | --- |
| `vfs/contract-assurance-root@1` | `vfs_symbolic_rollout` e2e root, pilot, contract checker, assurance contracts | `test_vfs_symbolic_assurance_e2e.py` |
| `vfs/repository-forest-receipt@1` | `repository_forest.py`, `repository_forest_manifest.py`, `freeze_multi_repository_fixture` | forest tests, e2e |
| `vfs/exhaustive-index-receipt@1` | `repository_corpus_index.py` (exhaustive inventory), AST adapters, adversarial inventory gates | corpus index tests, e2e |
| `vfs/autonomous-refill-exhaustion@1` | `symbolic_finding_refill.py` (`HealthyExhaustionReceipt`, `RefillReason.HEALTHY_EXHAUSTED`), `AdversarialGateId.REFILL_EXHAUSTION` / `BOUNDED_REFILL` | refill tests, e2e |

## Frozen fixture identity

Receipts bind the reproducible adversarial multi-repository fixture:

- `forest_id` / `fixture_cid`: `sha256:136501f1ba1a3c0e89973f7f138ee41f1a6ada91ac0aabc648ea947a14b4d8f2`
- adversarial e2e report: `sha256:5cb952ea3d9fe7289389f616c98a2b414490c2ec2a1427b2ac0e3d1396548422`
- four repositories, each `exhaustive=true`
- 10 included / 6 excluded paths with policy-bound exclusion prefixes

## Authority limits

- This package is **not** completion-authoritative.
- Receipts set `safe_for_completion_reasoning=false` and `authoritative=false`.
- Write authority, merge gates, and objective completion remain under the
  existing supervisor control path and protected architecture files.
- External SwissKnife remains read-only in the initial policy.
- Refill proposals never authorize completion or execution
  (`REFILL_AUTHORIZES_COMPLETION=false`, `REFILL_AUTHORIZES_EXECUTION=false`).

## Validation

```bash
python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
test -d data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/receipts/contract_assurance_root.receipt.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/receipts/repository_forest_receipt.receipt.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/receipts/exhaustive_index_receipt.receipt.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/receipts/autonomous_refill_exhaustion.receipt.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/covered_evidence_terms.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/objective_gap.json
grep -F 'vfs/contract-assurance-root@1' data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/covered_evidence_terms.json
```

## Backlog alignment

The supervisor-fed board
`docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md` and objective heap
`docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md` already name
VFS-G132..G135 as leaf owners for these terms. No child-goal refinement is
required: VFS-G134 is the smallest stable evidence owner for
`vfs/contract-assurance-root@1` under VFS-G000.
Protected plan/todo/validator paths are not modified by this package.

VFS-062 closes the objective scan gap for `vfs/contract-assurance-root@1` by
materializing the root envelope receipt and retaining sibling leaf receipts in
one cohesive package under `data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance`.
