# Evidence Coverage — VFS-G134 / `vfs/contract-assurance-root@1`

**Task:** VFS-062  
**Goal:** VFS-G134 — Prove and repair IPFS Kit VFS contracts from a content-addressed SwissKnife program graph (root envelope)  
**Date:** 2026-07-30  
**Discovery input:** `/home/barberb/.local/state/ipfs_accelerate_py/ipfs_kit_vfs_symbolic_assurance/projection/discovery/2026-07-30-vfs-062-objective-gap-63a470e6d548.md`  
**Fingerprint:** `63a470e6d54804ee5e67ac758689da07d52f0b4d`  
**Bundle / lane:** `vfs-assurance/root`  
**Track:** vfs-symbolic-assurance  
**Status:** implemented (nomination/provenance package; completion remains subject to the objective completion gate)

## Missing evidence terms (from objective scan)

| Evidence term | Goal | Role | Coverage produced by VFS-062 |
| --- | --- | --- | --- |
| `vfs/contract-assurance-root@1` | VFS-G134 | Root contract-assurance envelope for VFS-G000 | Typed aggregate receipt bound to adversarial e2e root, shadow rollout, control parity, and sibling leaf receipts |

Discovery reported **no present evidence** for this goal. This package materializes the durable leaf receipt and maps it to live producers and the acceptance validation command without rewriting operator-protected architecture files.

## Acceptance subset checklist (VFS-G000 / VFS-G134)

| Criterion | Proof artifact |
| --- | --- |
| Every direct child has fresh criterion-level evidence | Child leaf receipts under `receipts/` + root aggregate |
| No unsupported or stale result promoted to proof | Receipts mark non-authoritative / not completion-safe; e2e shadow mode |
| Complete frozen repository inventory accounted for | Child `vfs/exhaustive-index-receipt@1` + exhaustive per-repo fixture |
| Accepted repairs retain normal auth/validation/merge gates | Root rollout decision + control parity / automatic-mutation-disabled gates |
| Drained work → bounded refill or exhaustion receipt | Child `vfs/autonomous-refill-exhaustion@1` + `HEALTHY_EXHAUSTED` / gate pair |

## Requirement evidence map

### VFS-G134 — `vfs/contract-assurance-root@1` (primary for VFS-062)

- **Producer:** `ipfs_accelerate_py.agent_supervisor.vfs_symbolic_rollout` adversarial e2e root (`vfs/adversarial-e2e-gate@1`, shadow rollout), with supporting surfaces in pilot, MCP contract checker, program assurance contracts, and VFS contract pack.
- **Evidence id:** `vfs/contract-assurance-root@1` (aggregate parent for the four VFS-G000 root terms).
- **Schema:** `vfs/contract-assurance-root@1` on `receipts/contract_assurance_root.receipt.json`.
- **Invariants:**
  - shadow mode default; no automatic mutation
  - control parity across Python/CLI/MCP projections
  - no stale or unsupported result promoted to proof
  - accepted repairs retain normal auth/validation/merge gates
- **Frozen binding:** four-repository adversarial fixture (`forest_id` / `fixture_cid` `sha256:136501f1ba1a3c0e89973f7f138ee41f1a6ada91ac0aabc648ea947a14b4d8f2`); clean adversarial population passes all 26 required gates (report `sha256:5cb952ea3d9fe7289389f616c98a2b414490c2ec2a1427b2ac0e3d1396548422`).
- **Child evidence terms (not promoted as root substitutes):**
  - `vfs/repository-forest-receipt@1` → `receipts/repository_forest_receipt.receipt.json`
  - `vfs/exhaustive-index-receipt@1` → `receipts/exhaustive_index_receipt.receipt.json`
  - `vfs/autonomous-refill-exhaustion@1` → `receipts/autonomous_refill_exhaustion.receipt.json`
- **Receipt:** `receipts/contract_assurance_root.receipt.json`
- **Index:** `covered_evidence_terms.json`, `objective_gap.json`
- **Tests:** `test/api/test_vfs_symbolic_assurance_e2e.py` (primary packet validation)

## Shared package terms (sibling leaves under VFS-G000)

| Evidence term | Goal | Receipt |
| --- | --- | --- |
| `vfs/repository-forest-receipt@1` | VFS-G132 | `receipts/repository_forest_receipt.receipt.json` |
| `vfs/exhaustive-index-receipt@1` | VFS-G133 | `receipts/exhaustive_index_receipt.receipt.json` |
| `vfs/autonomous-refill-exhaustion@1` | VFS-G135 | `receipts/autonomous_refill_exhaustion.receipt.json` |

Related packet aggregates retained for cohesion:

- `goal_packet_13f52635926c_evidence.json` (VFS-055)
- `goal_packet_ab6d1ed417d3_evidence.json` (VFS-056)
- `goal_packet_7414e86c1a7c_evidence.json` (VFS-052)

## Backlog and heap alignment

- VFS-062 is the board task that closes the objective scan gap for **VFS-G134** / `vfs/contract-assurance-root@1`.
- Discovery listed **no goal packet** for this task; the root term is still shared by packets `13f52635926c`, `7414e86c1a7c`, and `ab6d1ed417d3`.
- **No objective-heap child refinement** is required: VFS-G134 is already the smallest stable evidence owner for this term under VFS-G000; sibling leaves VFS-G132/G133/G135 own the other root terms.
- Operator-protected files are **not** rewritten:
  - `docs/architecture/IPFS_KIT_VFS_SYMBOLIC_ASSURANCE_PLAN.md`
  - `docs/architecture/ipfs_kit_vfs_symbolic_assurance.objectives.md`
  - `docs/architecture/ipfs_kit_vfs_symbolic_assurance.todo.md`
  - `scripts/ops/agent_supervisor/validate_ipfs_kit_vfs_symbolic_assurance.py`

## Exact evidence term strings

These discovery keys appear in this package and in the bound producers/tests:

1. `vfs/contract-assurance-root@1` (**primary**)
2. `vfs/repository-forest-receipt@1`
3. `vfs/exhaustive-index-receipt@1`
4. `vfs/autonomous-refill-exhaustion@1`

## Validation

Primary acceptance command:

```text
python -m pytest test/api/test_vfs_symbolic_assurance_e2e.py -q
```

Package integrity checks:

```text
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/receipts/contract_assurance_root.receipt.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/covered_evidence_terms.json
test -f data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/objective_gap.json
grep -F 'vfs/contract-assurance-root@1' data/agent_supervisor/ipfs_kit_vfs_symbolic_assurance/covered_evidence_terms.json
```

This discovery/coverage artifact documents the producer and test map. It is a
nomination and durable package record, not standalone completion authority;
fresh current-tree validation and the canonical objective completion transition
remain required.
