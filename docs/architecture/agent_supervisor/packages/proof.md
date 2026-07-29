# agent_supervisor.proof

**Code:** `ipfs_accelerate_py/agent_supervisor/proof/` · [code README](../../../../ipfs_accelerate_py/agent_supervisor/proof/README.md) · [Developer guide](../DEVELOPER_GUIDE.md) · [Package map](../PACKAGE_MAP.md)


## Purpose

Formal verification and codebase-proof surfaces: proof cache, provers, obligations, queries, re-proof, attestation, property catalog, and related assurance modules.

## When to use this package

You are adding properties, obligations, prover adapters, cache behavior, or attestation policy.

## Public modules

| Module | Role |
| --- | --- |
| `formal_verification_cache` | Trust-aware proof cache |
| `formal_verification_provider` | Provider adapters |
| `formal_verification_policy` | Assurance policy |
| `formal_verification_contracts` | Proof contracts |
| `formal_verification_capabilities` | Capability probes |
| `code_property_catalog` | Reviewed property catalog |
| `code_claim_contracts` | Claim / evidence lifecycle |
| `code_proof_obligations` | Obligation compiler |
| `code_proof_query` | Query API (open/satisfied/impact/delta) |
| `code_proof_reproof` | Cache-aware re-proof |
| `code_proof_context` | Obligation-first context helpers |
| `code_edit_packet` | Edit packet contracts |
| `code_edit_materialize` | Materialize implementable packets |
| `proof_attestation` | Attestation / ZK policy bindings |
| `prover_matrix_registry` | Prover matrix registry |
| `multi_prover_router` | Multi-prover routing |
| `kernel_verification` | Kernel-checked verification |

Prefer absolute imports:

```python
from ipfs_accelerate_py.agent_supervisor.proof import ...
# or
from ipfs_accelerate_py.agent_supervisor.proof.<module> import ...
```

## Dependencies

| Direction | Rule |
| --- | --- |
| **Inbound** | Planning, context, validation, codebase-proof programs, control capability probes. |
| **Outbound** | `core`; optional solver backends only through explicit provider paths. |
| **Forbidden** | Importing `todo_daemon` implementation loops; silent trust promotion across evidence tiers. |

## Extension notes

1. Keep the package DAG acyclic ([package map](../PACKAGE_MAP.md)).
2. Use **semantic** symbol names; do not name public APIs after board prefixes.
3. Update this README when you add or move modules.
4. Add focused tests under `test/api/` (or the package’s established suite).

## Program evidence (optional)

Historical domain-layout and feature programs may cite this package in boards
and objective heaps. See [PROGRAMS.md](../PROGRAMS.md). Product code and docs
should not require those IDs to understand the package.