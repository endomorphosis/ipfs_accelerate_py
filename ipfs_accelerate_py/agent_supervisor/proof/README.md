# agent_supervisor.proof

**Layer:** Mid · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Formal verification and attestation: prover contracts, caches, capabilities, and codebase-proof support modules.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, analysis contracts as needed |
| **Typical dependents** | planning, validation, context, control probes |

## Modules

| Module | Path |
| --- | --- |
| `code_claim_contracts` | `proof/code_claim_contracts.py` |
| `code_edit_materialize` | `proof/code_edit_materialize.py` |
| `code_edit_packet` | `proof/code_edit_packet.py` |
| `code_proof_context` | `proof/code_proof_context.py` |
| `code_proof_obligations` | `proof/code_proof_obligations.py` |
| `code_proof_query` | `proof/code_proof_query.py` |
| `code_proof_reproof` | `proof/code_proof_reproof.py` |
| `code_proof_srt_bridge` | `proof/code_proof_srt_bridge.py` |
| `code_property_catalog` | `proof/code_property_catalog.py` |
| `formal_counterexamples` | `proof/formal_counterexamples.py` |
| `formal_logic_vocabulary` | `proof/formal_logic_vocabulary.py` |
| `formal_verification_cache` | `proof/formal_verification_cache.py` |
| `formal_verification_capabilities` | `proof/formal_verification_capabilities.py` |
| `formal_verification_contracts` | `proof/formal_verification_contracts.py` |
| `formal_verification_policy` | `proof/formal_verification_policy.py` |
| `formal_verification_provider` | `proof/formal_verification_provider.py` |
| `hyperproperty_verification` | `proof/hyperproperty_verification.py` |
| `intent_constraint_adapter` | `proof/intent_constraint_adapter.py` |
| `interface_contract_codegen` | `proof/interface_contract_codegen.py` |
| `ir_adapters` | `proof/ir_adapters.py` |
| `ir_constraint_compiler` | `proof/ir_constraint_compiler.py` |
| `ir_registry` | `proof/ir_registry.py` |
| `kernel_verification` | `proof/kernel_verification.py` |
| `leanstral_goal_benchmark` | `proof/leanstral_goal_benchmark.py` |
| `leanstral_goal_development` | `proof/leanstral_goal_development.py` |
| `leanstral_goal_lifecycle` | `proof/leanstral_goal_lifecycle.py` |
| `leanstral_proof_provider` | `proof/leanstral_proof_provider.py` |
| `legal_constraint_adapter` | `proof/legal_constraint_adapter.py` |
| `logic_translation_validation` | `proof/logic_translation_validation.py` |
| `multi_prover_resources` | `proof/multi_prover_resources.py` |
| `multi_prover_router` | `proof/multi_prover_router.py` |
| `proof_attestation` | `proof/proof_attestation.py` |
| `proof_context` | `proof/proof_context.py` |
| `proof_directed_retrieval` | `proof/proof_directed_retrieval.py` |
| `proof_fallbacks` | `proof/proof_fallbacks.py` |
| `proof_metrics` | `proof/proof_metrics.py` |
| `proof_obligation_templates` | `proof/proof_obligation_templates.py` |
| `proof_scheduler` | `proof/proof_scheduler.py` |
| `proof_scope_index` | `proof/proof_scope_index.py` |
| `protocol_verification` | `proof/protocol_verification.py` |
| `prover_conformance` | `proof/prover_conformance.py` |
| `prover_evidence_store` | `proof/prover_evidence_store.py` |
| `prover_matrix_registry` | `proof/prover_matrix_registry.py` |
| `security_constraint_adapter` | `proof/security_constraint_adapter.py` |
| `supervisor_code_proof_benchmark` | `proof/supervisor_code_proof_benchmark.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.proof.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/proof.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
