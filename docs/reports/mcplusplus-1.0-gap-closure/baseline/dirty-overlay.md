# MCPP-001 Dirty Overlay Inventory

Schema: `DirtyOverlayInventory@1`

- Task: `MCPP-001`
- Goal: `MCPP-G010`
- Observed at: `2026-08-15T09:31:31Z`
- Program branch: `codex/mcplusplus-1.0-gap-closure`
- Policy: record only; no reset, stash-drop, checkout, clean, or force-push
- Result: every inventoried dirty and untracked path was left untouched

## Summary

| Checkout | Role | HEAD | Branch | Dirty count | Program branch |
| --- | --- | --- | --- | ---: | --- |
| `lift_coding` | superproject | `b6f40c05e088` | `chore/fmt-check-main` | 19 | present `b6f40c05e088` |
| `Mcp-Plus-Plus` | protocol_conformance | `6965f89f0667` | `main` | 0 | present `6965f89f0667` |
| `ipfs_accelerate_py` | runtime_accelerate_implementation_worktree | `e01a7f4ec4ac` | `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925` | 2 | present `e01a7f4ec4ac` |
| `ipfs_accelerate_py_operator` | runtime_accelerate_operator | `ea11293bb996` | `fix/implementation-auto-rescue-20260809` | 147 | present `e01a7f4ec4ac` |
| `ipfs_datasets_py` | runtime_datasets_worktree | `ac82107e246b` | `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925-submodule-ipfs_datasets_py` | 0 | present `ac82107e246b` |
| `ipfs_datasets_py_operator` | runtime_datasets_operator | `ac82107e246b` | `main` | 8 | present `ac82107e246b` |
| `ipfs_kit_py` | runtime_kit_worktree | `6196017ca3df` | `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925-submodule-ipfs_kit_py` | 0 | present `6196017ca3df` |
| `ipfs_kit_py_operator` | runtime_kit_operator | `6196017ca3df` | `main` | 0 | present `6196017ca3df` |
| `mcplusplus_gitlink` | accelerate_nested_spec_submodule | `6965f89f0667` | `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925-submodule-ipfs_accelerate_py-mcplusplus` | 0 | present `6965f89f0667` |
| `swissknife` | runtime_swissknife | `afdbf885175f` | `main` | 1 | present `afdbf885175f` |

## Spec authority and drift

- `lift_coding/Mcp-Plus-Plus` HEAD: `6965f89f066769f3b3ac7b5f753b1a0044562570`
- `ipfs_accelerate_py/mcplusplus` gitlink HEAD: `6965f89f066769f3b3ac7b5f753b1a0044562570`
- Same SHA: `True`
- Note: Both checkouts currently share the same HEAD; bind lift_coding/Mcp-Plus-Plus as spec authority when they diverge. Nested gitlink is older only when SHAs differ.

## SwissKnife binding (discovered, not invented)

- Bound checkout: `/home/barberb/lift_coding/swissknife`
- HEAD: `afdbf885175fde34505ef05a2ea6aac5535ad03e`
- Branch: `main`
- Remote `origin`: `https://github.com/endomorphosis/swissknife`
- Remote `upstream`: `https://github.com/dnakov/anon-kode.git`
- Not an accelerate submodule; sibling under `lift_coding`.

## Program branch actions

- `lift_coding`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `b6f40c05e0884867eb8557f8882cd25cb760ca2f`)
- `Mcp-Plus-Plus`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `6965f89f066769f3b3ac7b5f753b1a0044562570`)
- `ipfs_accelerate_py`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `e01a7f4ec4ac2275f64a8f980ee614c10ebaae5e`)
- `ipfs_accelerate_py_operator`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `e01a7f4ec4ac2275f64a8f980ee614c10ebaae5e`)
- `ipfs_datasets_py`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `ac82107e246b30e35a2bbdcf75e01370d22350c6`)
- `ipfs_datasets_py_operator`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `ac82107e246b30e35a2bbdcf75e01370d22350c6`)
- `ipfs_kit_py`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `6196017ca3df016c7159dce43af60f2a0d96a9ae`)
- `ipfs_kit_py_operator`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `6196017ca3df016c7159dce43af60f2a0d96a9ae`)
- `mcplusplus_gitlink`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `6965f89f066769f3b3ac7b5f753b1a0044562570`)
- `swissknife`: **already_present** (`codex/mcplusplus-1.0-gap-closure` sha `afdbf885175fde34505ef05a2ea6aac5535ad03e`)

No operator checkout was mutated to create branches. Where the program branch
was already present it is recorded; where absent it is recorded as absent for
later isolated-worktree creation.

## Per-checkout dirty paths

### `lift_coding`

- Path: `/home/barberb/lift_coding`
- Role: `superproject`
- HEAD: `b6f40c05e0884867eb8557f8882cd25cb760ca2f`
- Tree: `db6c28fdc119fab6cd6c109412c5d1ccda02c0c6`
- Branch: `chore/fmt-check-main`
- Remotes: `{"origin": "https://github.com/endomorphosis/lift_coding.git", "wt": "/home/barberb/.local/state/ipfs_accelerate_py/proof-backed-test-reuse-v9/worktrees/ptr_lane_1/workspace-01d0ca0bb69f-d3043386de3c/external/ipfs_datasets"}`
- Dirty: `True` (count=19)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `b6f40c05e0884867eb8557f8882cd25cb760ca2f`
- Notes: Operator superproject; dirty gitlinks and untracked worktree/backup dirs preserved.
- Dirty paths:
  - ` M` `external/ipfs_accelerate`
  - ` M` `external/ipfs_datasets`
  - ` M` `hallucinate_app`
  - ` M` `swissknife`
  - `??` `.backups/`
  - `??` `.codex-sandbox-probe.IbJduK/`
  - `??` `.cvefixes-build/`
  - `??` `.dqp-recovery-candidates.G6apeG/`
  - `??` `.dqp012-diagnostic.kIF1vO/`
  - `??` `.dqp016-review.U8kbij/`
  - `??` `.git-sync-recovery-20260727-202941.md`
  - `??` `.kit-semantic-state-supervisor-worktrees/`
  - `??` `.merge-backups/`
  - `??` `.recovery-backups/`
  - `??` `.tmp-dqp008-recovery.YTuEbE/`
  - `??` `.tmp-readme-compare/`
  - `??` `.venvs/`
  - `??` `.worktrees/`
  - `??` `tmp_readme_compare/`
- Recorded gitlinks (mode 160000):
  - `Mcp-Plus-Plus` @ `6965f89f066769f3b3ac7b5f753b1a0044562570`
  - `external/ipfs_accelerate` @ `485edc0871c55b0e2ef21d83bece9fa12c2c8d84`
  - `external/ipfs_datasets` @ `ac82107e246b30e35a2bbdcf75e01370d22350c6`
  - `external/ipfs_kit` @ `6196017ca3df016c7159dce43af60f2a0d96a9ae`
  - `external/meta-wearables-dat-android` @ `4e56e1864a5e78194bababc3a68775c4196cbed0`
  - `external/meta-wearables-dat-ios` @ `2b5695d16a710f3d2d7341f88570b86d01723d50`
  - `hallucinate_app` @ `8a5f89e6d4c1566675d2676b84a99abcf9c48419`
  - `swissknife` @ `afdbf885175fde34505ef05a2ea6aac5535ad03e`

### `Mcp-Plus-Plus`

- Path: `/home/barberb/lift_coding/Mcp-Plus-Plus`
- Role: `protocol_conformance`
- HEAD: `6965f89f066769f3b3ac7b5f753b1a0044562570`
- Tree: `fd328606da5dae3314a87365987f5052180eb807`
- Branch: `main`
- Remotes: `{"origin": "https://github.com/endomorphosis/Mcp-Plus-Plus.git"}`
- Dirty: `False` (count=0)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `6965f89f066769f3b3ac7b5f753b1a0044562570`
- Notes: Canonical Mcp-Plus-Plus spec authority checkout under lift_coding.
- Dirty paths: *(none)*

### `ipfs_accelerate_py`

- Path: `/home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/worktrees/workspace_4fbf5a625b5c_cf8849b0b2e2`
- Role: `runtime_accelerate_implementation_worktree`
- HEAD: `e01a7f4ec4ac2275f64a8f980ee614c10ebaae5e`
- Tree: `b4b7df739b31fab7304771499c9e48ccfd7bb085`
- Branch: `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925`
- Remotes: `{"origin": "https://github.com/endomorphosis/ipfs_accelerate_py"}`
- Dirty: `True` (count=2)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `e01a7f4ec4ac2275f64a8f980ee614c10ebaae5e`
- Notes: Isolated supervisor implementation worktree for MCPP-001; program branch already present.
- Dirty paths:
  - `A ` `docs/reports/mcplusplus-1.0-gap-closure/baseline/dirty-overlay.md`
  - `A ` `docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json`
- Recorded gitlinks (mode 160000):
  - `docs/fastmcp` @ `1d932cc778a24cc0bf46fc4baad8306d4fed9c4b`
  - `docs/mcp-python-sdk` @ `0da9a074d09267a927d72faa58c26d828f0f8edb`
  - `ipfs_accelerate_py/mcplusplus` @ `6965f89f066769f3b3ac7b5f753b1a0044562570`
  - `ipfs_datasets_py` @ `ac82107e246b30e35a2bbdcf75e01370d22350c6`
  - `ipfs_kit_py` @ `6196017ca3df016c7159dce43af60f2a0d96a9ae`
  - `ipfs_model_manager_py` @ `f6151d2113f42e75ea7d83a1b2362fc97e55e44d`
  - `ipfs_transformers_py` @ `b397988ed9e3e656475c1cf4417b84efdb95daf3`
  - `test/doc-builder` @ `6108e850ae1cf2f71bb0815a600bcd50c39abfa7`
  - `test/huggingface_doc_builder` @ `6108e850ae1cf2f71bb0815a600bcd50c39abfa7`
  - `test/huggingface_transformers` @ `44752c8dd99f3fb0da23006dc4fde4a07d9c417f`

### `ipfs_accelerate_py_operator`

- Path: `/home/barberb/lift_coding/external/ipfs_accelerate`
- Role: `runtime_accelerate_operator`
- HEAD: `ea11293bb996f052d620eae989f5377a956764b1`
- Tree: `ea6869d70e25c7bc8b80e6458c1a46b8c03f945f`
- Branch: `fix/implementation-auto-rescue-20260809`
- Remotes: `{"origin": "https://github.com/endomorphosis/ipfs_accelerate_py"}`
- Dirty: `True` (count=147)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `e01a7f4ec4ac2275f64a8f980ee614c10ebaae5e`
- Notes: Operator accelerate checkout with substantial dirty overlay; must not be reset or stash-dropped.
- Dirty paths:
  - ` M` `docs/architecture/agent_supervisor/PROGRAMS.md`
  - ` M` `ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/analysis/runtime_contract_evidence_compiler.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/control/default_doctor_factory.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/planning/adaptive_planner.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/planning/default_planner_factory.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/planning/formal_plan_validator.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/planning/symbolic_candidate_planner.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/proof/code_proof_obligations.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/proof/ir_adapters.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/proof/ir_registry.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_obligations.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/runtime/event_log.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_kernel.py`
  - ` M` `ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_provider_gate.py`
  - ` M` `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni158_embedding_tools.py`
  - ` M` `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni184_embedding_dispatch_compat.py`
  - ` M` `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
  - ` M` `ipfs_accelerate_py/mcp_server/server.py`
  - ` M` `ipfs_accelerate_py/mcp_server/tools/backend_management_tools/native_backend_management_tools.py`
  - ` M` `ipfs_accelerate_py/mcp_server/tools/embedding_tools/native_embedding_tools.py`
  - ` M` `ipfs_accelerate_py/mcp_server/tools/ipfs/native_ipfs_tools.py`
  - ` M` `ipfs_accelerate_py/mcp_server/tools/ipfs_cluster_tools/native_ipfs_cluster_tools.py`
  - ` M` `ipfs_accelerate_py/mcp_server/tools/workflow/native_workflow_tools.py`
  - ` M` `scripts/index_repository_contracts.py`
  - ` M` `test/api/test_agent_supervisor_configured_board_scheduler.py`
  - ` M` `test/api/test_agent_supervisor_deterministic_doctor_fixed_point.py`
  - ` M` `test/api/test_agent_supervisor_grok_quota_terra_gate.py`
  - ` M` `test/api/test_agent_supervisor_implementation_daemon_planner_doctor_hook.py`
  - ` M` `test/api/test_agent_supervisor_mcp_contract_catalog.py`
  - ` M` `test/api/test_agent_supervisor_pre_implementation_kernel.py`
  - `??` `data/agent_supervisor/.control-transaction.lock`
  - `??` `data/agent_supervisor/control-audit.jsonl`
  - `??` `docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md`
  - `??` `docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md`
  - `??` `docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.objectives.md`
  - `??` `docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md`
  - `??` `docs/architecture/mcplusplus_1_0_gap_closure.objectives.md`
  - `??` `docs/architecture/mcplusplus_1_0_gap_closure.todo.md`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/deterministic_desktop_expectations.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/deterministic_repair_analyzer_health.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/deterministic_repair_current_state.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/deterministic_repair_forest.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/hermetic_conformance.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/live_service_conformance.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_graph.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_identity.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_mismatch.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/mcp_live_observer.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/provider_surface_health.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/analysis/runtime_service_identity.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/autonomous_repair/`
  - `??` `ipfs_accelerate_py/agent_supervisor/evaluation/`
  - `??` `ipfs_accelerate_py/agent_supervisor/objectives/deterministic_repair_selection.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/objectives/repair_authority_projection.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/planning/deterministic_candidate_portfolio.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/planning/deterministic_failure_memory.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/planning/ir_logic_consumers.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/planning/ir_logic_hooks.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/planning/proof_carrying_repair_dag.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/planning/repair_resource_scheduler.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/proof/dcr_proof_cache.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/proof/ir_integration.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/proof/ir_logic_application.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/proof/ir_structural_application.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/proof/kernel_reconstruction.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/runtime/deterministic_repair_provider.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_doctor_bridge.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_ir_integration.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_ir_logic_applicator.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_rpr_admission.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_structural_ir_applicator.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_symbolic_planning.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/sca_symbolic_repair.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/todo_daemon/deterministic_repair_composition.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/todo_daemon/deterministic_repair_recovery.py`
  - `??` `ipfs_accelerate_py/agent_supervisor/validation/runtime_contract_evaluation.py`
  - `??` `ipfs_accelerate_py/mcp_server/package_mcp_interop.py`
  - `??` `ipfs_accelerate_py/mcp_server/surface_identity_bindings.py`
  - `??` `scripts/generate_mcplusplus_1_0_gap_closure_board.py`
  - `??` `scripts/validate_mcplusplus_1_0_gap_closure_board.py`
  - `??` `test/api/test_agent_supervisor_autonomous_repair_source_edit_gate.py`
  - `??` `test/api/test_agent_supervisor_datasets_logic_ir_integration.py`
  - `??` `test/api/test_agent_supervisor_dcr031_mcp_contract_obligations.py`
  - `??` `test/api/test_agent_supervisor_dcr032_multi_prover_router.py`
  - `??` `test/api/test_agent_supervisor_dcr033_kernel_reconstruction.py`
  - `??` `test/api/test_agent_supervisor_dcr042_protocol_repairs.py`
  - `??` `test/api/test_agent_supervisor_dcr045_ui_projection_repairs.py`
  - `??` `test/api/test_agent_supervisor_dcr_adversarial.py`
  - `??` `test/api/test_agent_supervisor_dcr_analyzer_health.py`
  - `??` `test/api/test_agent_supervisor_dcr_candidate_portfolio.py`
  - `??` `test/api/test_agent_supervisor_dcr_codegen_roundtrip.py`
  - `??` `test/api/test_agent_supervisor_dcr_current_evidence.py`
  - `??` `test/api/test_agent_supervisor_dcr_desktop_expectations.py`
  - `??` `test/api/test_agent_supervisor_dcr_doctor_composition.py`
  - `??` `test/api/test_agent_supervisor_dcr_doctor_diagnosis.py`
  - `??` `test/api/test_agent_supervisor_dcr_doctor_transform.py`
  - `??` `test/api/test_agent_supervisor_dcr_drift_monitor.py`
  - `??` `test/api/test_agent_supervisor_dcr_forest.py`
  - `??` `test/api/test_agent_supervisor_dcr_merge_provenance.py`
  - `??` `test/api/test_agent_supervisor_dcr_ordered_provider_fallback.py`
  - `??` `test/api/test_agent_supervisor_dcr_plan_dag.py`
  - `??` `test/api/test_agent_supervisor_dcr_planner_factory.py`
  - `??` `test/api/test_agent_supervisor_dcr_post_repair_validation.py`
  - `??` `test/api/test_agent_supervisor_dcr_proof_cache.py`
  - `??` `test/api/test_agent_supervisor_dcr_provider_surface_health.py`
  - `??` `test/api/test_agent_supervisor_dcr_recovery.py`
  - `??` `test/api/test_agent_supervisor_dcr_repair_admission.py`
  - `??` `test/api/test_agent_supervisor_dcr_replan_memory.py`
  - `??` `test/api/test_agent_supervisor_dcr_resource_scheduler.py`
  - `??` `test/api/test_agent_supervisor_dcr_runtime_service_identity.py`
  - `??` `test/api/test_agent_supervisor_dcr_security_operators.py`
  - `??` `test/api/test_agent_supervisor_dcr_selection_refill.py`
  - `??` `test/api/test_agent_supervisor_dcr_self_improvement.py`
  - `??` `test/api/test_agent_supervisor_dcr_transaction.py`
  - `??` `test/api/test_agent_supervisor_deterministic_contract_dispatch_repairs.py`
  - `??` `test/api/test_agent_supervisor_deterministic_contract_operator_repairs.py`
  - `??` `test/api/test_agent_supervisor_deterministic_contract_operators.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_artifacts.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_capabilities.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_contracts.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_daemon_composition.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_no_llm.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_provider.py`
  - `??` `test/api/test_agent_supervisor_deterministic_repair_root_ownership.py`
  - `??` `test/api/test_agent_supervisor_hermetic_conformance.py`
  - `??` `test/api/test_agent_supervisor_ir_logic_required_fail_closed.py`
  - `??` `test/api/test_agent_supervisor_mcp_contract_graph.py`
  - `??` `test/api/test_agent_supervisor_mcp_contract_identity.py`
  - `??` `test/api/test_agent_supervisor_mcp_contract_mismatch.py`
  - `??` `test/api/test_agent_supervisor_mcp_live_observer.py`
  - `??` `test/api/test_agent_supervisor_no_llm_runtime_barrier.py`
  - `??` `test/api/test_agent_supervisor_runtime_contract_evaluation.py`
  - `??` `test/api/test_agent_supervisor_runtime_integrity_repair_projection.py`
  - `??` `test/api/test_agent_supervisor_sca_doctor_bridge.py`
  - `??` `test/api/test_agent_supervisor_sca_rpr_admission.py`
  - `??` `test/api/test_agent_supervisor_swissknife_mcplusplus_live_services.py`
  - `??` `test/api/test_agent_supervisor_transport_repairs.py`
  - `??` `test/api/test_agent_supervisor_ui_ir_projection_integration.py`
  - `??` `test/api/test_mcplusplus_1_0_gap_closure_board.py`
- Recorded gitlinks (mode 160000):
  - `docs/fastmcp` @ `1d932cc778a24cc0bf46fc4baad8306d4fed9c4b`
  - `docs/mcp-python-sdk` @ `0da9a074d09267a927d72faa58c26d828f0f8edb`
  - `ipfs_accelerate_py/mcplusplus` @ `15c1816d6c63a2b11edd505704f6a04a9abc6167`
  - `ipfs_datasets_py` @ `a2f5400b7cb89c8481819379a1b7b9959fe81d45`
  - `ipfs_kit_py` @ `e164bb21c7a73b722a83aea7623e5677391bce54`
  - `ipfs_model_manager_py` @ `f6151d2113f42e75ea7d83a1b2362fc97e55e44d`
  - `ipfs_transformers_py` @ `b397988ed9e3e656475c1cf4417b84efdb95daf3`
  - `test/doc-builder` @ `6108e850ae1cf2f71bb0815a600bcd50c39abfa7`
  - `test/huggingface_doc_builder` @ `6108e850ae1cf2f71bb0815a600bcd50c39abfa7`
  - `test/huggingface_transformers` @ `44752c8dd99f3fb0da23006dc4fde4a07d9c417f`

### `ipfs_datasets_py`

- Path: `/home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/worktrees/workspace_4fbf5a625b5c_cf8849b0b2e2/ipfs_datasets_py`
- Role: `runtime_datasets_worktree`
- HEAD: `ac82107e246b30e35a2bbdcf75e01370d22350c6`
- Tree: `2b3d892dd1c31fb6b8a3eebdb88616d411c49a47`
- Branch: `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925-submodule-ipfs_datasets_py`
- Remotes: `{"origin": "https://github.com/endomorphosis/ipfs_datasets_py.git"}`
- Dirty: `False` (count=0)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `ac82107e246b30e35a2bbdcf75e01370d22350c6`
- Notes: Worktree-bound datasets gitlink used by this implementation workspace.
- Dirty paths: *(none)*

### `ipfs_datasets_py_operator`

- Path: `/home/barberb/lift_coding/external/ipfs_datasets`
- Role: `runtime_datasets_operator`
- HEAD: `ac82107e246b30e35a2bbdcf75e01370d22350c6`
- Tree: `2b3d892dd1c31fb6b8a3eebdb88616d411c49a47`
- Branch: `main`
- Remotes: `{"origin": "https://github.com/endomorphosis/ipfs_datasets_py"}`
- Dirty: `True` (count=8)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `ac82107e246b30e35a2bbdcf75e01370d22350c6`
- Notes: Operator datasets checkout; dirty overlay preserved.
- Dirty paths:
  - ` M` `.tools/ipfs_kit_py`
  - ` M` `ipfs_datasets_py/core_operations/logic_processor.py`
  - `??` `ipfs_datasets_py/logic/ui_ux_ir/`
  - `??` `ipfs_datasets_py/mcp_server/mcplusplus/p2p_libp2p_transport.py`
  - `??` `tests/fixtures/ui_ux_ir/`
  - `??` `tests/unit/logic/ui_ux_ir/`
  - `??` `tests/unit/mcp_server/test_mcplusplus_capability_truthfulness.py`
  - `??` `tests/unit/mcp_server/test_mcplusplus_p2p_tool_parity.py`

### `ipfs_kit_py`

- Path: `/home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/worktrees/workspace_4fbf5a625b5c_cf8849b0b2e2/ipfs_kit_py`
- Role: `runtime_kit_worktree`
- HEAD: `6196017ca3df016c7159dce43af60f2a0d96a9ae`
- Tree: `93070c709af29095fdff11f3e2698543449c08ef`
- Branch: `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925-submodule-ipfs_kit_py`
- Remotes: `{"origin": "https://github.com/endomorphosis/ipfs_kit_py.git"}`
- Dirty: `False` (count=0)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `6196017ca3df016c7159dce43af60f2a0d96a9ae`
- Notes: Worktree-bound kit gitlink used by this implementation workspace.
- Dirty paths: *(none)*

### `ipfs_kit_py_operator`

- Path: `/home/barberb/lift_coding/external/ipfs_kit`
- Role: `runtime_kit_operator`
- HEAD: `6196017ca3df016c7159dce43af60f2a0d96a9ae`
- Tree: `93070c709af29095fdff11f3e2698543449c08ef`
- Branch: `main`
- Remotes: `{"origin": "https://github.com/endomorphosis/ipfs_kit_py"}`
- Dirty: `False` (count=0)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `6196017ca3df016c7159dce43af60f2a0d96a9ae`
- Notes: Operator kit checkout; preserved as observed.
- Dirty paths: *(none)*

### `mcplusplus_gitlink`

- Path: `/home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/worktrees/workspace_4fbf5a625b5c_cf8849b0b2e2/ipfs_accelerate_py/mcplusplus`
- Role: `accelerate_nested_spec_submodule`
- HEAD: `6965f89f066769f3b3ac7b5f753b1a0044562570`
- Tree: `fd328606da5dae3314a87365987f5052180eb807`
- Branch: `implementation/mcpp-001-e94eb1589242-attempt-3-1786785925-submodule-ipfs_accelerate_py-mcplusplus`
- Remotes: `{"origin": "https://github.com/endomorphosis/Mcp-Plus-Plus.git"}`
- Dirty: `False` (count=0)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `6965f89f066769f3b3ac7b5f753b1a0044562570`
- Notes: Accelerate nested Mcp-Plus-Plus gitlink; compare to lift_coding/Mcp-Plus-Plus for drift.
- Dirty paths: *(none)*

### `swissknife`

- Path: `/home/barberb/lift_coding/swissknife`
- Role: `runtime_swissknife`
- HEAD: `afdbf885175fde34505ef05a2ea6aac5535ad03e`
- Tree: `6d73fab35179eac9edd87aa81bd7569414857d6b`
- Branch: `main`
- Remotes: `{"origin": "https://github.com/endomorphosis/swissknife", "upstream": "https://github.com/dnakov/anon-kode.git"}`
- Dirty: `True` (count=1)
- Program branch `codex/mcplusplus-1.0-gap-closure`: `present` at `afdbf885175fde34505ef05a2ea6aac5535ad03e`
- Notes: Discovered sibling checkout under lift_coding (not an accelerate submodule). Remotes discovered via git remote -v; not invented.
- Dirty paths:
  - ` M` `test-results/virtual-desktop-ipfs-mcp-orb/svd-132.json`

## Invariants

1. No uncommitted operator file was deleted, overwritten, stashed, or reset.
2. Forest JSON lists lift_coding, Mcp-Plus-Plus, accelerate, datasets, kit, and SwissKnife remotes and SHAs.
3. SwissKnife remote URLs were discovered with `git remote -v` from the bound checkout.
4. Program branch `codex/mcplusplus-1.0-gap-closure` is present or explicitly recorded as absent.
5. Implementation continues in isolated worktrees; operator dirty overlays remain authority for preservation.

