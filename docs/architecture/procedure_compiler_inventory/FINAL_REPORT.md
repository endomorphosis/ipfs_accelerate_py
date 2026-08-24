# Procedure compiler current-tree qualification report

Disposition: **not qualified; not eligible for promotion or release**.

This report records observations for commit `4e733c2aa33da459670131880b6a2fd85cf6c48c`, tree `5a13dc008a28fcc180587e52df1900135f119249`, on branch `implementation/pcpc-031-8e98deb8e2a7-attempt-1-1787541849`. The observed checkout was clean apart from ignored state. These report files were written afterwards and are not evidence or qualification receipts for that tree.

The starting baseline was commit `bbf7f68799072c2b81f7d96eac91f2df3c4b3952`, tree `a698da9e4b54e2929adacb613bc61ba3e72eed58`. The initial source rollback target remains that exact commit/tree; no release or rollback operation was authorized or performed.

## Current-tree inventory

The current tree adds the 25-file `ipfs_accelerate_py/agent_supervisor/procedure_compiler` package (39,915 added lines), a three-file benchmark corpus, and 32 focused test modules (60 added files and 58,901 added lines across those areas). The machine report contains the exact 25-file package inventory. Its principal symbols include `ProofCarryingProcedureCompiler`, `ProcedureInterpreter`, `ProcedureCegis`, `ProcedureCertificateVerifier`, `ProcedureRegistry`, `ProcedurePlannerAdapter`, `ProcedurePromotionGate`, `ProcedureTransferGate`, `ProcedureDriftMonitor`, `ProcedureRecoveryPlanner`, `ProcedureRollbackService`, `ProcedureControlServiceAdapter`, and `ProcedureGuidedRepairAdapter`.

The declared contracts include `ProcedureCompilerContracts@1`, `ProcedureIR@1`, `ProcedureInterpreter@1`, `RepositoryWorldModel@1`, `TransitionModel@1`, `ProcedureControlService@1`, `ProcedureCLI@1`, `ProcedureMCPSurface@1`, and `ProcedureCompilerReleaseReceipt`.

The frozen synthetic benchmark is structurally valid: 138 cases across 23 task families and six disjoint partitions (23 cases each), with corpus SHA-256 `2f22bef626d0ab2257953a97f96771e9915f05264faec44b20af5e5bd5221618`. This is fixture-structure evidence only; it is not live performance, safety, transfer, or release evidence.

Synthesized procedures: 0 observed; promoted procedures: 0 observed; rejected procedures: 0 observed; admitted boundary/counterexample ledger entries: 0 observed. No admitted current-tree proof, adversarial, held-out, shadow, materialization, or authorized release-decision receipt was supplied or found. Fixture-created artifacts and fixture gate outcomes are not reported as live procedures or decisions.

## Qualification execution

| Producer | Result | Evidence |
|---|---|---|
| `python scripts/validate_agent_supervisor_procedure_compiler_board.py --check-all` | Failed | `scheduler_schema`: configured external isolation is unavailable because its Docker endpoint is not an admitted local socket. |
| `python -m pytest -q test/api/procedure_compiler` | Failed, 609 passed / 12 failed / 1 skipped | Full 622-item run completed in 18.60 seconds. Failures comprise the board scheduler-isolation check; two current-tree inventory drift checks for `control_plane.py`; one DuckLake checkpoint-sidecar projection check; and eight program-launcher isolated-Python checks. |

The failing inventory checks report `control_plane.py` expected blob `f06b091ff4e5a0a3ea09da81c866845bf064af9b`, observed blob `01bdede85b470ca7fbf8d3274e3e32875b8553db`. The program-launcher failures instead report `python_environment_invalid`: the active Python user site is unavailable for isolated qualification. No independent proof producer ran. No test or proof omission is waived. There is no successful post-merge execution of the 622-item target.

## Metrics, safety, and transfer

No admitted provider, token, cost, retry, model-call, human-intervention, amortization, idle-stability, or qualified autonomous-meta-controller comparison-baseline receipts bound to the ending tree were supplied or found. Consequently, all numeric promotion gates are unevaluable. The required cost categories (`match`, `synthesis`, `shadow`, `hole_filling`, `validation`, `rollback`, and `review`) have no observed values.

Likewise, there is no admitted live safety-gate population, unsafe-match ledger, held-out transfer result, or cross-repository transfer result. Do not treat implementation constants or fixture tests as a live zero-unsafe-transfer result. Post-merge qualification is failed/incomplete.

## Blocking residual gaps and unavailable features

- `PCPC-031-SCHEDULER-ISOLATION`: blocking — the scheduler’s external-isolation Docker endpoint is not admitted.
- `PCPC-031-PREREQUISITE-DRIFT`: blocking — the sealed inventory rejects the current `control_plane.py` blob.
- `PCPC-031-LIVE-EVIDENCE`: blocking — current-tree materialization, proof, adversarial, held-out, shadow, and release-decision receipts are absent.
- `PCPC-031-METRIC-BASELINE`: blocking — the qualified autonomous-meta-controller comparison baseline and complete measurement denominators are absent.
- `PCPC-031-ISOLATED-PYTHON`: blocking — the complete target reports an unavailable active Python user site for isolated qualification; this is not evidence that the program launcher is usable in the authoritative environment.
- `PCPC-031-IDLE-STABILITY`: blocking — no admitted idle-stability observation shows the absence of task, model, or mutation churn.
- `PCPC-031-ADAPTIVE-PLANNER`: unavailable — `HAMMER_TRACE_SCHEMA` is undefined in `multi_prover_router.py`.
- `PCPC-031-TRANSFER`: unavailable — dynamic cross-lane transfer awaits a schema-compatible Quack coordination adapter and recovery qualification.
- `PCPC-031-LIVE-QUACK`: unavailable — live launch requires admitted owner/provider/candidate-validation boundaries; direct writer serving and host-mode Quack are not fallbacks.

Therefore AdaptivePlanner procedure integration, dynamic cross-lane transfer, live Quack/writer-serving fallback, autonomous-baseline-dependent savings claims, procedure promotion, and release remain unavailable. The tree may only be described as containing implementation and fixture coverage; it is not qualified for promotion, release, autonomous execution, or performance claims.

## Rollback

The declared initial rollback target is commit `bbf7f68799072c2b81f7d96eac91f2df3c4b3952`, tree `a698da9e4b54e2929adacb613bc61ba3e72eed58`. Any source rollback requires a reviewed Git operation and must not rewrite unrelated operator work.
