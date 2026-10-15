# Shared successor receiver milestone

An additive [paired receiver](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_shared_successor_receiving.py) and 35 native test cases have been added for qualification. The [bounded qualification](codebase_shared_successor_receiver_qualification_v1.md) passed 120 new and unchanged receiving controls. Full-300 and fresh-process receiver qualification, signed admission, artifact-store and worker integration remain proposed. The receiver holds its admitted source and model owners across application callbacks and exposes one closing fence. This is the prerequisite for integrating shared completions into signed successor workers. Existing signed profiles and production acceptance rows retain their current status.

## Current compatibility boundaries

The [shared scanner](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_shared_scan.py#L582) validates completion bytes and closes its native observation within the receiving call. It does not expose the paired scope needed to perform application work before the final native close. The [existing successor paired receiver](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_successor_receiving.py#L185) provides that lifecycle for `CodebaseScanResumeCompletion` only.

Three existing boundaries deliberately reject shared records:

- [Inventory native admission](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_evidence_admission.py#L43) and [successor native admission](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_successor_dispatch_admission.py#L48) require the legacy completion type.
- [Public inventory replay](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_evidence_worker_context.py#L53) accepts the legacy advisory schema. Its declaration validator reconstructs completion version 1 and requires a DAG-JSON completion CID.
- [Preview worker admission](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_preview_successor_worker_admission.py#L114) delegates those native scopes. Passing a structurally similar shared record cannot supply its required closing capability.

Add separate APIs and explicit schema branches. Keep existing exact-type checks, wire formats and rejection behavior.

## Receiver contract

The additive receiver uses the ordinary root version 1, the shared completion version 3, the selected direct model child and its source delta. Reuse the [shared record validators](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_shared_chunks.py#L21) and current source/model observations through explicitly reviewed internal interfaces.

The receiver must:

1. Acquire the existing resource envelope before reading retained scan bodies. Check cancellation, exact native owners, implementation pins, root membership, source transition, direct model ancestry and complete frozen model state.
2. Walk every completion page and numerical chunk in order. Reconstruct each full native target and validate source bytes, AST provenance, authored contracts, coverage and row mapping. Retain one chunk body at a time; make no numerical forward or fitting call.
3. Yield a callable bound to the issuing process, native thread, owner identities, admitted lease and immutable input bytes. A saved receipt or serialized callable cannot recreate this capability.
4. Require the caller to finish artifact reads, signature checks, task compilation and application observers before consuming the callable exactly once. The callable independently closes source, SQL, model, registry and history observations and checks retained root/page/chunk/completion bytes. Preserve the current ordering of extension callbacks before native close and direct byte checks afterward.
5. Require successful close before normal scope exit. Reuse, foreign-thread use, expired ownership, cancellation, closing failure or unsuccessful context exit must reject the operation. No successful closing result survives into another receiving boundary.

Resource accounting, actual sampled RSS and process cleanup are distinct. Preserve existing limits and owned cleanup custody; qualify the new receiver's retention and memory observations before admitting it to an executing profile.

## Artifact identity

The shared profile has an ordinary DAG-JSON root version 1, RAW CIDv1 numerical chunks version 2, RAW pages and completions version 3, and advisory references version 3. Validate the exact profile alongside each schema. Match canonical bytes, SHA-256, byte count, member order and every referenced identity without rewriting any artifact.

Numerical chunks contain finite float values. The [compact artifact design](codebase_compact_successor_artifact_contract_design.md) currently proposes float-free supervisor artifacts with DAG-JSON identities. Its future resolver therefore needs a registered native scan role that preserves RAW identities and applies the bounded finite numerical decoder, or a DAG-JSON descriptor referencing the original RAW objects. Keep generic signed-wire float restrictions intact. A hash or lifecycle record does not establish numerical execution, currentness or proof authority.

## Signed integration boundaries

After the paired receiver passes, add explicit shared inventory and successor context decoders. Join advisory version 3 to the complete head, membership, selected model, ordinary root, completion, pages and coverage. Preserve the original signed version 6 parent admission and its identities. When its legacy completion differs from a current shared completion, retain both identities and bind the shared selection separately to the same source and model; do not silently replace the parent's declaration.

The next compact profile remains the proposed manifest version 8, context version 2, pending version 4, and planning receipt, receipt reference and public instruction version 6. Verify each actual new signature and schema. Reusing parent comparison rules must not turn an unsigned synthetic version 7 object into an accepted new admission.

Future integration must cover these concrete boundaries:

| Boundary | Required change |
| --- | --- |
| [Local manifest validation and pending construction](../../ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py#L380) | Select explicit new decoders; replace repeated full child bodies with a reference to one complete signed `{manifest, graph, receipt}` artifact. Retain exact task specifications, dependencies, acceptance checks and outputs. |
| [Staged Intent installation](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_preview_successor_worker_admission.py#L337) | Publish and durably pin the complete closure before staging. Consume the native close and exit its scope inside the writer transaction so failure rolls back the complete task population. |
| [Public instruction replay](../../ipfs_accelerate_py/agent_supervisor/runtime/router_public_instruction.py#L273) | Resolve only registered roles from an approved immutable offline mirror; replay signatures, complete source/task bindings and allocated-worktree checks. |
| [Execution reservation and receiving](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_execution.py#L1263) | Add an explicit compact execution profile and paired receiver selection at reservation, construction and START. Preserve callback custody and native close before Popen. |
| [Owner completion service](../../ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py#L382) | Support the new profile explicitly and retain exact validation, STOP, isolated UID cleanup and reservation lifetime checks. |

The future artifact store must enforce owner custody, held directory descriptors, canonical bounded reads, immutable durable publication and conservative pin retention. Initial implementation has no garbage collection. Missing offline closure members reject before reservation or START. The existing 262,144-byte Intent task-body limit remains unchanged; the proposed store bounds require measurement before qualification.

## Success and refusal criteria

The receiver milestone passes only with genuine native controls for both optimized and reference shared scans, unchanged legacy receiving, cold reopen and complete resumed populations. For the existing full 300-member fixture, preserve all 294 eligible numerical rows, 588 projection records and the six explicit remaining dispositions. Receiving must perform zero fitting and zero numerical forward calls.

Refuse missing, corrupt, reordered or foreign pages/chunks; mixed schemas or codecs; boolean aliases; wrong source, AST, target, model or row bindings; altered private state; source or SQL changes during application callbacks; late cancellation; and owner or implementation replacement. Controls must show that the close is consumed once and that failed close or scope exit prevents acceptance and rolls back staged installation. Retain independent control receipts and a finite ordinary-byte reader with explicit execution and proof limits.

Actual full 300-source signed installation, dispatch, START, authored validation, completion and STOP cleanup are a later qualification. A worker edit also requires a newly published source generation, explicit compatible model selection or training, fresh scans and independently checked evidence before successor acceptance. Receiver success alone does not supply those results.

## Separate head qualification

The current CPU structural fixture has 53 compiler-feature columns and latent width 8. It is distinct from a semantic 8D representation. The [four-head integration note](codebase_larger_head_native_owner_integration_note.md) separates source representation width from a trained CodebaseIR head.

For 384D, 768D and 4096D, require a declared source task, exact spans and tokenizer policy, pinned encoder and numerical implementation, full parameter and optimizer state, immutable lineage, fixed train/tune/canary roles and native task correspondence. Qualify complete CPU/default/reference/resumed scans and signed worker integration for each selected family. CUDA needs its own actual encoder/head, cancellation, memory and equivalence qualification. Existing GTE or Leanstral source evidence and weight-only sidecars cannot satisfy these gates. Keep structural limits and decoder behavior unchanged.

Formal proof obligations discharge through independently accepted checker evidence bound to the exact source generation and policy. Learned vectors and similarity scores remain planning inputs. All [repository proof index production acceptance rows](../architecture/repository_proof_index_and_codebase_ir.todo.md#L608) remain open.
