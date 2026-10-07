# Reviewed finite data contracts

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Developers extending symbolic task coverage

**Sources:** `planning/intent_data_transform.py`, `runtime/doctor_data_contract.py`,
and `ipfs_datasets_py.logic.software_contracts.finite_record_projection`

**Last verified:** 2026-10-07

The supervisor can create one NDJSON file by copying or renaming one field in
every record of a signed input. The reviewed declaration drives symbolic
planning and Doctor dispatch. Shared synthesis and independent checking belong
to `ipfs_datasets_py`; the supervisor consumes their results through its
existing staged candidate worker.

## Declaring the operation

Use `build_intent_requirement_contract(..., symbolic_operations=operations,
reviewed_data_transform=selector)` from
`ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage`. This produces
`intent-plan-requirement-contract@4`. The existing requirement ledger must bind
the exact immutable instruction. The operation must bind its grounded output
and validation to the independently authored task manifest.

The selector has these exact fields:

```json
{
  "schema": "reviewed-ndjson-data-transform@1",
  "review_ref": "review:field-rename@1",
  "operation_id": "operation:data",
  "input_path": "input.jsonl",
  "output_path": "output.jsonl",
  "mode": "rename",
  "source_field": "name",
  "target_field": "label",
  "correspondence_policy": "ordered-record-field-correspondence@1",
  "validation_key": "public-record-correspondence",
  "semantic_alignment_verified": false,
  "proof_authority": false,
  "execution_authority": false,
  "publication_authority": false,
  "completion_authority": false
}
```

This is a reviewed input, not an automatically verified translation of prose.
The operation ID and validation key must match the sole symbolic operation;
the independently signed task must own exactly one `create` output of media
type `application/x-ndjson`. Input and output paths must be distinct `.jsonl`
paths in the task scope. The input must be signed and immutable; the output
must be absent. Output parent directories must already exist in the Git
baseline. Field identifiers are distinct ASCII identifiers of at most 64
characters.

The full Terminal-Bench harness accepts the contract through its existing
`--intent-requirement-contract` option. A data task also needs a compatible
public task profile; the contract does not expand that profile's inputs or
outputs. Public profile `@2` still classifies files and supplies structural
checks only. Without the reviewed transformation selector it authorizes no
record transformation. Intent contract `@3` remains the separate header route.

## Candidate and evidence lifecycle

`prepare_terminal_doctor_dispatch` recognizes signed `@4` requirements before
the Python repair route. Direct native consumers can call
`prepare_ndjson_contract_candidate(repository=..., admission=..., intent=...,
task_cid=..., state=...)` while the Intent database is file-backed. The state
directory must be fresh and outside the repository.

The workflow revalidates admission, task revision, complete signed source
bindings and the declared operation. It captures exact input bytes, synthesizes
a candidate and independently checks record correspondence. It hydrates a
DuckDB/DuckLake world record and dependency index, then freshly checks source,
task, receipt and complete wrapper bindings before exposing immutable candidate
bytes. Index rows never replace fresh checks. Canonical source and task status
remain unchanged during preparation.

The worker creates the declared file in its allocated worktree. Native leases,
validation, proposal checks, publication and completion still run. Existing
candidate schema `@1` uses the legacy field `proof_receipt_id` for the bound
finite-check receipt; its `proof_scope` explicitly describes the evidence.
The workflow records `evidence_kind=finite_record_check`, `kernel_proved=false`
and no active kernel-proof receipts. This is an exact check of finite input
and output records, not a Lean theorem or a proof that the declaration captures
every meaning of the instruction.

The shared profile admits nonempty scalar records containing null, Boolean,
safe integers and Unicode strings. It preserves order, multiplicity, types and
every unrelated field. Both modes reject missing source fields, existing target
fields, duplicate JSON keys, floats and nested containers. Fixed bounds include
1,000,000 bytes per file, 4,096 records and 128 fields per record. Unsupported
inputs retain a residual. Incorrect synthesized bytes or stale/forged bindings
are hard refusals.

## Validation and remaining coverage

The authored fixtures exercise copy and rename, strict type correspondence,
admission and public replay, source/task/index drift, staged creation and a real
signed START → validation → publication → completion → STOP lifecycle with
model calls forbidden. Run the supervisor checks with the matching datasets
checkout on `PYTHONPATH`:

```sh
python -m pytest -q test/api/test_intent_data_transform.py \
  test/api/test_doctor_data_contract.py test/api/test_doctor_data_dispatch.py \
  test/integration/test_native_data_contract_repair.py
```

The native lifecycle test requires the installed Quack transport. Existing
header/candidate-runner and intent-planning regression suites remain relevant.
The shared module has its own adversarial test suite in
`tests/unit/logic/software_contracts/test_finite_record_projection.py`.

This qualification uses authored data and establishes no new Terminal-Bench
score, model-token saving ratio or whole-suite completion result. Filtering,
aggregation, scheduling, nested records, directory creation, multiple outputs
and automatic prose-to-contract verification remain outside this operator.
