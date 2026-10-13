# Signed Terminal semantic source selection

The opt-in `terminal-public-task-profile@4` declaration selects a bounded input
population for semantic context and retrieval while retaining the complete
signed public input inventory. Its `input_paths` field still lists every original
tracked input. Generated task scope, output effects, structural validation, and
native manifest source custody continue to cover the complete public contract.

This extends the [repository source nomination](repository_source_scope.md) and
[native task-owner handoff](current_source_task_admission.md#bind-a-pinned-historical-nomination)
with a reviewed declaration at the Terminal preparation boundary. It supplies
bounded planning coverage. Native source capabilities, allocations, current
source admission, and actual coding-child adoption remain separate checks.

## Declare a selected population

The profile is flat, with exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema` | `terminal-public-task-profile@4` |
| `instruction_sha256` | Digest of the normalized public instruction |
| `input_paths` | Complete original tracked input population |
| `outputs` | Existing exact output path, effect, and media-type declarations |
| `semantic_scope` | Closed selection using `terminal-public-task-semantic-scope@1` |

The nested scope has exactly `schema`, `input_paths`, `required_input_paths`, and
`program_paths`. Its three ordinary path lists must already be sorted, unique,
and canonical. They cannot contain generated supervisor supports, aliases,
reserved paths, or inputs absent from the complete profile. The producer refuses
invalid selections rather than truncating them or silently adding ordinary
inputs.

For a repository whose complete original tracked population is `other.py` and
`source.py`, this declaration selects only `source.py` for context:

```python
from pathlib import Path
from benchmarks.agent_supervisor.container_coding import (
    terminal_indexed_preparation as prep,
    terminal_task_profile as profiles,
)

repository = Path("/path/to/task-repository").resolve(strict=True)
instruction = Path("/path/to/public-instruction.md").resolve(strict=True)
state = Path("/path/to/new-state-outside-repository")

declared = {
    "schema": profiles.SCOPED_SCHEMA,
    "instruction_sha256": profiles.instruction_sha256(
        instruction.read_text(encoding="utf-8")
    ),
    "input_paths": ["other.py", "source.py"],
    "outputs": [
        {"path": "source.py", "effect": "modify", "media_type": "text/x-python"}
    ],
    "semantic_scope": {
        "schema": profiles.SEMANTIC_SCOPE_SCHEMA,
        "input_paths": ["source.py"],
        "required_input_paths": ["source.py"],
        "program_paths": ["source.py"],
    },
}
prepared = prep.prepare(
    repository=repository,
    instruction=instruction,
    state=state,
    task_profile=declared,
    disable_intent_autoencoder=True,
)
```

The state directory must be new and outside the repository. Existing preparation
requirements still apply, including exact tracked input coverage, absent
generated support paths, and the original native source/output bounds. Disabling
the optional instruction advisor in this example does not change the signed
scope declaration or replace any existing model assets.

## Roles and bounds

Three generated supports are implicit in every selected context:

- `.supervisor-instruction.md`
- `.supervisor-public-smoke.py`
- `.supervisor-task-profile.json`

All three are required raw inputs and count toward the same selected population
bound. The profile permits 1 to 61 ordinary selected inputs, giving at most 64
total files. Their combined original source bytes, including all three generated
supports, must be at most 1,000,000. This uses original per-path UTF-8 bytes,
before any compact view or projection. The selected byte bound is independently
replayed from actual signed files; a small declaration alone does not establish
it. Existing complete-inventory bounds remain in force separately.

`program_paths` must be nonempty and name selected `.py` inputs. Every selected
nonprogram input must be required raw context. Every modified output must be both
selected and required; a modified `.py` output must also be a program input.
`required_input_paths` may be empty for a create-only task whose selected inputs
are all program files. Generated supports remain required in that case.

The shared [profile producer](../../ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py)
exposes separate helpers:

| Helper | `@4` result |
| --- | --- |
| `task_profile_worker_inputs(profile)` | Three supports plus **all** original `input_paths` |
| `task_profile_semantic_inputs(profile)` | Three supports plus selected ordinary inputs |
| `task_profile_semantic_required_paths(profile)` | Three supports plus selected required ordinary inputs |
| `task_profile_index_paths(profile)` | Selected program inputs |

Legacy profiles keep their canonical bytes, smoke generation, and task
declarations. The new context helpers fall back to the complete legacy input
population and generated supports. This version does not combine selected scope
with `data_inputs`, multi-task ownership, or nested task profiles.

## Signed custody and replay

Preparation records the original input population, writes the canonical profile
and fixed supports, and signs the complete manifest source bindings. The owner
signature binds the profile through its exact source hash. The declaration
contains no post-commit inventory CID, source digest map, or scope receipt that
would create a profile-to-inventory identity cycle.

The [scope replay helper](../../ipfs_accelerate_py/agent_supervisor/runtime/terminal_semantic_scope.py)
verifies the complete signed population before reading the selected originals.
It checks generated support bytes, selected roles, raw source identities, and
the combined byte ceiling. Its receipt binds the selection to the retained
manifest and records full, selected, and omitted counts plus a frontier identity.
Omitted inputs remain present in the signed manifest. Editing an omitted input
invalidates complete source replay just as editing a selected input does.

Receipts are observations: their historical/currentness and authority flags do
not grant a source capability, proof, execution permission, completion, or
permission to omit task obligations. The native coding broker's separate
32-path limit is unchanged by the semantic producer's 64-file limit.

The consumer paths perform complementary checks:

- Initial preparation builds selected retrieval and semantic artifacts while
  capturing a real empty Intent owner. Final complete-source replay precedes
  publication of the initial descriptor and result.
- Cold loading replays the signed selection and compares the descriptor,
  retrieval population, semantic manifest, required originals, program roles,
  and source identities. A rehashed unsigned descriptor cannot choose a new
  population.
- Admitted context can reuse those exact source artifacts and recapture the
  admitted native task world. Direct admitted preparation without initial
  artifacts derives the same selected population.
- Warm owner rebinding rechecks complete signed source custody and the selected
  payload around native world capture. Worker context compilation retains the
  selected semantic, retrieval, and world evidence.
- Historical capsule audit checks retained declaration and evidence identities
  without requiring a surviving checkout. It does not independently re-read
  every selected source length, verify a live source state, or establish fresh
  signature or allocation authority. Its explicit observation flags preserve
  those limits.

## Decoder and Doctor boundaries

Selected scope currently refuses context options that assume a complete
population for local autoencoder training, frozen security decoder advice,
Source384 context, numerical formalization, or learned numerical selection.
Refusal occurs before creating the selected context outputs. These lanes need
separate qualification of their selected populations, decoder roles, and
coverage claims before combining them with this opt-in declaration.

Selected semantic diagnostics remain planning evidence. The complete Doctor
repair gate explicitly abstains for this scope, with
`signed_subset_requires_separate_doctor_repair_scope`; it does not treat subset
diagnostics as complete repair eligibility. The complete Doctor input ledger and
source partition retain their existing meaning. A selected nonempty program
population does not replace the independent complete-population emptiness
contract. Selected code without eligible vector symbols refuses instead of
claiming that the complete repository is empty.

The implementation does not replace checkpoints, embeddings, per-family IR
inventories, or model registration. Separate 8D, 384D, and 768D decoder paths and
their existing assets remain independent of this context-selection declaration.

## Qualification and next steps

The [scoped-profile tests](../../benchmarks/agent_supervisor/container_coding/test_terminal_scoped_task_profile.py)
exercise the declaration, complete-versus-selected helpers, role constraints,
canonical path refusal, file/profile bounds, and legacy byte compatibility.
The [signed-scope integration tests](../../benchmarks/agent_supervisor/container_coding/test_terminal_signed_semantic_scope.py)
exercise real Git inputs and owner signatures, selected context construction,
reuse and direct context paths, current-source drift, exact generated-support
byte accounting, native owner rebinding, compiled worker evidence, and historical
audit. The [qualification bundle](evidence/signed-terminal-source-scope-20261013/README.md)
records the executed checks and their exact revision and dependency boundaries.

The [comprehensive plan](reversible_codebase_views_plan.md) retains the next
integration work: bind independently pinned repository nominations to reviewed
declarations, qualify scope expansion and native coding-child adoption, and
measure complete sessions with matched workloads. This implementation makes no
token-savings, benchmark-success, model-quality, or formal-proof claim.
