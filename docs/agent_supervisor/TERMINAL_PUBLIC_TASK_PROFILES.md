# Reviewed Terminal-Bench public task profiles

The full supervisor can retain immutable structured task inputs alongside the
program files it indexes. `terminal-public-task-profile@2` adds an explicit
`data_inputs` declaration to the existing exact-file profile. Every original
input remains in the signed manifest, planner scan, worker scope, and source
hash ledger. Only the separately declared program population supplies code
symbols and Source384 Python inference inputs.

The reviewed catalog currently covers these additional tasks:

| Task | Program inputs | Immutable task data | Created outputs |
| --- | --- | --- | --- |
| `tune-mjcf` | `eval.py` | `model_ref.xml` | `model.xml` |
| `llm-inference-batching-scheduler` | Three public Python modules | Two request JSONL files | Two plan JSONL files |
| `constraints-scheduling` | None | Three ICS calendars | `meeting_scheduled.ics` |

The catalog binds the public instruction, Dockerfile, task metadata, and each
reviewed environment COPY input to its SHA256 digest. A changed upstream file
requires review; the tool refuses to infer a replacement profile. It never
opens the benchmark verifier or solution directories.

Generate a profile for the existing benchmark launcher:

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_profile_catalog \
  --dataset /path/to/terminal-bench-2 --task tune-mjcf \
  --output /new/output/tune-mjcf-profile.json
```

Pass that file to `full_supervisor_benchmark prepare --task-profile ...`, with
the matching `--task`, selected provider, runtime archive, resource profile,
and index/checkpoint configuration. Each run needs a new output directory.
The catalog reports the task's original agent time allowance; an explicitly
selected supervisor resource profile can differ from that allowance and must
remain visible in comparisons.

Exercise native preparation and initial index hydration without a model call:

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_profile_catalog \
  --dataset /path/to/terminal-bench-2 --task tune-mjcf \
  --output /new/output/tune-mjcf-qualification --qualify
```

This constructs a disposable Git checkout from the exact reviewed COPY bytes,
authors the real signed declaration, and invokes the ordinary native index
preparation path. The receipt explicitly distinguishes that local check from
a live Docker inventory, full supervisor execution, checkpoint inference, and
an official benchmark reward. No candidate solution is generated.

## Data and output contracts

Version 2 accepts only explicitly declared JSON (`.json`), object-record JSONL
(`.jsonl`), XML (`.xml`), and calendar (`.ics`) data inputs. Names and media types
must agree. Data inputs cannot also be modified outputs. The fixed parser
replays the bound bytes; a Python file cannot be placed on an arbitrary ignore
list. Data files retain the existing 262144-byte complete planner scan limit.
All task files still share the existing population and manifest limits.

Structured input data uses explicit fetch references in the bounded worker
context. Each reference retains its path, SHA256, source CID, and media type;
the worker must read the exact repository bytes before deriving results. All
original bytes remain in the source block store and complete manifest. This
avoids inserting large data tables into the prompt while retaining the
32768-byte worker context limit. Currentness checks still read and verify the
whole declared population. Program changes can produce a fresh context;
immutable task data or generated support changes are refused.
Fetch references establish availability and an obligation to read the data;
they do not attest that the model actually read those bytes. Publication refresh
preserves the same references and rechecks immutable data before admitting a
successor context.

The generated isolated structural smoke check preserves the file and total
output byte bounds, rejects symlinks/nonregular outputs, and checks Python
syntax and declared structured formats without executing candidate code.
JSON rejects duplicate keys and nonfinite numbers; JSONL requires nonempty
object records. XML rejects document types and entity declarations. Calendar
validation checks framing and required headers, not scheduling constraints.
Version 1 producer bytes and acceptance text remain unchanged.

These checks establish format and provenance, not a correct task solution.
For example, XML parsing cannot establish MuJoCo numerical equivalence or a
speedup, and JSONL parsing cannot establish a batching plan's coverage or cost.
Those remain obligations for separately admitted behavioral checks and the
official benchmark verifier.

## Symbolic and checkpoint boundaries

Source384 receipts account separately for program inputs, the three generated
harness support files, and immutable task data. The canonical version 2
profile is hash-bound to the signed source ledger so historical scope replay
can reproduce the distinction. Data bytes stay in the complete ledger and
cannot change unnoticed during inference or subsequent currentness checks.

The Doctor currently returns `doctor_task_data_contract_unavailable` for a
mixed task until an independent contract covers those data semantics. It
does not turn a successful format parse into program proof coverage. The
symbolic capability report preserves this residual and counts task data
separately from generated harness support.

`constraints-scheduling` has an honest empty code population: zero code
symbols and zero embedding calls. Selecting Source384 still abstains before
loading the checkpoint because no declared Python program exists. This
profile does not claim that the full Source384-selected arm now supports
every empty-source task.

Build/install effects, persistent services, Git state transitions, inputs
outside `/app`, large binary data, and other unreviewed formats still require
additional independently bounded profiles. The three catalog entries alone
do not qualify the full Terminal-Bench suite or establish token savings.

## Local qualification on 2026-10-05

The reviewed public COPY inputs passed actual signed preparation and initial
index hydration on this machine, without model-provider calls:

| Task | Indexed symbols | Preparation and hydration | Separate Source384 check |
| --- | ---: | ---: | --- |
| `tune-mjcf` | 4 | 4.45 seconds | Checkpoint consumed; one model load; 11.33 seconds |
| `llm-inference-batching-scheduler` | 21 | 7.08 seconds | Checkpoint consumed; one model load; 13.92 seconds |
| `constraints-scheduling` | 0 | 2.29 seconds | Explicit abstention before checkpoint loading |

Both neural checks used checkpoint
`2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5`,
performed zero training steps, and passed current and historical scope replay.
These timings are individual local development observations, not comparative
performance estimates or benchmark rewards.

The batching scheduler retains all 99183 JSONL bytes as exact data inputs.
Its worker context is 32751 bytes. All 41 semantic capsules remain in the full
index, but none fit in that task's bounded worker projection; omission counts
and mandatory raw-source fetch references make that limit explicit. The
earlier preparation failures identified two repaired boundaries: fenced public
instructions needed a separate signed instruction envelope, and structured
data needed explicit fetch references instead of eager prompt inclusion.

The [expansion evidence](evidence/terminal-expansion-20261005/README.md) records
the qualified source revisions, regression checks and separate live Grok trial
outcomes. The [ready-root context qualification](evidence/supervisor-task-context-20261005/README.md)
adds an advisory route for independently ready version 3 tasks. Native multi-task
execution, dependent source successors and parallel worker execution remain
guarded.
