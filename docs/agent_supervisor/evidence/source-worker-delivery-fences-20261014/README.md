# Broker source-delivery closing checks, 2026-10-14

The native source worker broker now releases SERVER for both broker-owned
selected-source CAS replays. After each replay it completes the native task and
allocation fence outside SERVER, checks the protected endpoint, and closes
original owner, task, allocation, nomination and source custody with detached
checks. Both complete source populations and the selected original bytes are
checked before the backend call and before returning a public result. The
existing launch path shares the same closing section and keeps its original
check order.

The implementation is in [the existing broker](../../../../ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py).
The [maintained channel guide](../../source_lookup_worker.md#close-source-delivery-after-broker-replay)
describes the consumer boundary and accounting contract. The
[new regression suite](../../../../test/api/test_native_source_worker_delivery.py)
has 22 cases: two stable callback paths and ten custody mutations at each of
the pre-backend and post-backend replays. Retained-CAS callbacks enter the real
native task/allocation checks only after confirming SERVER has been released.
Mutations cover selected and omitted originals in both roots, native task body,
daemon/proxy grant revocation, released claim, descriptor bytes and socket mode.
Pre-backend refusal incurs no source RPC. Post-backend refusal preserves the
validated native response and wire byte charge without returning a success.

## Retained controls and qualification

The [unpatched control](baseline-01/commands.json) runs one new case against
published source `09ad10bd0e6694db7b0570614ab289184a0713c5`. Its deliberate
failure reports `broker-selected replay must release SERVER` before attempting
native reentry; it demonstrates the ordering error without deadlocking the
fixture. It completed in 26.873 seconds with one failed case, no setup error and
no skip. This control is separate from passing test counts.

The [first wider attempt](qualification-01/commands.json) collected 166 cases
but stopped at its first fixture setup with `AF_UNIX path too long`. The
temporary root was too long for the native Unix socket pathname. Its raw failed
record, logs, JUnit and original runner/source snapshots remain unchanged. The
runner now uses a shorter unique private temporary root; production source and
tests were not changed for that correction.

The [second full attempt](qualification-02/commands.json) passed **166 distinct
tests across the exact [eight-suite roster](targets.json)**, including all 22
new delivery cases, with zero failures, errors or skips. Its 933.237 seconds
measure authored-test execution rather than production latency. All listed
source bytes and fixed execution HEAD/index guards stayed unchanged. Counts
from the unpatched control or the failed attempt are not added.
The [source inventory](source-inventory.json) lists 273 dependencies: 222
accelerator and 51 datasets files. This is an explicit listed population,
rather than a complete import or toolchain closure. The actual datasets checkout
is `4b50ae4aac65da9166bd3b24608f32ba7e9fb23d`, matching the retained parent
publication pin; kit is `40c98c71b95be750509ffdb21e7e5494d91b4e79`.
The child base's embedded datasets gitlink is `904f885c180eb9dfcdef80199f708125428d3859`;
it is not relabeled as the checkout used for these tests.

Each attempt retains one pytest invocation, exact argv/environment updates,
interpreter/distribution/platform observations, raw outputs and changed-source
snapshots. The explicit AST-seal plugin runs with automatic plugin loading
disabled. Tests use private runtime stores and offline settings; inherited
private-owner, credential and metadata-index environment names are scrubbed.
No environment credential values or runtime databases are published. Source
bytes and fixed execution HEAD/index guards are checked before and after.
Shared reference changes are separate observations with no writer attribution.

The [architecture review](architecture-review.json) and
[parallel survey](parallel-review.json) bound the implementation and overlap
review. The original local documentation receipt is `documentation-checks.json`; its
exclusive raw log is `documentation-workflow.log`. The release checker validates
the retained attempts and executes the four exact reviewed workflow blocks.
The final `manifest.json` seals this phase without a self-hash cycle.

## Integration with concurrent world-packet audit

The initial qualified branch is `7883e693c5d90fbaa445a444eaeebc90789d9db1`.
Its main push encountered upstream world-packet audit
`8f5fa23317bc9609ea8cf7febf2d01c7b821f552`. Merge
`f58aa231805f2546d4f5cf650ef0bf3900d25540` preserves both contributions and
both plan updates. The [integration review](incoming-world-review.json) records
the four changed listed accelerator files and unchanged broker/new tests. The
first full passing run and its four-block documentation pass remain historical;
`first-qualified-manifest.json`, `first-README.md.snapshot.txt` and
`first-release_checker.py.snapshot.txt` retain their original bytes.

The fresh merged-source record is `qualification-03/commands.json`, and its
documentation receipt is `documentation-checks-02.json` with exclusive log
`documentation-workflow-02.log`. The merged run passed all 166 distinct tests
in the same eight complete suites, including all 22 new cases, with zero
failures, errors or skips in 899.151 seconds. All 273 listed bindings and fixed
execution HEAD/index guards stayed unchanged. Incoming world-audit suites and
earlier counts are not added to this result.
Actual execution HEADs remain distinct from the original published source base;
every attempt retains its own exact source snapshots and Git guards.

## Scope and remaining work

The new 22 cases perform owner-side reads using genuine Git/CAS, native claim,
grants, Portal allocation/lifecycle and typed source connections. Listener and
deployment UID metadata are authored host seams. These cases launch no child.
Existing suites in the selected roster also launch and reap real inert
same-owner Python children; they do not qualify a separate worker UID. There are
no external provider/model calls, encoder runs, decoder training steps, solver
runs, Docker launches, target-program executions or official task results.

The change covers the broker's two selected-source delivery replays. Constructor
replay and the privileged native `source.get` gateway retain separate transaction
contracts, so this phase establishes no universal retained-CAS callback safety.
Permission delegation, authenticated cross-process native wrapper/child birth,
worker UID and signed compiled-context adoption remain separate prerequisites.
No proof, execution, completion, edit, omission or whole-session token-saving
authority follows from these checks. Currentness checks are sequential; there
is no filesystem writer lease.

The existing model work remains separate: CodebaseIR, IntentIR, SecurityIR,
LegalIR and UI/UXIR use their retained assets and independent family,
schema/version, decoder task and 8D/384D/768D checkpoints/inventories. Actual
encoder conditioning, fresh semantic and legal-text reconstruction, registry
custody and teacher promotion must be qualified before distillation. This source
delivery fix supplies no learned reconstruction or token-saving measurement.
