# Semantic state serialization: joined consumer controls

All 558 selected test identities execute and pass against unchanged recorded
source files: 212 supervisor tests, 153 focused datasets tests, 186 additional
datasets integration/AST observation tests and seven source-unit controls,
including one actual pinned-checkpoint/GTE inference and warm/reopened-registry
replay case plus six preparation/tamper cases. Failures, errors and skips are zero.
The 39 manifest memo tests already covered by the focused run are excluded from
the later integration run, so these counts contain no duplicate identities.

The change is confined to native `RepositoryState.to_dict`: one freshly built
identity payload is hashed before adapting it to the existing public schema.
The same detached child dictionaries serve the hash and output. No body or
identity is cached between calls. Native dispatch/record guards preserve the
historical path for custom serializers, subclasses and replaced outer
containers. Schema, ordering, canonical bytes, CIDs, validation, source fences,
proof authority, reservations and deadlines remain unchanged.

These controls include exact output/CID equivalence, detachment/current values,
corruption and duplicate rejection, custom dispatch, native manifest memo
compatibility, durable index observations, supervisor initialization/context,
semantic capsules and cleanup. The seven source-unit controls comprise one bounded-preparation case, five
parameterized tamper cases, and one case that executes the actual pinned
checkpoint/GTE numerical worker and then verifies warm and reopened-registry
replay. Both replay calls occur in the same host process; the numerical worker
uses a fresh subprocess. The actual inference case asserts one model load,
zero provider calls and no training. Successful tests
do not establish a successful Docker task or a token-efficiency advantage.

The separate datasets package `semantic-state-serialization-20261004` retains
the producer snapshots, focused controls and local public Bottle diagnostic.
Its two fresh samples per mode show identical manifest/snapshot IDs and median
CPU 15.621 to 14.653 seconds, wall time 15.721 to 14.738 seconds. That small host
sample is diagnostic only; it does not establish native-container speed, RSS
reduction or admission recovery.

This package contains execution metadata and test reports; retained log hashes
refer to local artifacts. Private stores, model/credential bodies and hidden
benchmark verifier inputs are excluded. The source revisions describe the
committed producer state at export; each test command also records the exact
file hashes consumed by its run. The backlog remains 18 of 32 closed pending
current-generation Docker qualification, a completed task and matched arms.

Version 2 corrects the source-unit execution scope and adds an explicit 1+5+1
case breakdown. All 558 unique test identities, command/exit/XML bodies and
source pins are unchanged. Previously recorded run-admission, build and command
metadata retain their verbatim historical scope wording; this clarification
does not change their recorded execution or establish a new runtime result.
