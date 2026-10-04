# Semantic state serialization: joined consumer controls

All 558 selected test identities execute and pass against unchanged recorded
source files: 212 supervisor tests, 153 focused datasets tests, 186 additional
datasets integration/AST observation tests and seven actual checkpoint/GTE
source-unit inference and replay tests. Failures, errors and skips are zero.
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
semantic capsules and cleanup. The seven native cases execute the actual local
checkpoint and embeddings with no provider call or training. Successful tests
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
