# Semantic-world acceptance tests

This directory is the protected landing point for the focused acceptance
suite of `semantic-addressed-world-model-v1`. The R2 control board assigns
individual test modules to implementation tasks; this file does not claim
that those tests or capabilities already exist.

The suite must remain hermetic by default. Cold imports, CLI discovery, schema
round trips, CID verification, Merkle/logical graph behavior, retrieval and
reuse admission, traces, transition prediction, repair proposals, procedure
promotion, root CAS, VFS outbox, recovery, privacy, and adversarial cases are
all tested without implicit network access, installers, model downloads, or
authoritative simulation.

Learned-capability tests must accept the typed `training_unavailable` and
`checkpoint_unavailable` outcomes when admitted corpora or checkpoints are
absent. They must never manufacture a checkpoint or count an LLM label as
ground truth. Every neural result is proposal-only and exact resolution plus
current policy, environment, proof, test, and validation gates remain in
force.

The board validator and frozen benchmark manifest are operator-owned controls.
Implementation workers may add task-owned test modules below this directory,
but may not modify this README or use generated runtime state as source.
