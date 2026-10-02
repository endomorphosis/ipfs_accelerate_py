# Current repository evidence at admission and worker dispatch

`runtime.repository_behavioral_admission` adds an opt-in evidence fence to the
existing independently signed local task admission and finite candidate handoff.
It invokes the v2 behavioral planner itself. It does not accept caller-provided
facts, a saved successful match, or a reduced native task population.

`prepare_behavioral_repository_handoff` accepts the existing finite handoff
arguments plus the exact `IntentCodebaseCatalog`, `FiniteCheckedCache`, and
`semantic_manifest_cid`. The source index, discovery catalog, and proof cache
must share their existing native artifact owner. The profile is explicitly
model-off. It returns the existing candidate result plus
`repository_admission_path`, `repository_admission_sha256`, `behavioral_preview`,
and `public_context`.

The signed artifact retains both original and v2 requests, the complete planning
snapshot, native graph and independently signed planning receipt, every declared
task CID, the original candidate signature, complete retained observation bodies,
source head, catalog and proof-store identities, producer hashes, and the public
context reference. Native IntentIR and prompt-graph JSON retain their finite
confidence values losslessly; native owners decode them during replay. Those
values are not proof claims.

`verify_behavioral_repository_handoff` reopens the existing native evidence
owners, observes the current repository, invokes the fresh behavioral matcher,
and reconstructs the archived observation/fact provenance and complete planning
snapshot. It rejects a changed source, scope, root, owner inode, producer, model
mode, task population, archived body, or proof meaning. A stored signature or
receipt is not a live checker capability. The native Intent database is separate
and is not opened by worker evidence replay.

The worker entry point is:

```text
python -m ipfs_accelerate_py.agent_supervisor.runtime.repository_behavioral_runner
  --repository-admission PATH --repository-admission-sha256 SHA256
  --artifact CANDIDATE --sha256 SHA256 --task-cid TASK
  --owner-did DID --profile-id PROFILE
  --public-context PATH --public-context-sha256 SHA256
```

It receives the existing native prompt on standard input and runs in the
allocated Git worktree. It performs a fresh fence before and after the unchanged
candidate materializer. Native claimed-task revisions can differ from the ready
revision; exact signed task/objective custody remains required. The worker needs
no owner private key. Native daemon validation, publication and completion retain
their existing ownership.

Before replay, close the owning source DuckDB connection. The source and proof
databases must be different existing canonical files, with no group/world write
permission; the CAS directory is likewise owner-controlled. The signed path,
device, inode, mode and owner are checked. Missing databases are refused before
any constructor can create storage. The tests provision private files and
directories explicitly; the API does not change existing permissions.

This is the declared bounded Python integer-offset profile. All native task
records remain present. Before/after fences are not an atomic filesystem/database
transaction: a failed post-write fence can leave private worktree bytes, but the
worker returns failure and grants no publication or completion. Broader language,
model-dependent interpretation, alternative/prohibition task synthesis, generic
Quack/DQP transport, and hard host-resource enforcement are separate work.

# Bounded typed intent and residual queries

`analysis.repository_intent_evidence` supplies immutable
`IntentResolutionQuery` and `ResidualObligationsQuery` values. Their native Intent
and tool-policy inputs use exact bounded canonical JSON. The source instruction
and complete native statement population are bounded; unknown fields and
ambiguous encodings are refused.

`RepositoryIntentEvidence` delegates to the actual `RepositoryCodeEvidence`
owner. It returns the complete finite-profile requirement population atomically,
with all residuals, the full native result, the query/root identity and exact
consumed match, record and receipt commitments. There is no partial cursor for
this small complete population. Empty residuals do not grant task omission,
execution or completion authority. The underlying larger unit/evidence queries
retain their existing root-bound pagination.
