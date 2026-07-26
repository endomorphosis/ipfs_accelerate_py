# Agent Supervisor External Completion Receipts

Operational work performed outside the repository is not proved by an
implementation marker, a completed local task, or an artifact filename. The
agent supervisor therefore treats repository scanning as discovery only and
uses an explicit, typed external-completion authority for verification.

The authority is supplied to `objective_daemon` with:

```text
--objective-external-completion-receipt-path /external/state/completion.json
```

The JSON file must contain only canonical identities and bounded status
metadata. It must not contain source text, labels, case or corpus content,
manifests, proof obligations, holdout material, filesystem paths, command
output, or secrets. Unknown fields fail parsing instead of being copied into
the objective heap.

## Trust and lifecycle

Declare every operational objective with:

```text
- Completion authority: external
```

That property governs the goal from its first reconciliation, before an
authority file has ever been supplied. A same-named implementation marker
therefore cannot bootstrap an operational completion decision.

An authority contains two distinct sets:

- `requirements` are supervisor-selected expected identities for each exact
  `(goal_id, evidence_term)` pair.
- `receipts` are assertions returned by external execution and independent
  validation.

For a governed goal, external authority replaces local or previously
persisted completion evidence; it does not augment it. An absent or invalid
external receipt therefore fails closed. Once a goal records external
governance, omitting the explicit authority path on a later run cannot turn
its historical receipt into ordinary local evidence: the receipt is cleared
from the usable evidence set and a verified goal reopens. As with other
objective completion, one reconciliation may move a drained goal to
`provisionally_complete`; a later reconciliation reruns the goal's local
validation before it may become `verified_complete`.

Marker scanning remains useful for finding implementation gaps, but its paths
and text cannot satisfy an externally governed criterion.

## Authority shape

All fields ending in `_cid` or `_id` below are lowercase base32 CIDv1 values,
except Git commits and trees, which are complete lowercase Git object IDs.

```json
{
  "schema": "ipfs_accelerate_py.agent_supervisor.external-completion-authority.v1",
  "requirements": [
    {
      "schema": "ipfs_accelerate_py.agent_supervisor.external-completion-requirement.v1",
      "goal_id": "GOAL-001",
      "evidence_term": "OPAQUE_EVIDENCE_TERM",
      "source_identity_cid": "b...",
      "run_plan_cid": "b...",
      "parent_ledger_cid": "b...",
      "required_artifact_ids": ["b...", "b..."],
      "expected_producer_id": "b...",
      "expected_validator_id": "b...",
      "requirement_cid": "b..."
    }
  ],
  "receipts": [
    {
      "schema": "ipfs_accelerate_py.agent_supervisor.external-operational-completion.v1",
      "goal_id": "GOAL-001",
      "evidence_term": "OPAQUE_EVIDENCE_TERM",
      "source": {
        "schema": "ipfs_accelerate_py.agent_supervisor.external-source-identity.v1",
        "outer_commit": "0123456789abcdef0123456789abcdef01234567",
        "outer_tree": "89abcdef0123456789abcdef0123456789abcdef",
        "clean": true,
        "recursive_gitlinks_complete": true,
        "recursive_gitlinks": [],
        "submodule_map_cid": "b...",
        "source_identity_cid": "b..."
      },
      "run_plan_cid": "b...",
      "parent_ledger_cid": "b...",
      "artifacts": [
        {
          "schema": "ipfs_accelerate_py.agent_supervisor.external-artifact-identity.v1",
          "artifact_id": "b...",
          "artifact_cid": "b..."
        }
      ],
      "producer_id": "b...",
      "validator_id": "b...",
      "validator_receipt_cid": "b...",
      "observed_at": "2026-07-25T12:00:00+00:00",
      "fresh_until": "2026-07-25T13:00:00+00:00",
      "status": "completed",
      "receipt_cid": "b..."
    }
  ],
  "authority_cid": "b..."
}
```

The three derived identities—`requirement_cid`, `receipt_cid`, and
`authority_cid`—are recomputed during loading. Supplying a mismatched value is
an error.

## Source binding

`inspect_external_source()` uses Git metadata only. It binds:

- the exact outer `HEAD` commit and tree;
- a completely clean outer worktree, including the objective heap and any
  supervisor state stored inside the repository;
- every recursively initialized gitlink checkout;
- each gitlink's recorded commit and actual tree; and
- a canonical CID of the complete recursive map.

Gitlink checkout locations are folded into opaque `gitlink_id` CIDs and are
never persisted. A missing checkout, dirty submodule, recorded/actual commit
mismatch, dirty objective heap, recursion failure, or changed outer source
rejects the receipt. If lifecycle state is stored in a tracked objective heap,
that transition must be committed before a later receipt can attest the new
source state. Source inspection is repeated after legacy-goal migration, so a
migration rewrite cannot reuse a clean snapshot captured before that rewrite.

## Fail-closed checks

Validation rejects, among other cases:

- unknown or marker-only JSON shapes;
- unbounded status text or a receipt binding with no matching requirement;
- missing or duplicate requirements and receipts;
- wrong goal, evidence term, run plan, source, producer, validator, or parent
  ledger identity;
- dirty, stale, incomplete, future-dated, or expired source/receipts;
- missing, unexpected, or duplicate artifact slots/content CIDs;
- a producer acting as its own validator;
- a validator receipt identity reused as an artifact or other authority
  identity; and
- any unsupported field that could carry a path or external content.

The supervisor persists the authority, requirement, source, artifact,
validator, and receipt identities plus reason codes. It does not persist the
external artifacts or validator payloads.

## Design limit

The adapter verifies identity consistency and current-source binding; it does
not fetch external CIDs or verify a validator's cryptographic signature. The
configured authority file is the trust root for expected producer, validator,
run-plan, parent-ledger, and artifact-slot identities. Deployments requiring
cryptographic authorship must verify signatures or attestations before
handing this identity-only authority to the supervisor.
