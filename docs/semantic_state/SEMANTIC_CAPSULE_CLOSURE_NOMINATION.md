# Nominate known declared capsule dependencies

`semantic_capsule_closure_nomination.nominate_declared_capsule_closure` adds all
known outgoing declared dependencies of supplied seed capsules. It returns
whole native records and keeps unresolved names explicit. This is an advisory
nomination; it creates no worker artifact, owner, source substitution or live
supervisor policy.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_closure_nomination import (
    nominate_declared_capsule_closure,
)

nomination = nominate_declared_capsule_closure(
    full_capsules=verified_full_worker_payload["capsules"],
    seed_symbol_ids=selected_seed_ids,
    max_capsules=32,
    max_packet_bytes=65536,
)
```

The caller must first verify current source, the complete permitted inventory
and native producer root. The pure helper snapshots literal input JSON,
revalidates native capsule identities and reuses the existing dependency audit.
The seed and nominated-set audits must bind the same graph and inventory.

## Candidate and receipt budgets

An available report contains `nominee_packet`, whose `capsules` hold the exact
copied native records. Packet metadata binds seeds, capsule identities, known
closure, unresolved names and cycles. The complete canonical packet, including
its wrappers, must fit the capsule count and byte bounds. Its measured bytes
and SHA256 are stored outside the packet as `nominee_packet_bytes` and
`nominee_packet_sha256`.

Count or byte insufficiency refuses the whole candidate. There is no partial
dependency set: `nominee_packet`, its size and its hash become `None`, and
refusal reasons remain in a body-free receipt. Empty seeds also refuse. Invalid
native records, duplicate or foreign seeds, expanded limits, and graph-bound
violations raise errors.

`packet_budget_scope="nominee_packet_only"` distinguishes the candidate from
the diagnostic receipt. The receipt can exceed a very small candidate budget;
it has an independent two-megabyte hard ceiling and is not provider input.
The nominee packet is not a native worker or complete model prompt. Task,
authority, raw source, acceptance, reference and router overhead still require
the ordinary compiler and complete rendered user-message measurement.
These local counts exclude provider/system instructions and subsequent tool
turns; whole-session usage remains a separate measurement.

The helper returns bodies detached from mutable input. Consumers should verify
the packet's retained digest before later use and repeat source/root checks at
the actual decision boundary.

## Interpret closure precisely

Only outgoing `dependency_stable_ids` edges are traversed. Native dependencies
can be symbolic text such as `global:helper_operation`; a matching source name
does not resolve that reference. Known dependencies, unresolved targets and
cycles remain distinct. Cycles terminate without making a finite closure
invalid by themselves.

`declared_graph_closed` refers only to the supplied declared graph. It does not
establish complete extraction, incoming caller impact, test/fixture coverage,
proof obligations, effects, serialization/schema relations, or task success.
Source freshness, semantic completeness, proof, acceptance, deployment and
activation authority remain false in every nomination.

The existing native selector is lexical. It cannot accept this packet as a new
selection policy. The real fixture uses an explicit `target_operation
helper_operation` query and checks that its actual selected identities equal
the nomination. Failure to reproduce is recorded as `cannot_reproduce`; the
probe never repairs or forges a native artifact to fit the proposed set.

## Retained real-producer control

Three native preparations share the same task, root, source inventory,
acceptance inputs, required raw paths and 6,144-byte reference policy. All three
source/test/instruction files remain embedded. Each V2 arm restores exactly to
its own native prompt, and current-source changes refuse.

| Complete rendered context | Native bytes | V2 bytes |
| --- | ---: | ---: |
| Full context | 62,162 | 48,975 |
| Target seed | 16,783 | 13,963 |
| Target plus known helper | 21,194 | 17,617 |

The selector reproduced the two-capsule nomination. Missing known dependencies
dropped from one to zero; the separate `global:helper_operation` reference
remains unresolved. The nomination packet alone is 6,348 bytes, illustrating
why its size cannot replace the complete rendered user-message measurement.

These are bytes for one authored task with real native producer/codec checks.
Model-token counts, acceptance execution, whole-session savings and Terminal
Bench outcomes remain unknown. The previous source-drift development trial is
retained separately; final evidence uses the frozen helper revision.

See [selection comparison](SEMANTIC_SELECTION_COMPARISON.md),
[whole-session accounting](TOKEN_USAGE_RECONCILIATION.md), and the retained
`artifacts/supervisor-closure-nomination-20261012-01` evidence in the enclosing
workspace. A live selection policy requires its own current-source,
dependency/coverage and owner integration contract.
