# Datasets semantic-root discovery through a separate owner

`CoordinatedSemanticStateProvider` is an explicit composition of the existing
datasets provider and a scoped `DerivedCoordinationClient`. It publishes a
discovery reference only after datasets builds and verifies the complete bundle.
The original bundle object is returned. A failed or lost registry response is
reported separately as `last_coordination.status == "unknown"`; neither the
producer nor the possibly committed registry request is retried.

Existing provider construction remains unchanged unless a client is supplied:

```python
client = DerivedCoordinationClient.from_fleet_deployment(
    deployment_path,
    repository_id=repository_id,
    client_id="supervisor:semantic-producer",
    timeout_seconds=5,
)
provider = load_semantic_state_provider(derived_coordination_client=client)
bundle = provider.build_semantic_state(datasets_repository_state)
```

The client reads the explicit managed fleet deployment, authenticates a fresh
typed session for each operation, and closes that session afterward. It never
opens the derived DuckDB file. The provider wrapper grants no database, network,
or registry access to datasets itself. Existing plain provider calls and the
sealed pure delegation methods remain unchanged.

A second supervisor supplies its own authorized block reader:

```python
consumer = CoordinatedSemanticStateProvider.from_fleet_deployment(
    existing_datasets_provider,
    deployment_path,
    repository_id=repository_id,
    client_id="supervisor:semantic-consumer",
)
for discovered in consumer.iter_discovered_views(
    tree_id=bundle.root.producer.repository_snapshot_cid,
    get_block=authorized_block_reader,
):
    view = discovered.view
```

Every result is reopened by the datasets verifier. Its repository, snapshot,
state and root identities and exact root-block hash must reproduce the hint.
Subsequent view reads retain datasets verification. Missing or corrupt blocks,
wrong identities and invalid discovery pages refuse; the registry is not a
block store. Callers retain responsibility for authorized, bounded block access.

The existing reference wire uses legacy names. This composition defines them
explicitly and reports the descriptive names in `last_coordination`:

| Wire field | Exact meaning in this composition |
| --- | --- |
| `tree_id` | Datasets producer `repository_snapshot_cid`, unchanged |
| `ast_cid` | Datasets producer `repository_state_cid`, unchanged |
| `state_root` | Datasets semantic root CID, unchanged |
| `content_hash` | SHA-256 of the exact semantic-root block bytes |

These fields do not assert a standalone AST CID, Git tree OID, or hash of all
repository source bytes. A working snapshot may have no Git OID. Root blocks
are limited to 32 KiB for this reference profile. Discovery uses the existing
bounded paginated client and is not a point-in-time snapshot.
If a valid producer result exceeds that coordination-only profile, it is still
returned with `last_coordination.status == "not_published"`; no registry request
is made. Semantic verification errors always remain errors.

References remain unverified registry hints. A datasets-verified view does not
grant current-root, proof, completion, merge or complete committed-population
authority. Opaque and heuristic producer limitations remain intact. This hook
does not publish kit root-CAS placeholders or modify task rows.

Deployment still must explicitly compose this provider in the intended
supervisor workflow and supply durable authorized blocks. This source change
does not enable native boards. Embedding, BM25, knowledge-graph, proof-cache and
certificate producers require their own publication and consumer verification
hooks; their registry capability names alone do not prove those flows exist.
