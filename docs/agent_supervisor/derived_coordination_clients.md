# Sharing derived code state

The separate derived DuckDB + Quack owner accepts bounded AST snapshots,
content hashes, and references to source state. Its results remain derived
evidence. Git, datasets, and kit keep their existing source and semantic
authority; the derived service cannot accept a task or complete a board.

`attach_typed_instance(..., derived_repository_id=repository_id)` returns a
`QuackStateClient` that supports `derived_coordination(payload)`. The public
client checks the returned owner against its attached session. Callers do not
need access to a private transport or a local database file.

Owner-issued derived session grants expire after two minutes. A supervisor
that keeps the same connection across idle periods must not reuse an expired
grant. Use a connection factory to get a fresh authenticated session for each
bounded operation:

```python
api = DerivedCoordinationClient(
    repository_id=repository_id,
    connection_factory=open_admitted_derived_client,
)
result = api.call("list_references", tree_id=tree_id)
```

The factory should call `attach_typed_instance` for the configured separate
owner and matching repository scope. Resolve its owner-issued credential
through the existing private credential provider on each call. The client
closes the connection on success or failure. It never replays a failed
operation: a transport failure after a write can leave its outcome unknown.
The caller can inspect the immutable snapshot/reference through a fresh read
before deciding how to recover. Supplying an existing connection still leaves
that connection's lifetime with the caller.

The dedicated owner health probe now performs a bounded derived reference
read as well as checking its generation. A listening owner whose derived
service cannot respond is therefore detected by the existing repeated-failure
recovery policy.
