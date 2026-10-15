# Lossless transport for bounded Python code views

The additive wire format reduces repeated metadata keys while retaining the
entire public packet of the [bounded Python view](PYTHON_CODE_VIEWS.md).
It is an explicit library API, not a registered provider tool or a production
launch default. Native source and existing patch acceptance remain authoritative.

## Packet APIs and format

The [pure wire module](../../ipfs_accelerate_py/agent_supervisor/semantic_refactoring/python_code_view_wire.py)
exports `pack_python_code_view_packet(packet)`,
`unpack_python_code_view_packet(wire_packet)` and
`require_bound_python_code_view_packet(packet, view=...)`.
Packing returns `(wire_packet, stats)`; statistics stay controller-side.
Unpacking alone validates representation and source inventory. Authenticating
the packet requires comparison with the captured view's actual public packet.
A caller cannot authenticate substituted content by retaining its view ID.

Wire schema: `supervisor-python-code-view-wire@1`.
Profile: `python-arithmetic-model-packet-columns@1`.
The exact top-level fields are `schema`, `profile`, `view_id`, `bindings`,
`source`, `functions` and `slots`.

| Row | Fixed columns |
| --- | --- |
| `bindings` | path, repository ID, tree ID, task ID |
| function | function ID, name, alias rows |
| alias | original name, alias, kind index |
| slot | slot ID, function index, kind index, original-expression SHA-256, editable, refusal reason |

Alias kinds are `0=parameter`, `1=local`; slot kinds are `0=assignment`,
`1=return`. Function indices refer to ordered function rows. Indices are exact
integers; booleans cannot stand in for them. Names, source, binding values,
hashes, ordering and null/boolean distinctions remain literal. Full identifiers
and hashes are retained.

The generation profile restores the exact model-packet schema, revision and
five false authority fields. Unknown fields, schemas, profiles, enums, duplicate
IDs, mismatched source inventories, invalid row widths and noncanonical
encodings refuse. Constants implied by the profile cannot introduce new
authority or omit unknown metadata. Existing model packets and schemas stay
unchanged.

Input limits are checked before copying or serialization. Virtual expanded
bytes, nodes and depth are checked before repeated function names and row
objects are constructed. The format uses the engine's existing 2 MiB,
100,000-node and 48-depth artifact envelope, and existing source/AST/token
limits. It does not raise a context or source limit.

## Native worktree transport

The [wire worktree adapter](../../ipfs_accelerate_py/agent_supervisor/runtime/python_code_view_wire_worktree.py)
provides:

```python
read_worktree_code_view_wire(worktree=..., scope=..., view=...,
                            expected_view_id=..., lease_id=..., fence=...)
restore_worktree_code_view_wire(worktree=..., scope=..., view=...,
                               expected_view_id=..., wire_packet=...,
                               lease_id=..., fence=...)
build_python_code_view_wire_edit(view_id=..., slot_id=...,
                                before_sha256=..., expression=...)
parse_python_code_view_wire_edit(text)
expand_python_code_view_wire_edit(request)
prepare_worktree_code_view_wire_expression(worktree=..., scope=..., view=...,
                                          expected_view_id=..., wire_request=...,
                                          lease_id=..., fence=...,
                                          visible_sources=...)
```

Read obtains the current public packet through the existing native owner/source
checks, packs it, checks exact restoration and rechecks the producer. Restore
compares the complete expanded public packet with that same freshly checked
producer. Both require current repository/task/tree/source, scope, lease and
fence bindings. These checks are separate from pure syntax unpacking.

The edit schema is `supervisor-python-code-view-wire-edit@1`, with exact fields
`schema` and `edit`. Its fixed row is:

```text
[full_view_id, slot_id, full_original_expression_sha256, literal_expression]
```

Raw JSON requests use the engine's duplicate-key parser before conversion into
a request dictionary. JSON text is bounded to 64 KiB; identifiers and replacement
expressions retain their native bounds. Dictionary callers must preserve raw
syntax evidence at their own parsing boundary; an ordinary `json.loads` cannot
recover a duplicate key after discarding it.

Preparation expands the row, compares its full view identity with the expected
captured identity and calls the original `prepare_code_view_expression`.
Application uses the original `apply_code_view_patch`. No parallel patch
executor or database authority is added.

Caller-provided `visible_sources` passes unchanged. It must represent native
source actually available to the model; private originals or packed views do
not satisfy that requirement. Accepted transport, packet restoration, a native
patch or fixture execution grant no task acceptance or Lean admit. Existing
controller cleanup and acceptance responsibilities remain in place.

## Measurement and remaining work

The [retained campaign](../../../../artifacts/supervisor-python-code-view-wire-20261012/README.md)
compares native, unpacked and packed complete messages for the same authored
task/source at three sizes. Packed-message sizes include layout and edit
instructions. Public-packet bytes, full-message bytes, edit JSON body bytes,
required native-source expansion and warm helper timing remain separate.

The larger fixtures are arithmetic stress fixtures inside the existing closed
profile. They do not represent arbitrary application modules or an expanded
encoder context. Their native reconstructed edits are executed against
independent expected results. Unknown tokenizer/session values stay unknown.

Even an initially smaller packed message still requires native-source visibility
under the unchanged gate. The measured expansion source bytes exclude message
framing. The edit measurement counts its JSON body, not a complete provider
reply or a matched native reply baseline. No overall token or session savings
follow from these component observations.

The next production trial needs a registered broker, a compatible exact
tokenizer, explicit native visibility accounting and unchanged task acceptance.
Immutable profile/legend reuse could amortize repeated layout costs across
reads and edits, but this prototype does not assume such reuse or select wire
packets from byte counts. A useful comparison counts planning, coding, reads,
searches, expansions, diagnostics, edits and retries together with actual task
success.

Only an actual `lake build <Lib>` grants a Lean admit. This campaign performs no
Lake build or autoencoder training; the Constitution remains unformalized.
