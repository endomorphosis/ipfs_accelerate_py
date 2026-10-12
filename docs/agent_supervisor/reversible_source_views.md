# Reversible read-only source views

The [source-view codec](../../ipfs_accelerate_py/agent_supervisor/runtime/reversible_source_view.py)
and [repository view inventory](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_code_view_inventory.py)
implement compact historical views backed by the
[retained committed source inventory](repository_source_inventory.md).
Every retained byte is covered by a canonical KEEP/DELETE map. The unchanged
compact view restores the exact original bytes, including removed comments and
whitespace, by using those bytes from existing source custody.

This is a repository-wide dual representation: original committed bytes remain
in the current supplied content store; read-only compact candidates are separate
artifacts. No executable checkout is rewritten. Edited compact views refuse
restoration; translating edits requires its own mapping contract. The current
coding input and its required source remain unchanged until model-tool access
and complete-input selection are qualified.

## Profiles and fidelity

| Profile | Candidate view | Preservation boundary |
| --- | --- | --- |
| `identity@1` | Exact original bytes | All languages, binary data and encodings can remain exact |
| `python-whitespace@1` | Remove safe trailing trivia | Default preserves all comments, strings/docstrings, names, imports, indentation and physical lines |
| `python-comments@1` | Also remove ordinary Python comments | Explicit opt-in; keep headers, shebang/cookies, licenses, conservative tool/security/contract comments and caller-protected lines |

Python transforms use strict UTF-8 token/AST checks without `ast.unparse`,
renaming or target execution. Unicode token columns are converted to exact
UTF-8 byte offsets through original LF-separated prefixes; visual columns and
AST byte offsets are not substituted. CRLF/LF and final-newline bytes remain
exact. Strings/docstrings keep their contents. Valid f-string sources pass
through entirely with `unsupported_fstrings`, including Python 3.12 expression
comments; this first version does not attempt that source mapping.

Unsupported encodings, invalid syntax/tokens, lone-CR newlines, too-large inputs
and excessive token/piece populations use explicit identity fallback. BOM bytes
remain exact; the Python parser's unsupported BOM text uses identity fallback.
A candidate changing physical line count or checked token/AST structure also
falls back. The original committed inventory remains complete.

The default's comment preservation matters: task-relevant explanations can live
in otherwise ordinary comments. The comment profile's keyword protection is a
conservative display rule, not a proof of semantic relevance or permission to
omit required facts. Structural checks do not establish executable equivalence,
formal proof, current source or task completion.

## Pure codec

```python
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import (
    build_source_view, restore_source_view,
)

candidate = build_source_view(original_bytes, profile="python-whitespace@1")
restored = restore_source_view(
    original=original_bytes, view=candidate.view, manifest=candidate.manifest,
    expected_manifest_sha256=candidate.manifest_sha256,
)
assert restored == original_bytes
```

The caller independently admits original source and the manifest identity.
The manifest records source/view SHA-256 and lengths, requested/effective
profile, fallback reason, protected lines, codec binding and merged KEEP/DELETE
byte ranges. Those ranges partition the complete original source without gaps
or overlap. Empty input permits an empty map. The sidecar copies no removed text.

Restoration validates the manifest's closed fields, source/view pins, bounded
partition and false authority flags. It regenerates the deterministic profile
from exact original bytes and compares the complete candidate/map/manifest.
It then joins retained view bytes for KEEP ranges with original bytes for DELETE
ranges and checks exact equality. A rehashed arbitrary deletion map, edited
view, stale binding or changed profile cannot masquerade as that codec output.

## Complete repository inventories in the same CAS

```python
from ipfs_accelerate_py.agent_supervisor.runtime.repository_code_view_inventory import (
    retain_code_view_inventory, load_code_view_inventory,
)

# existing_cas, source_ref, request and source_limits come from the admitted
# retained-source workflow. No new content store or database owner is created.
views_ref = retain_code_view_inventory(
    cas=existing_cas, expected_source_inventory_cid=source_ref.inventory_cid,
    request=request, source_limits=source_limits, profile="python-whitespace@1",
)
views = load_code_view_inventory(
    cas=existing_cas, expected_inventory_cid=views_ref.inventory_cid,
    expected_source_inventory_cid=source_ref.inventory_cid,
    request=request, source_limits=source_limits, profile="python-whitespace@1",
)
page = views.metadata_page(offset=0, limit=64)
```

Only regular `.py`/`.pyi` entries are nominated for Python transformation.
Other regular files, binaries, unsupported source and symlink targets preserve
exact identity views. Gitlinks remain explicit child references with no view
or child bytes. Original-source identity views reference the existing source
inventory; they do not create another whole-blob CAS object for large passthrough
files. Changed views are raw blocks, and sidecars/index pages are structured
blocks in the same supplied CAS.

Each row binds original raw path, Git mode/OID/source CID, view CID/length,
storage kind and sidecar CID/SHA. The closed root binds independent source
inventory/request/limits, profile, protected-line policy, codec binding and
canonical index pages. Readback joins every row to the full source population,
rejects omissions/duplicates/noncanonical paging and restores every retained
blob. Gitlink frontiers and `content_complete` remain explicit.

Request, limits, protected-line policy and codec binding are checked around
storage callbacks using the same original CAS/configuration/directory bindings.
Metadata records and byte totals are bounded before expansion. The root is
published last after replay, then the actually stored root and views are replayed
again. Partial immutable artifacts may remain on failure; no successful reference
is returned. Checks are sequential, without an atomic store lease or rollback.

`protected_lines` is a host mapping from raw path hex to sorted unique positive
physical line numbers. At most 256 Python paths and 4,096 total protected lines
are accepted. Unknown/non-Python paths refuse. The identical policy must be
supplied on reload. A later file's invalid line can refuse after earlier artifacts
were retained; that does not publish a successful inventory.

## Bounds, portability and measured scope

Python transformation is limited to 1 MiB input, 131,072 token records and
4,096 merged pieces. Larger or unsupported source stays exact through the
original inventory. Source population/materialization limits remain inherited
from that inventory. Changed raw views and structured metadata blocks fit within
1 MiB; the aggregate metadata budget and bounded host page sizes still apply.

Codec binding includes codec and `ast`/`token`/`tokenize` source-file hashes plus
reported Python version, implementation and cache tag. It is not a complete
interpreter/toolchain attestation. Reload requires the same current binding and
deterministic profile. A changed runtime/codec refuses the view rather than
silently migrating it; the independent original-source inventory remains usable.

Binding checks reread these implementation files, and codec replay repeats local
token/AST work. The Python executable is not rehashed per callback. Native Stage 1
export still rehashes full Git blobs per frame. Qualify CPU/I/O cost before broad
deployment; this stage establishes no speed improvement.

Observation byte counts cover raw historical source and raw candidates, without
model guidance, wrapper labels, sidecars or tool costs. Metadata pages contain no
source bodies and are host data, not a registered Quack operation. Delivering
all sidecars to a model can erase compact-view gains. Complete provider-bound
input selection and total planning/coding/lookup/repair usage remain separate
experiments. Identity fallback and edited-view refusal remain explicit.

Existing LegalIR, SecurityIR, CodebaseIR and IntentIR inventories/checkpoints
retain their schema/version/decoder-task and 8D/384D/768D separation. This reuses
exact source custody without inference, training, embedding regeneration or
new model assets. Current-owner expansion, original-source editing, working-overlay
refresh and later mapped compact edits remain the next integration stages.

The [qualification evidence](evidence/reversible-source-views-20261012/README.md)
retains 40 passing cases, including a separate-process unchanged inverse without
the original checkout. Source hashes, exact commands, initial failure and final
corrected fixture are preserved. Rebuild and qualify a current worker archive
before operational deployment; this host API is not a live benchmark result.
