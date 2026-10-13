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
artifacts. No executable checkout is rewritten by the view codec. Edited compact
views still refuse ordinary restoration; the separately qualified edit mapper
below constructs original-source candidates through an explicit edit contract. The current
coding input keeps its required source. An opt-in
[coding-worker source reader](worker_source_views.md) now serves task-scoped
compact views and exact original expansion, with counted guidance and explicit
stale-source refusal. Selecting source omissions requires separate qualification.

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

## Share an explicit policy across owner APIs

The additive [policy resolver](../../ipfs_accelerate_py/agent_supervisor/runtime/source_view_policy.py)
lets the inventory, host source reader and edit adapter use the same regular
`.py`/`.pyi` classification and caller-protected physical lines:

```python
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy

policy = SourceViewPolicy.from_mapping(
    profile="python-comments@1",
    protected_lines={b"pkg/types.pyi".hex(): [2]},
)
resolved = policy.resolve(raw_path=b"pkg/types.pyi", regular_file=True)
assert resolved.python_eligible and resolved.protected_lines == (2,)
```

Pass the same `view_policy=policy` to `retain_code_view_inventory`,
`load_code_view_inventory` and `WorkerSourceViews`. Leave their legacy `profile`
argument at its default and the inventory's legacy `protected_lines` unset;
the policy supplies both. Protection keys must refer to admitted regular Python
paths. The immutable policy detaches caller mappings and binds configuration,
resolver implementation bytes and the unchanged codec. Inventory roots use
`supervisor-historical-code-view-inventory@2`; old roots and defaults retain
their original schema and identities.

The owner source reader binds that policy to its admitted source population and
candidate generations. Its checked `working_projection` accessor supplies the
edit adapter's view/map instead of a second suffix/protection recipe. Shared
application receipts use `supervisor-source-view-edit-application@2` and retain
the policy binding digest. Protected lines preserve display text and physical
coordinates; they do not prohibit edits within retained bytes. The existing
mapper refuses newline-changing edits while caller-protected lines exist.

The first [policy qualification](evidence/shared-source-view-policy-20261013/README.md)
covered owner APIs. The subsequent
[transport increment](evidence/source-view-discovery-transport-20261013/README.md)
now carries shared baseline/working pages (versions 3/4) and discovery (version 2)
through an owner-selected broker, descriptor version 2 and request version 3.
The client automatically uses the descriptor's policy digest, including exact
expansion and working generations; legacy requests still refuse shared pages.
`--discover` lists only already admitted paths and their source/view pins.
Standard benchmark configuration still uses legacy policy defaults. Repository
scope admission, benchmark policy selection, full filesystem export, token
savings and model-driven compact editing remain distinct concerns.

## Map an explicit compact edit to original bytes

[`source_view_edits.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/source_view_edits.py)
adds a pure edit mapper. Original bytes and the complete codec map remain with
the owner. A proposal identifies an exact compact byte range, its preimage and
replacement; no fuzzy matching or arbitrary whole-file inverse is used.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_edits import (
    map_source_view_edit,
)

mapped = map_source_view_edit(
    original=original_bytes, view=candidate.view, manifest=candidate.manifest,
    expected_manifest_sha256=candidate.manifest_sha256,
    view_start=start, view_end=end,
    expected_view_preimage=candidate.view[start:end], replacement=replacement,
)
assert mapped.view == candidate.view[:start] + replacement + candidate.view[end:]
# mapped.candidate contains original-source bytes with only the mapped span changed.
```

The mapper first authenticates the unchanged original/view/map using the existing
codec. Nonempty edits must fit inside one retained KEEP interval. Insertions are
allowed inside that interval or at an exposed file endpoint; collapsed deleted
regions are ambiguous and refuse. Every byte outside the mapped original range
is preserved, including hidden comments. Rebuilding the view from the proposed
original source must produce exactly the requested compact edit under the same
requested/effective profile, fallback status and codec binding.

For transformed Python, edits preserve UTF-8 character boundaries and whole CRLF
pairs. When caller-protected physical lines exist, preimage and replacement may
contain no LF, avoiding protection shifts. Identity views can still contain
arbitrary bytes and carry no syntax guarantee. Profile/fallback transitions,
invalid transformed Python, no-op edits, stale pins and failed replay refuse.
Source/candidate sizes are at most 1 MiB, preimage/replacement at most 64 KiB each,
and canonical manifests at most 512 KiB. The immutable result provides candidate
bytes, the rebuilt view/manifest and an advisory mapping receipt.

This checks exact mapping, not behavioral correctness, live source freshness or
permission to write. The codec reads its own implementation bindings; no target
source is executed and the pure mapper writes no files.

## Apply through the existing isolated worktree owner

[`source_view_edit_candidate.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/source_view_edit_candidate.py)
connects that mapper to `IsolatedWorktree.apply_patch`. The caller supplies an
existing fenced allocation, a matching `WorkerSourceViews` owner and an independently
authorized `PatchScope`. It requires the generation, source/view/manifest hashes
returned by a prior working-source read, plus the exact range/preimage/replacement.

The adapter checks owner, task, allocation, source scope and all admitted working
bytes before mapping and again before native application. It submits a normal
original-source patch through the existing lease/fence and patch-scope checks.
Afterward it verifies the staged file and complete admitted source population,
then explicitly refreshes the working generation. Prior-generation source and
formal-context continuations can no longer be used. The receipt remains
`applied_validation_pending`: tests, prover checks and acceptance are separate.

This first adapter accepts one modified file in a fresh clean candidate. It
supports ordinary unquoted paths and UTF-8 text with LF/CRLF or no final newline.
Binary patches, special diff-path quoting, creates/deletes/moves and incremental
patching on a dirty overlay require separate support. It preserves the existing
writer's staged-candidate behavior; the original repository remains unchanged.
Checks do not exclude external writers, and failure after native application can
leave a failed isolated candidate for owner cleanup. No atomic transaction is
claimed.

The adapter is a host API. The current worker source socket remains read-only;
this does not yet expose compact edits as an authenticated model tool or enable
them in a live Terminal-Bench arm. Source fidelity and local application tests
do not establish provider-token savings.

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
new model assets. A separate [current-owner source read](current_source_lookup.md)
now checks original bytes through native task custody and the working filesystem.
Its host-issued CLI remains a separate typed-owner route. The
[coding-worker adapter](worker_source_views.md) instead uses the task's already
verified semantic source scope and the existing isolated child lifecycle. Its
working-candidate reader now refreshes compact views after original-file edits,
with generation-bound exact expansion and stale-continuation refusal. The
[candidate context reader](worker_candidate_context.md) now rebuilds advisory
capsules and numerical formulas for the captured working bytes. Accepted formal
index publication and model-facing compact edit submission remain subsequent
integration stages; ordinary coding edits still use exact original files.

The [qualification evidence](evidence/reversible-source-views-20261012/README.md)
retains 40 passing cases, including a separate-process unchanged inverse without
the original checkout. Source hashes, exact commands, initial failure and final
corrected fixture are preserved. Rebuild and qualify a current worker archive
before operational deployment; this host API is not a live benchmark result.

The [mapped-edit qualification](evidence/compact-source-edit-20261012/README.md)
retains 339 passing regressions and a public-source fixture with 11 authored
numerical comparisons. It qualifies host mapping, native application and formal
context invalidation; model-tool integration and official benchmark performance
remain unmeasured.
