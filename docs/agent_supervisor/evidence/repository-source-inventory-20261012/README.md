# Retained source inventory qualification — 2026-10-12

The [implementation](../../../../ipfs_accelerate_py/agent_supervisor/runtime/repository_source_inventory.py)
retains the complete native committed Git inventory and exact frames in a
caller-supplied existing datasets `ImmutableCAS`. It publishes a closed root
after validating content and source, then replays the stored content again and
performs final source/store/input checks. The
[guide](../../repository_source_inventory.md) documents bounds and usage.

Cold replay validates the native canonical manifest DAG and population, every
frame and whole source CID, and the Git blob object hash. It returns historical
bytes and Git modes. Symlink targets stay data; Gitlinks are explicit child
commit references without child-repository acquisition. It performs no target
execution, filesystem restoration, proof admission or coding-model disclosure.

## Tests

| Run | Passing distinct cases | Evidence |
| --- | ---: | --- |
| New integration, final source | 28 | [Commands/hashes](authored-final-02-commands.json), [node IDs](authored-final-02-nodeids.json), [log](authored-final-02-tests.log), [JUnit](authored-final-02-junit.xml) |
| Existing reconstruction | 26 | [Command](reconstruction-command.json), [node IDs](reconstruction-nodeids.json), [log](reconstruction-tests.log), [JUnit](reconstruction-junit.xml) |

These **54 distinct tests** passed without skips, with disjoint node populations.
The authored suite's before/after production and test-file hashes match.
Both pytest processes used normal AST sealing and separate temporary seal and
orchestration paths. Source qualification used immutable datasets
`c18390b350c7ab40f5bd342578204a03dc34fe05`, preserving the concurrent LegalIR
storage-recovery integration. The native source/content/parser/CAS modules are
unchanged from the preceding `c1b1f65` pin.

New controls cover mixed CRLF/LF, no final newline, Unicode, non-UTF-8 content,
binary/empty data, duplicate blobs, normally filtered tracked paths, executable
modes, real Git SHA-1/SHA-256 formats, source/store/request/limit drift, corruption,
independent pins, rehashed omissions and duplicate pages, early work budgets and
bounded metadata pages. A fresh process loads retained bytes after deleting the
authored original checkout, while Git/source acquisition and new-owner creation
are prohibited after explicit store admission. The output remains historical.

The initial [collection log](initial-collection.log) rendered pytest's verbose
tree. The evidence harness expected flat node IDs and stopped before running
tests. The corrected [collection](collection.log) uses explicit quiet output;
the unchanged implementation and test source then passed. Both logs are retained.
This was an evidence-format correction, not a production test failure.

The [original passing run](authored-commands.json) is also retained. A final
source lint check identified one extra blank line at the test file's end.
Removing that newline preserved the [AST including positions](authored-final-02-ast-comparison.json).
The same 28 cases passed again against final published source; repetitions are
not added to the distinct count. Production source remained unchanged.

The [qualification](qualification.json) records scope and counts. The
[source manifest](source-manifest.json) binds the new code, test and native reused
files. Test log bytes are retained exactly; artifact-local `.gitattributes`
suppresses whitespace lint for those immutable logs only.

## Limits

Native export rehashes an entire blob per frame, with explicit predicted work
bounds checked before acquisition. Replay consumes every declared chunk
occurrence. No efficient random-access or complete-run token improvement is
established. This changes no model weights, embeddings, inference or training.
No provider/tokenizer/prover call or full Terminal Bench run occurred.

CAS/source checks are sequential, without a mutation lease, rollback or permanent
availability guarantee. Failures can leave immutable artifacts but return no
successful capture. Reload revalidates content; current workspace equality needs
its own source/owner admission. The metadata pages are host data, with no new
Quack operation, registered coding-model expansion tool or omission authority.
File writing, compact source transforms and mapped edits remain later work.
