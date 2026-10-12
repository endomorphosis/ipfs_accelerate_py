# Current source lookup qualification

**44 distinct cases passed**, zero failures/errors/skips: 29 source-owner,
four child CLI, seven wire controls and four current owner regressions.
The [final run](corrected-controls-commands.json) retained normal AST sealing
and isolated per-process state; its source hashes match before and after.
[Collected nodes](corrected-controls-nodeids.json), [raw log](corrected-controls-tests.log),
[JUnit](corrected-controls-junit.xml) and [qualification](qualification.json)
bind the actual outcome. The test phase took 78.22 seconds, not a performance
comparison against a deployed reader.

Failed runs remain separately bound. The [first collection](initial-commands.json)
selected a legacy test importing a symbol already removed in base main; no tests
ran. The [first current suite](current-suite-commands.json) caught a product
generation check iterating `DuckDBRow` column names. It now compares indexed
values, preserving the exact fence. The [next run](indexed-rows-commands.json)
passed 19 cases and stopped on the CRLF fixture's stale Git index stat, before
the source read. Both conversion controls now refresh the same filtered blob
and assert unchanged HEAD/tree/blob identities with genuinely clean status.
No product change was needed for that fixture correction. The final run uses
the same 44 node IDs; repetitions are not added to the passing count.

This bundle qualifies the [bounded exact-source reader](../../current_source_lookup.md),
its attachment to the existing typed owner and the host-issued subprocess CLI.
It retains exact commands, collected node IDs, raw output/JUnit, source hashes
and independent review. Actual local DuckDB and typed Unix sockets are used;
the hermetic `FakeQuackTransport` fixture does not qualify native Quack extension
networking. No provider, model, training, embedding or prover session is run.

The profile admits selected clean committed regular files with native task/body
custody and exact working-file equality. Checks are sequential without a writer
lease. Each slice still cold-replays the full historical inventory and reads the
selected file in full; no speed or total-token saving is established.

The CLI is a host-issued child interface. Current coding containers do not yet
receive a qualified session/descendant channel, and required facts remain inline.
Dirty overlays, capsule and residual-obligation operations, mapped compact edits
and a rebuilt live coding archive remain separate qualification stages.

Historical source/view/model assets remain unchanged. The qualification does
not alter the 8D/384D/768D or family/schema/version/decoder-task separation.

Parallel main was merged before publication, including the
[numerical obligation retrieval](../numerical-obligation-retrieval-20261012/README.md)
and legal storage/training closeout. That owner API recovered all 45 pending
rows, with isolated model access still unqualified. Its 263-test result is a
separate qualification and is not added to this suite's 44 cases.
The source reader was tested against datasets `c18390b3`; the parent retains
the newer `f47161b7` pin. Their complete `logic/software_contracts` trees are
identical, and all selected datasets source hashes remain exact. Publication
checks retain these joins without rerunning unchanged source tests.
