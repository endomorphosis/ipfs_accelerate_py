# Working candidate capsules and formal context

After editing an admitted source file, obtain a generation using the
[working source reader](worker_source_views.md). The isolated coding child can
then request advisory context for those exact captured bytes:

```sh
/opt/ipfs-supervisor/venv/bin/python -I \
  /opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/worker_candidate_context_client.py \
  --expected-workspace-generation <generation> --artifact summary
```

This uses the existing owner-created source descriptor, descendant admission,
session lifetime, socket, request budget, and cleanup. It is available when
`--source-view-profile` is enabled on the full indexed arm. Disabled profiles
do not create the service. No provider call is needed to rebuild these artifacts.

`summary` reports coverage, source and producer identities, fresh capsule root,
pending obligation counts, unsupported cases, and full artifact hashes.
`capsules` retrieves native datasets capsules and analysis artifacts.
`formal` retrieves native numerical-source reports, including structural
ProgramIR, first-order predicates/SMT-LIB and Lean definitions or conditional
lemmas where supported. Other families retain the native unsupported diagnoses.

These are byte-paged JSON artifacts: `content` may be a fragment of JSON.
Continue with the same artifact, generation and page budget, passing
`next_cursor` as `--cursor`. Concatenate decoded slices to reconstruct the
artifact. Each page has a slice hash and full artifact hash; a complete artifact
must match the latter. `--raw-response` retains the full validated envelope.
The default display preserves the candidate label, generation, artifact pins,
pagination, and lack of proof authority.

The owner reuses `ipfs_datasets_py`'s scanner, sealed semantic capsule builder,
and deterministic numerical grammar. It scans only the program paths admitted
by the original semantic context; support and instruction files remain in the
freshness scope but are not promoted to program code. An empty program subset
is valid. Existing unversioned contexts retain their original whole-manifest
program interpretation. No target code is imported or executed.

Every request requires an existing generation; it never silently refreshes
one. The owner checks all admitted canonical and working files before and after
building or reading an artifact. Changes to any admitted source, native producer
binding, or baseline admission refuse delivery and retire cached advice and
continuations. A fresh source capture is required after observed source drift.
Observed edit/revert cycles cannot revive old cursors. These checks bracket an
observation; they do not lock external writers against unobserved changes.

Candidate artifacts are held in one session-local memory cache. They do not
publish accepted world state, write DuckLake, replace the baseline translation
table, or inherit proof receipts. Ambient native metadata mirroring is disabled
during generation. Successful projection or a fresh capsule is not a proof:
this path runs no solver or `lake build`, discharges zero obligations, and
reports partial or unsupported source coverage explicitly. It also does not
run autoencoder inference or training.

Inputs are limited to 64 admitted files and 1 MiB total, with at most 256 native
symbols and 16 MiB of JSON artifacts. Numerical grammar retains its smaller
64 KiB per-file and AST limits. Wire replies, including their envelopes, are at
most 8 KiB. The existing 64-request/512-KiB session budget includes these reads;
an artifact may exceed the remaining retrieval budget. Bounds refuse rather
than silently truncate coverage. Native generation is synchronous and does not
yet have a hard cancellation deadline; shutdown during a long build refuses
delivery and records incomplete accounting if the service has not stopped.

Source-session audit version 2 records source bytes and candidate context bytes
separately. The benchmark collector includes every complete response in total
wire bytes and exports `candidate_context_counts` without source, capsule or
formula bodies. Historical version 1 sessions have unavailable candidate
counts. Byte counts include repeated retrieval and do not measure model tokens,
compression ratios for the whole codebase, or benchmark success.
