# Source-view minification audit — 2026-10-12

This is source/design evidence and three authored offline controls, not a new
production minifier, inverse, edit mapper or Terminal Bench result. The
[plan](../../reversible_codebase_views_plan.md) separates exact source restoration,
accepted edit preservation and complete native token measurement.

The [source audit](source-audit.json) binds 13 published source/evidence files at
`ffc6643826e546e5a0b3436b766b22f39f243b71`, following an initial audit of
`1f6d998a92db4024f35b57bbdb183a6626a7fa31`. Parallel ranked benchmark preparation,
accepted successor refresh and the latest instruction census are incorporated.
The untracked architecture plan in the foreign canonical checkout is recorded
by path/hash as design input; that checkout was not edited or staged.

Three retained maintenance snapshots contain identical read-only Python helper
bytes, SHA-256
`0c3ed9348189ed0bf5f45a1c11c24fc3a6acff9c45d9400c853181fe58e3bebd`.
The helper is absent from the audited published runtime. Its
[historical copy](historical_code_view_snapshot.py) is archived here only for
replay; it is not installed as a runtime API. Its comment records contain
offsets, hashes and lengths, without removed text. There is no inverse or edit
mapper. Untouched exact reconstruction requires retained original source bytes.

The [observation script](observe_historical_preview.py) checks the historical
helper digest before loading it and uses only authored source fixtures. The
[exact command](command.json), [JSON output](observation.json) and
[empty stderr](observation.stderr.txt) are retained. Replay from the repository
root with:

```bash
python -B docs/agent_supervisor/evidence/reversible-codebase-views-20261012/observe_historical_preview.py \
  --source docs/agent_supervisor/evidence/reversible-codebase-views-20261012/historical_code_view_snapshot.py
```

| Authored control | Observation |
| --- | --- |
| Large ordinary comment | 3,266 source bytes become 41 preview bytes plus 234 guidance bytes: 275 guided bytes |
| Tiny source | 14 original bytes retained; guided candidate is 240 bytes |
| Edit in retained code | Edited preview parses, but the old view/source pair is stale; no inverse or mapper is invoked |

The large fixture removes one 3,218-byte comment and seven trailing whitespace
bytes. It retains the physical line count and reports the helper's AST/token
structural checks. These observations do not prove semantic equivalence,
restoration, edit correctness or behavior on a real repository. Three controls
are not counted as a production test suite. Provider token counts remain null;
provider and tokenizer calls are zero.

The retained snapshot and observation are enough to reproduce these byte
observations without the historical maintenance directories. The source audit
hashes also distinguish those observations from future runtime implementations.
No model weights, embeddings, training, source under test, credentials or
database owner were modified.

The [qualification](qualification.json) records the observation scope and four
passing local documentation workflow steps. Changed guide paths and anchors
also passed. The [artifact manifest](artifact-manifest.json) binds these files
and both updated guides. No production pytest suite was run for this plan.
