# Proof-context CLI

The proof-context CLI is available before console-script packaging through one stable module entrypoint. From a clean checkout, run commands from `external/ipfs_accelerate`:

```bash
python -m ipfs_accelerate_py.proof_context.cli --help
```

Every command requires an explicit repository, task, and correlation identity; the current directory is never interpreted as the repository. Machine output is the default. Use `--output-mode human` for the concise operator rendering, or `--human-report` to append the bounded PCCE-043 patch report.

```bash
python -m ipfs_accelerate_py.proof_context.cli init --repository /tmp/proof-context-demo --task demo-1 --correlation trace-1
python -m ipfs_accelerate_py.proof_context.cli status --repository /tmp/proof-context-demo --task demo-1 --correlation trace-1 --output-mode human
```

## Commands and input files

The discoverable command set is `init`, `scan`, `status`, `plan`, `run`, `verify`, `resume`, `expand-context`, `explain-impact`, `assurance`, `seal`, and `report`. State commands have no input files. `run` requires `--run-id` and a `--request` JSON file with schema `ipfs-accelerate.proof-context.v0.1/cli-run-request@1`. It contains admitted `task`, `context_pack`, `route`, and `adapter` records. For external patches, adapter options contain one `patch_file` (relative paths resolve relative to the request file) or `patch_base64`, plus exact `declared_files`.

```bash
python -m ipfs_accelerate_py.proof_context.cli run --repository /tmp/proof-context-demo --task demo-1 --correlation trace-1 --run-id run-1 --request request.json
```

Replay requests use adapter name `replay`, a JSON array of full replay fixture records, and exact `selected_fixture_cid` and `selected_response_artifact_cid` selectors. Replay remains permanently marked as replayed evidence, so it never silently becomes live production success. Malformed patches, unadmitted replay fixtures, wrong selectors, wrong run records, and wrong evidence parents return typed nonzero results.

`verify` accepts exactly one of `--run-record` or `--patch-id`. `resume` optionally accepts an identity-bound `--checkpoint`; repeating the same resume is idempotent. Evidence commands require `--run-id`, `--repository-id`, `--patch-id`, and an immutable `--parent` record. The parent must have the expected prior operation, matching identities, live success provenance, and the current repository head.

Exit codes are: `0` live success, `1` typed failure, `2` invalid input, `3` rejected, `4` simulated/replayed result, `5` unavailable capability, and `6` stale evidence. The JSON envelope always includes its schema, status, exit code, correlation, identities, provenance, and any observed artifact CID.
