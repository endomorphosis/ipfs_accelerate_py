# PGIR-014 semantic campaign freeze

This directory is the immutable `IRCampaignInputRoot@1` produced for
PGIR-014. It binds exact source snapshots, schema and example-contract code,
corpus/rights/lineage/split artifacts, deterministic compiler and decompiler
configurations, the tokenizer policy, and governing policy roots.

The freeze verifies successfully but its campaign decision is deliberately
`no_go`: all 7,173 source records remain rights-quarantined, the corpus is not
materialized, required holdouts are incomplete, and the historical semantic
baseline is not current-input qualified. Consequently, all 26 transitive
descendant task revisions are recorded with zero lease-eligible tasks. This is
an integrity success and an execution denial, not a failed build.

Run the independent verifier from the repository root:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/freeze/verify_freeze.py
```

Rebuild replay is read-only by default:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/freeze/build_freeze.py
```

`build_freeze.py --initialize` only creates absent artifacts and refuses to
replace different bytes. Any admitted replacement must use a new task revision
and a separately located root whose `previous_root_cid` points to this one.

The earlier `manifest.json` and `result.json` are retained as superseded, write-once
initialization receipts. The `.v2` records are also retained; the `.v3` records are current
bundle/result authorities.

Key artifacts:

- `campaign_input_root.json`: the semantic root and fail-closed decision.
- `descendant_task_revisions.json`: exact revised task CIDs and lease barrier.
- `plan_admission_receipt.json`: a supervisor-compatible typed rejection.
- `verification_receipt.json`: persisted independent-verifier evidence.
- `manifest.v3.json` and `result.v3.json`: bundle and PGIR-014 result identities.
