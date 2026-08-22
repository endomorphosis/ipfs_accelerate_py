# LGCVF-121 and LGCVF-123 authority checklist

This document is a **schema and checklist**. It is not
`external_qualification_receipt.json`, not
`production_authorization_receipt.json`, and not a grant of release or
production authority.

Grok, the agent supervisor, pytest, CID identity, task-state flips, and this
session cannot issue LGCVF-121 or LGCVF-123. The frozen board forbids
self-authoring those receipts. Leaving both tasks blocked is valid.

Current hermetic closeout (122) is `partial`. Completing 121/123 does not
erase missed benchmark thresholds unless a later current evaluation meets
them.

## Current roots packet (bind these, or refuse)

These identities were current when this checklist was written. If any of them
moves, stop and re-bind. Do not authorize a previous tree.

| Binding | Value |
|---|---|
| Formal plan CID | `baguqeerabxn5kkewz44v4chz6vbt3kcozfj4rvhh4gpdp54645blhemhvloq` |
| Predecessor plan CID | `baguqeeraqe65yknsg7gy5vkze76exc3qhe4kn2owecnwa65zg6kaepl7id3q` |
| Qualification result CID | `baguqeeraf3z4orey2z4ziu2ova5qsuf2b5lgwti62pn5prj7u76bv3o4x5uq` |
| Qualification authority CID | `baguqeera27kvd2aqkl5jm4hgisisygd2jukokjvhbm3q4ovmubguuvm6pekq` |
| Qualification checkout fingerprint | `baguqeeraxvbzhhhpmqukfwzntza2klxjirawoxer76vgeny7duln5ne65una` |
| Benchmark report CID | `baguqeera2lz7qz5bjolufndbrayp7tz4p3hsx244zaqhksvcf5m3zezawlwa` |
| Benchmark authority CID | `baguqeerajuq4m4rpsm55yhgzx22ikl2pnjazyvirbb5iiy4dw6lqoe5f2ixa` |
| Benchmark projection CID | `baguqeeraiymgtt7ji4kkscvtp4mr42efbrlu575f2ijsryss4fjoyf5hdj6a` |
| Release report SHA256 | `sha256:28a7222b9769832039f39678b6c41dedc74995cc89ca0dc792c29aab8da24616` |
| Evidence cohort | `hermetic_local_execution` |
| Last committed accelerator HEAD | `4ee047a46b59f94be4aa70de20f14effe7ddad60` |
| Last committed accelerator tree | `663f2766c3f9ff70edb127f7e8d52f2c234d1635` |
| Datasets gitlink / HEAD | `af1d2d76d2cd6332baf8cea50df6b2eb4e988203` |
| Datasets tree | `8f44e00a49d6f67c67ee6810e04e4aae4d869af5` |
| Branch | `agent/logic-governed-compositional-verification-fabric-v1` |

Hermetic 113/120 **excluded** these cohorts: `simulated`,
`live_local_model_execution`, `live_remote_model_execution`,
`production_authoritative_evidence`.

Uncommitted 081–124 source was present when 113 was replayed. A verifier
judging “the current working tree” must hash that dirty forest themselves or
wait until those paths are committed and 113/120/122 are reconstructed.

## LGCVF-121 external qualification

**Issuer.** A human or organization that is not this model, not this
supervisor, and not the author of the candidate test suites. Independence
must be stated in the receipt.

**Output path.**
`data/agent_supervisor/logic_governed_compositional_verification_fabric/external_qualification_receipt.json`

**Must bind**

1. The current plan CID above.
2. The current qualification result CID and checkout fingerprint.
3. The current benchmark report CID.
4. Exact accelerator HEAD/tree and datasets gitlink/tree actually judged.
5. Live local model evidence, or an explicit typed miss for that cohort.
6. Live remote/provider evidence plus a provider-disclosure policy, or a
   typed miss.
7. Production-authoritative evidence, or a typed miss.
8. Qualified multi-writer capability (Quack is not that today), or a typed
   miss.

**Must not**

- Treat 113’s 361 hermetic tests as 121.
- Install missing tools and then claim they were live.
- Network-probe from this hermetic worker and call that live evidence.
- Copy this checklist or the current-roots packet into the receipt path
  and call it issued.
- Set `production_authorized` true. 121 does not grant 123.

**Closed receipt schema (`lgcvf-external-qualification-receipt@1`)**

```json
{
  "schema": "lgcvf-external-qualification-receipt@1",
  "issuer": {
    "identity": "string, legal or organizational name of the verifier",
    "independence_statement": "string, states the issuer is not the model, supervisor, or candidate-suite author",
    "not_model": true,
    "not_supervisor": true,
    "not_candidate_author": true
  },
  "issued_at": "RFC3339 timestamp",
  "expires_at": "RFC3339 timestamp",
  "plan_cid": "baguqeera…",
  "qualification_result_cid": "baguqeera…",
  "benchmark_report_cid": "baguqeera…",
  "source_revisions": {
    "ipfs_accelerate_py": {"head": "git sha", "tree": "git sha"},
    "ipfs_datasets_py": {"head": "git sha", "tree": "git sha", "gitlink": "git sha"}
  },
  "cohorts": {
    "live_local_model_execution": "passed | missed | unavailable",
    "live_remote_model_execution": "passed | missed | unavailable",
    "production_authoritative_evidence": "passed | missed | unavailable"
  },
  "provider_disclosure_policy": "string, what providers were invoked and what was disclosed",
  "multi_writer": {
    "quack_qualified": false,
    "disposition": "passed | missed | unavailable",
    "notes": "string"
  },
  "passed": false,
  "release_qualified": false,
  "production_authorized": false,
  "limitations": ["non-empty unique strings"],
  "receipt_cid": "content identity of the object without this field"
}
```

`python -m json.tool` on that path only checks JSON syntax. It is not
qualification.

## LGCVF-123 operator production authorization

**Issuer.** A named human operator acting in their own name. Not a model,
not a fixture, not a CID, not a supervisor run. Do not reuse LGSWF-006.

**Depends on.** LGCVF-121 **and** LGCVF-122. Today 122 is `partial` and
`not_authorized`. 123 cannot honestly issue while 121 is blocked.

**Output path.**
`data/agent_supervisor/logic_governed_compositional_verification_fabric/production_authorization_receipt.json`

**Must bind**

1. Operator identity and role.
2. Issued-at and expiry.
3. The same plan CID and source HEAD/tree that 121 judged.
4. The 121 receipt CID (current, not stale).
5. The 122 report SHA256 current at authorization time.
6. Scope: this plan and these roots, not “the project in general.”
7. Explicit statement that LGSWF-006 is not reused.

**Must not**

- Authorize a tree 121 did not judge.
- Authorize after expiry of 121.
- Let Grok, pytest, or the supervisor write this file as a pass.
- Treat a passing JSON parse as authorization.

**Closed receipt schema (`lgcvf-production-authorization-receipt@1`)**

```json
{
  "schema": "lgcvf-production-authorization-receipt@1",
  "operator": {
    "identity": "string, named human",
    "role": "string"
  },
  "issued_at": "RFC3339 timestamp",
  "expires_at": "RFC3339 timestamp",
  "plan_cid": "baguqeera…",
  "external_qualification_receipt_cid": "baguqeera…",
  "release_report_sha256": "sha256:…",
  "source_revisions": {
    "ipfs_accelerate_py": {"head": "git sha", "tree": "git sha"},
    "ipfs_datasets_py": {"head": "git sha", "tree": "git sha", "gitlink": "git sha"}
  },
  "scope": "string, this plan and these roots only",
  "lgswf_006_reused": false,
  "depends_on_lgcvf_121": true,
  "depends_on_lgcvf_122": true,
  "production_authorized": false,
  "limitations": ["non-empty unique strings"],
  "receipt_cid": "content identity of the object without this field"
}
```

## Order of operations

1. Independent verifier issues 121 against **current** roots, or refuses.
2. If 121 changes evidence, reconstruct 113/120 and re-issue 122.
3. Operator issues 123 only for those exact roots, with expiry.
4. If you do not want production, stop. Blocked 121/123 plus 124 is an
   allowed closeout.

## Machine copies

JSON Schema drafts (not filled receipts) live next to this file:

- `docs/architecture/lgcvf_external_qualification_receipt.schema.json`
- `docs/architecture/lgcvf_production_authorization_receipt.schema.json`
- `docs/architecture/lgcvf_current_roots_packet.json`
