# CVEfixes GraphRAG → Security IR → Agent Supervisor

Status: approved implementation program

Program ID: `CVESIR`

Source dataset: `hitoshura25/cvefixes`

Pinned source revision: `d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2`

Proposed derived dataset: `sofiyapervane/cvefixes-security-ir-graphrag`

Objective heap: `docs/architecture/cvefixes_security_ir.objectives.md`
Task board: `docs/architecture/cvefixes_security_ir.todo.md`

## 1. Outcome

Build a reproducible, content-addressed pipeline that:

1. reads the pinned CVEfixes Parquet snapshot without executing source content;
2. projects vulnerable/fixed code pairs into a typed security knowledge graph;
3. produces candidate forbidden-action policies and formal Security IR views;
4. evaluates the candidates against held-out vulnerable and fixed code;
5. packages and publishes a derived Hugging Face dataset;
6. loads a pinned derived release into `ipfs_datasets_py.logic.security_ir`; and
7. makes the `ipfs_accelerate_py` agent supervisor check both Intent IR and
   generated-code facts against the pinned Security IR before plan admission,
   execution permits, and merge completion.

The end-to-end flow is:

```text
hitoshura25/cvefixes@d4f5c4e
  ├─ source snapshot + row/body identities
  ├─ vulnerable/fixed semantic projection
  ├─ GraphRAG nodes + typed edges + bounded indexes
  ├─ candidate forbidden predicates + SecurityIR declarations
  ├─ repository-isolated evaluation and review state
  └─ reproducible Hugging Face release
         │
         ▼ pinned revision + manifest root
ipfs_datasets_py.logic.security_ir.cvefixes
         │
         ├─ exact SecurityAuthorizationRequest from Intent IR
         ├─ exact SecurityAuthorizationRequest from generated diff facts
         └─ deny / allow / unknown receipts
                    │
                    ▼
ipfs_accelerate_py agent supervisor
  plan admission → pre-execution permit → post-generation gate → merge gate
```

## 2. Ground truth and constraints

The source has one `default/train` split with 12,987 rows and 23 columns. The
Dataset Viewer reports approximately 1.23 GB of Parquet across three shards.
Relevant columns are:

- identity and provenance: `cve_id`, `hash`, `repo_url`, `published_date`,
  `commit_date`, `version_tag`;
- classification: `severity`, `cwe_id`, `cwe_name`, `cwe_description`,
  `security_keywords`, `language`;
- change location: `file_paths`, `diff_stats`;
- source material: `cve_description`, `commit_message`,
  `diff_with_context`, `vulnerable_code`, `fixed_code`.

All source strings and code are inert, untrusted data. They must never be
concatenated into an agent system prompt as instructions, executed, imported,
or used to grant authority.

The derived public release defaults to metadata, graph records, formal
candidates, hashes, and bounded context excerpts. Republishing complete
third-party source bodies is a separate opt-in release profile because the
upstream repositories have licenses independent of the dataset card.

## 3. Trust and authority model

The implementation must preserve these levels:

| Level | Meaning | May deny/allow execution? |
|---|---|---|
| `observed_candidate` | Deterministic or model-assisted projection from a CVE row | No |
| `validated_candidate` | Passes vulnerable-positive, fixed-negative, provenance, and schema gates | No |
| `reviewed_pattern` | Human-reviewed generalization with bounded scope | Only through an explicit operator meta-policy |
| `authoritative_policy` | Signed/pinned Security IR policy admitted by the existing authority path | Yes |

Graph retrieval, embedding similarity, LLM output, CWE text, vulnerable code,
fixed code, and a formalized formula are evidence or candidates, not proof.
They never mint an allow. A deployment may adopt an authoritative
`block_on_validated_candidate` meta-policy; in that case the meta-policy, not
the candidate, supplies the denial authority.

Security evaluation is deny-overrides. In enforcement mode, a relevant
`unknown`, stale root, incomplete authoritative scan, unsupported language,
ambiguous action projection, or unresolved conflict rejects admission.

## 4. Derived dataset contract

The release uses logical configurations/tables that can be stored as
uniformly compressed Parquet:

| Config/table | Primary identity | Purpose |
|---|---|---|
| `source_records` | source row CID | Pinned source metadata and body hashes |
| `code_units` | code-unit CID | File/hunk/symbol-level vulnerable and fixed units |
| `graph_nodes` | node CID | CVE, CWE, repo, commit, file, symbol, pattern, action, effect, policy |
| `graph_edges` | edge CID | Typed, directed, provenance-bound relations |
| `policy_candidates` | candidate CID | Scoped forbidden predicates and Security IR projection |
| `formal_views` | formula/artifact CID | Deontic, transition, threat, and claim views |
| `evaluations` | evaluation receipt CID | Positive/negative, determinism, review, and diagnostics |
| `release_manifest` | manifest root CID | Shards, schema/config versions, source revision, split policy |

Every row includes:

- schema and producer versions;
- canonical CID and SHA-256 identity;
- source repository, commit, CVE, source row, and source revision references;
- extraction method, configuration digest, confidence, diagnostics, and review
  state;
- split and grouping keys;
- parent CIDs for every projection;
- an explicit `grants_execution_authority = false` unless it is a separately
  admitted authoritative policy record.

Large code bodies are referenced by CID and hash. A body sidecar may be built
for internal evaluation, but the default public release contains only bounded,
license-policy-approved excerpts.

## 5. GraphRAG ontology

Required node kinds:

- `CVE`, `CWE`, `Repository`, `Commit`, `File`, `Language`, `Symbol`;
- `VulnerableCodeUnit`, `FixedCodeUnit`, `ChangeHunk`;
- `VulnerabilityPattern`, `FixPattern`, `SecurityPrecondition`,
  `ForbiddenAction`, `SecurityEffect`, `Mitigation`;
- `SecurityPolicyCandidate`, `FormalView`, `EvaluationReceipt`.

Required edge kinds:

- `AFFECTS`, `CLASSIFIED_AS`, `FIXED_BY`, `CHANGES`, `CONTAINS`;
- `HAS_VULNERABLE_FORM`, `HAS_FIXED_FORM`, `INTRODUCES_RISK`,
  `REMOVES_RISK`, `MITIGATES`;
- `REQUIRES_PRECONDITION`, `MAY_CAUSE_EFFECT`, `FORBIDS`;
- `SAME_PATTERN_AS`, `EVALUATED_BY`, `DERIVED_FROM`, `SUPERSEDES`.

Edges are typed and directionally validated. Similarity edges retain the
index/model/config identities and score, and cannot be consumed as proof
edges.

## 6. Projection and formalization

Projection works at `(source revision, row index, file path, hunk, symbol)`
granularity where the source permits it. It must:

1. parse serialized CVE descriptions safely;
2. normalize CWE, severity, language, repository, commit, and file identities;
3. split diffs without losing vulnerable/fixed pairing;
4. use deterministic language adapters first, with an explicit unsupported
   result rather than a fabricated parse;
5. derive candidate preconditions, actions, effects, and mitigations;
6. canonicalize equivalent predicates without merging scopes that differ;
7. retain both positive evidence (vulnerable form) and negative evidence
   (fixed form); and
8. record model-assisted extractions separately from deterministic facts.

The Security IR adapter maps reviewed candidate records to:

- `SecuritySource` for pinned row/body/commit references;
- `Resource`, `Asset`, `Channel`, and `ThreatAssumption` where grounded;
- `Policy(effect=DENY)` with exact scope attributes;
- `SecurityClaim` describing the expected invariant;
- optional state machines when the vulnerable/fixed pair proves a meaningful
  transition vocabulary.

The formalization adapter emits deontic prohibitions and solver-neutral proof
obligations through the existing Security IR formalization views. An example
shape is:

```text
F(action=construct_path_from_untrusted_input
  | target=filesystem
  ∧ missing=canonicalize_and_confine
  ∧ effect=read_or_write_outside_allowed_root)
```

The concrete stored form must use typed symbols and stable IDs rather than
treating this presentation syntax as an executable formula.

## 7. Dataset splits and evaluation

Splits are group-isolated by repository and, where necessary, by CVE/fix
lineage. The same repository, commit, near-duplicate hunk, or normalized code
body may not cross train/validation/test boundaries.

Minimum gates:

- 100% manifest/shard/hash validation;
- deterministic rebuild identities for a fixed source/config/toolchain;
- 100% source and parent-CID traceability;
- zero split leakage under repository, commit, body-hash, and near-duplicate
  checks;
- vulnerable-positive and fixed-negative measurements by CWE and language;
- no fixed sample labeled forbidden merely because it shares a CVE;
- explicit unsupported/unknown coverage, never silent row loss;
- calibration and abstention metrics for model-assisted extraction;
- adversarial tests for prompt injection in descriptions, messages, diffs,
  paths, and code;
- canonical Security IR round-trip and formalization diagnostics;
- proof that no candidate/retrieval/evaluation row grants authority.

Initial release gates are reported rather than invented. Automatic policy
promotion is disabled until per-CWE thresholds are reviewed. A recommended
starting promotion threshold is precision ≥ 0.98 on vulnerable units, false
positive rate ≤ 0.01 on fixed units, at least 25 independent repositories, and
no critical provenance or leakage failures.

## 8. Hugging Face release

The release builder:

1. pins and verifies the source revision and Parquet shard hashes;
2. writes deterministic, bounded Parquet shards;
3. emits a Croissant-compatible dataset card and schema documentation;
4. emits the release manifest, build config, evaluation summary, query client,
   and example loader;
5. validates locally without credentials;
6. uploads only the validated staging directory; and
7. resolves the resulting Hub commit and records it in a publication receipt.

Target repository: `sofiyapervane/cvefixes-security-ir-graphrag`.

Publication is idempotent by `(target repo, source revision, release root CID)`.
If that tuple already exists, the publisher verifies it instead of creating a
second release. Tokens are read from the environment/keyring and never written
to artifacts or logs.

## 9. `ipfs_datasets_py` integration

The new `logic.security_ir.cvefixes` package owns:

- source snapshot and schema contracts;
- GraphRAG projection, graph, and retrieval;
- candidate evaluation and release packaging;
- the CVE/CWE security vocabulary;
- conversion into the existing immutable `SecurityIR`;
- a pinned Hugging Face source adapter that verifies repo, revision, manifest,
  shard hashes, and declared row identities;
- bounded lookup from exact code/action facts to candidate/reviewed policies.

It extends `security_ir.model`, `security_ir.formalization_adapter`, and shared
`ir_core` identities. It must not introduce another Security IR schema,
authority model, result family, or proof cache.

## 10. Supervisor enforcement

The supervisor evaluates two independent fact streams:

1. **Intent facts**: actions, tools, targets, data flow, expected effects,
   authority, and state derived from pinned Intent IR.
2. **Code facts**: changed AST/symbol/data-flow/capability facts derived from
   generated output and its diff, including effects not declared by intent.

Each fact becomes an exact `SecurityAuthorizationRequest` bound to the current
Security IR root. The existing constraint compiler and decision runtime must:

- reject intent/security root mismatch or staleness;
- reject any authoritative deny;
- reject conflicts and unknowns in enforcement mode;
- reject undeclared code effects even when the intent request passed;
- preserve intent conformance as a constraint, never an authorization grant;
- require a fresh decision before an execution permit;
- repeat the check on generated output before merge;
- re-check against the merged tree and current Security IR root;
- emit bounded receipts with matching policy IDs, source CIDs, reason codes,
  and counterexample facts.

No fuzzy GraphRAG query sits on the mutation boundary. Retrieval nominates a
bounded policy set; exact typed matching and the existing authority contracts
make the decision.

## 11. Goals, task waves, and parallel ownership

The machine-ingestible details live in the objective heap. Expected execution:

| Wave | Parallel work | Gate |
|---|---|---|
| 0 | source pinning, schema, release policy, CVE vocabulary, code-fact contract | independent |
| 1 | semantic projector, Security IR adapter, code extractor | Wave 0 contracts |
| 2 | graph builder, formal views, HF resolver | projection/adapters |
| 3 | retrieval, evaluation, intent/code comparator | graph/formal contracts |
| 4 | release builder/publisher, decision-runtime integration, receipts | quality and comparator |
| 5 | end-to-end enforcement, Hub publication receipt, rollout | all required gates |

Bundles own disjoint file families. Shared existing files
`security_constraint_adapter.py`, `ir_constraint_compiler.py`, and
`execution_permit.py` are edited only by the enforcement bundle after its
dependencies land.

## 12. Definition of done

The program is complete only when:

- the source revision and every derived artifact are content-addressed;
- local release validation and held-out evaluation pass;
- the derived dataset exists at a pinned Hub commit with a verified receipt;
- `ipfs_datasets_py` loads that exact release into canonical Security IR;
- candidate versus authoritative policy states remain distinguishable;
- intent-only, code-only, deny, conflict, unknown, stale-root, prompt-injection,
  fixed-negative, and allowed controls pass end-to-end tests;
- no execution permit or merge can bypass the security decision;
- operator docs include pin, build, publish, update, rollback, and incident
  procedures; and
- the supervisor objective graph has no open schedulable implementation task
  and the external publication goal has a current receipt.

## 13. Operator commands

Generate tasks and bundle shards:

```bash
PYTHONPATH=. python -m ipfs_accelerate_py.agent_supervisor.objective_daemon \
  --repo-root . \
  --objective-path docs/architecture/cvefixes_security_ir.objectives.md \
  --todo-path docs/architecture/cvefixes_security_ir.todo.md \
  --discovery-dir data/agent_supervisor/cvefixes_security_ir/discovery \
  --bundle-dir data/agent_supervisor/cvefixes_security_ir/bundles \
  --dataset-dir data/agent_supervisor/cvefixes_security_ir/datasets \
  --graph-path data/agent_supervisor/cvefixes_security_ir/objective_graph.json \
  --task-prefix CVESIR- \
  --max-findings 32 \
  --surplus-findings-per-goal 1 \
  --scan-exclude-path docs/architecture/CVEFIXES_SECURITY_IR_GRAPHRAG_PLAN.md
```

Plan lanes without starting:

```bash
PYTHONPATH=. python -m ipfs_accelerate_py.agent_supervisor.bundle_supervisor \
  --bundle-index-path data/agent_supervisor/cvefixes_security_ir/bundles/index.json \
  --repo-root . \
  --state-root data/agent_supervisor/cvefixes_security_ir/runtime \
  --worktree-root data/agent_supervisor/cvefixes_security_ir/runtime/worktrees \
  --log-dir data/agent_supervisor/cvefixes_security_ir/runtime/logs \
  --manifest-path data/agent_supervisor/cvefixes_security_ir/runtime/lane_manifest.json \
  --metrics-path data/agent_supervisor/cvefixes_security_ir/runtime/scheduler_metrics.json \
  --task-prefix '## CVESIR-' \
  --worktree-submodule-path ipfs_datasets_py \
  --max-lanes 4 \
  --allow-missing-provider-telemetry \
  --once
```

Add `--start --implement`, an explicit `--implementation-command`, and
`--merge-target-branch agent/cvefixes-security-ir` to launch one bounded
reconciliation cycle. The lane supervisors continue from their state
directories. Keep objective refill disabled: `CVESIR-G000` is a tracking root,
not an executable task.
