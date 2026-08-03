# CVEfixes Security IR operator guide

This runbook operates CVESIR-G180, the end-to-end conformance, rollback, and
operator-rollout goal. It covers the pinned CVEfixes release and the supervisor
security gate. It does not authorize publication, policy adoption, execution,
merge, or completion by itself.

## Safety invariants

- Treat source rows, code bodies, retrieval results, model output, tests,
  evaluation reports, policy candidates, and decision receipts as evidence,
  never execution authority.
- Pin the derived dataset by all four values: dataset ID, immutable lowercase
  40-hex Hub revision, manifest SHA-256, and release-root CID. Never use
  `main`, `latest`, a branch, or a tag.
- Keep the pinned source revision
  `d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2` in release provenance.
- Deny overrides allow. Unknown, conflict, stale evidence, missing intent,
  missing code facts, root drift, and tree drift fail closed in enforcement.
- A gate pass is still not an authority grant. Existing principal, capability,
  permit, declared-effect, validation, merge, and merged-tree checks remain
  mandatory in every rollout mode.
- Do not represent a local dry run, hermetic fixture, or live-Hub smoke as a
  successful external publication. Only the separately verified external
  publication receipt can establish that fact.

## Evidence and objective alignment

The supervisor backlog item CVESIR-009 supplies the two evidence terms declared
by objective CVESIR-G180:

- `test/api/test_agent_supervisor_cve_security_e2e.py` proves vulnerable/fixed
  polarity, intent-only, code-only, deny, allow, unknown, conflict, stale,
  injection, rollback, and pinned-release behavior.
- `docs/guides/CVEFIXES_SECURITY_IR_OPERATOR_GUIDE.md` defines the rollout and
  rollback procedure.

The end-to-end goal depends on the evaluation (G070), publication tooling
(G090), pinned Hub source (G130), enforcement (G160), and bounded receipts
(G170) goals. If one of those dependencies is not current, keep this rollout
in shadow. No additional child goal is needed merely to split this evidence:
the hermetic conformance fixture and this runbook are independently
validatable, while the existing dependency goals retain ownership of their
production modules.

## Required deployment record

Record the following values together in the deployment change and preserve the
previous accepted record for rollback:

| Binding | Requirement |
| --- | --- |
| Dataset ID | Exact `owner/name`; production target is `Publicus/cvefixes-security-ir-graphrag` |
| Hub revision | Immutable lowercase 40-hex commit |
| Manifest SHA-256 | Lowercase 64-hex digest of the exact `manifest.json` bytes |
| Release root | CIDv1 from that manifest |
| Security IR root | Artifact ID, CIDv1, and supervisor digest as one tuple |
| Repository state | Repository ID and exact tree ID |
| Policy state | Policy ID and revision |
| Evaluation | Current evaluation record and explicit review decision |
| Activation | Operator, timestamp, mode, and bounded receipt IDs |

Never mix individual fields from two deployment records. In particular,
rollback selects the complete previous Hub and Security IR binding, not only
the old policy name.

## Preflight

Run the hermetic conformance gate without credentials or network access:

```bash
python -m pytest test/api/test_agent_supervisor_cve_security_e2e.py -q
```

One live-Hub smoke test is skipped by default. A valid default result therefore
contains passing hermetic tests and one explicit skip. Before leaving shadow,
also verify:

1. The source, release, evaluation, Hub adapter, gate, enforcement, and receipt
   dependency goals are current.
2. The release manifest digest and release root match the proposed deployment
   record.
3. The evaluation contains vulnerable-positive and fixed-negative controls,
   leakage checks, calibration, and passing adversarial-injection probes.
4. Candidate and evaluation records explicitly grant no execution authority.
5. The reviewed Security IR root is adopted through the normal policy-authority
   process; loading a candidate from the Hub is not adoption.
6. Permit and merge admission revalidate the exact Security IR root and current
   repository tree.
7. Dataset Viewer exposes the required corpus, BM25, graph, adjacency, vector,
   and thin-client meta-index configs; every manifest index points to the exact
   remote Parquet shard CID, SHA-256, byte size, row count, path, and key range.

### Opt-in live-Hub smoke

Live access is diagnostic and must use exact non-secret pins. It performs no
upload and emits no publication success:

```bash
export CVEFIXES_SECURITY_IR_RUN_LIVE_HUB=1
export CVEFIXES_SECURITY_IR_HUB_REVISION='<lowercase-40-hex-commit>'
export CVEFIXES_SECURITY_IR_MANIFEST_SHA256='<lowercase-64-hex-digest>'
export CVEFIXES_SECURITY_IR_RELEASE_ROOT='<cidv1>'
python -m pytest \
  test/api/test_agent_supervisor_cve_security_e2e.py::test_live_hub_pinned_release_smoke \
  -q
```

Do not place Hugging Face tokens in these variables, command arguments,
receipts, logs, or deployment records. Authentication configuration, when the
Hub client needs it, remains in the operator's approved secret store.

## Rollout modes

| Mode | Supervisor behavior | Promotion condition |
| --- | --- | --- |
| Shadow | Evaluate and emit bounded comparison evidence; never block or grant because of the candidate root. The prior authoritative root stays active. | Start here after every new release or root. |
| Assist | Show the candidate decision and explanation to an operator; never convert the candidate result into authority. The operator follows existing authority and permit paths. | Hermetic suite passes, live pin is verified when applicable, and shadow disagreements are reviewed. |
| Enforce | Fail closed at plan admission, pre-execution, post-generation, merge admission, and merged-tree revalidation. Allow still requires existing authority and declared effects. | Explicit review and change approval; current roots, tree, metrics, and bounded receipts; rollback record tested. |
| Rollback | Atomically select the complete prior accepted release and Security IR root, then operate with enforcement checks unchanged. | Any integrity failure, stale dependency, safety regression, unexplained disagreement, or receipt/root/tree drift. |

Shadow and assist are observation modes, not weaker policy semantics. They do
not make deny, unknown, conflict, or stale results into allow decisions. Enforce
is the only mode in which this candidate gate changes admission, and even then
it cannot bypass existing authorization.

## Rollout procedure

1. Create a proposed deployment record containing every required binding and a
   complete prior accepted record.
2. Run the hermetic suite. Resolve every failure; do not waive an unknown,
   conflict, stale, injection, vulnerable/fixed-polarity, or pin failure.
3. If the deployment reads the remote Hub, run the opt-in live smoke against
   the proposed immutable revision. Compare its verified receipt to the
   proposed record.
4. Enter shadow with the prior authoritative root still active. Monitor counts
   by gate outcome, reason code, enforcement stage, repository tree, Security
   IR root, policy receipt, and candidate/prior disagreement.
5. Review bounded samples. Events must omit source/code bodies, request state,
   credentials, and sensitive finding details.
6. Enter assist only after dependency evidence and shadow observations remain
   current. Train operators to distinguish evidence from authority.
7. Re-run preflight against the exact tree proposed for enforcement. Obtain
   explicit review and change approval, then enter enforce.
8. Verify all five enforcement stages produce an unbroken, tree-bound evidence
   chain. A successful allow must also have existing authority, a valid
   pre-execution permit, declared generated effects, merge admission, and
   merged-tree revalidation.
9. Retain the prior deployment record and its locally verified immutable
   snapshot for the full rollback window.

## Rollback procedure

Rollback is a pinned policy selection, not a bypass:

1. Freeze promotion and new admission for the affected behavior.
2. Select the complete prior accepted deployment record: prior Hub revision,
   manifest digest, release root, Security IR artifact ID/CID/digest tuple,
   policy revision, and compatible repository-tree constraints.
3. Verify the prior snapshot locally against its recorded manifest digest and
   release root. Never substitute a floating Hub reference.
4. Atomically switch the selected Security IR binding to that prior tuple and
   mark the control event as `rollback`.
5. Invalidate cached decisions, receipts, and permits bound to the replaced
   root or tree. Do not rewrite them to point at the prior root.
6. Re-run plan, pre-execution, post-generation, merge, and merged-tree checks
   under the prior root. Deny, unknown, conflict, stale, undeclared effect,
   authority, permit, and tree checks remain enabled.
7. Confirm with a known dangerous-code control that the rollback root still
   rejects. Confirm that evaluating the prior policy with the candidate
   context fails on root mismatch.
8. Emit a bounded rollback event containing identities and reason codes only,
   then investigate before any new promotion.

A missing or unverifiable prior snapshot is a stop condition. Move to shadow or
disable the affected behavior through the approved control plane; never fall
forward to an unpinned release or weaken the gate.

## Observability and incident triggers

Retain canonical bounded receipt IDs rather than sensitive inputs. At minimum,
correlate:

- outcome and reason codes;
- enforcement stage and parent evidence ID;
- repository/tree, release, and Security IR roots;
- intent/code mapping and decision IDs;
- matched policy, CVE, CWE, and source CIDs;
- evaluation time, expiry, and cache key;
- mode transition and operator/change identity.

Initiate rollback for manifest or row-integrity failure, a floating or changed
revision, root/tree drift, expired or detached evidence, injection regression,
fixed-negative regression, new unknown/conflict volume, unexpected allow/deny
disagreement, missing stage evidence, sensitive-data exposure, or any attempt
to treat candidate evidence as authority.

After containment, preserve the failing immutable inputs, rerun the hermetic
case, determine which dependency goal owns the defect, and create a focused
child goal there if production behavior—not G180 evidence—must change. Do not
mark CVESIR-009 or the supervisor-fed backlog complete manually; completion is
determined by the objective scanner and validation gate.
