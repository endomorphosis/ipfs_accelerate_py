# Worldwide legal search provider bindings

Campaign implementation uses the canonical explicit agent route exported by `ipfs_accelerate_py.llm_router`. The campaign adapter in `ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search.py` is the only binding that admits that route for this namespace. Generic text generation stays prohibited for agent work.

## Canonical route

| Field | Admitted value |
| --- | --- |
| Router | `ipfs_accelerate_py.llm_router` |
| Primary provider | `grok_cli` |
| Primary model | `grok-4.7` |
| Fallback provider | `codex_cli` |
| Fallback model | Resolve from admitted existing Codex configuration |
| Policy | `grok_quota_auth_or_unavailable` |
| Fallback conditions | Typed quota exhaustion, typed authentication failure, typed launch unavailability |
| Unchanged workspace | Required |
| Unknown effects | Require reconciliation |
| Mandatory routine Codex review | No |

`bind_explicit_agent_route` checks this object exactly and then builds the production command through `worldwide_legal_search_provider.build_provider_argv`. That command is `provider_fallback_runner.py` with primary `grok_cli_runner.py --model grok-4.7 --mode agent`, fallback `build_codex_cli_command`, policy `grok_quota_auth_or_unavailable`, and `--probe-route-readiness`. Binding does not launch a provider. Codex model and reasoning effort stay deployment configuration. The binding does not add a second model preference, and it rejects sandbox bypass.

## Typed pre-effect emergency fallback

`evaluate_emergency_fallback` calls `route_agent_cli_failure` with the campaign policy, primary label `grok`, and fallback label `codex`. Codex is eligible only when the router classifies one of the three typed pre-effect failures and the candidate workspace digest is unchanged:

- `grok_quota_exhausted` with no activity and a trusted body-free receipt
- `authentication_failure` with no activity and a trusted body-free receipt
- `launch_failure` before dispatch, or with no activity and a trusted receipt

A pre-dispatch launch failure whose workspace digest changed is still reconciliation. The canonical router does not compare the workspace for pre-dispatch unavailability; this campaign binding does, because an unchanged workspace is required before any emergency Codex process starts.

## Ambiguous effects

The same decision returns `decision: reconcile` and does not select Codex after ambiguous effects:

- started or unknown activity
- an unreliable or missing workspace snapshot
- a mutated workspace
- timeout, transport failure, malformed output, generic nonzero exit, or task failure
- a missing, mismatched, or forged failure receipt

Those attempts stay on the fenced primary result. A later recovery attempt may use the same policy only after effects are reconciled and a new admission exists. This binding does not start that second writer.

## Route telemetry

Every decision writes one body-free `worldwide-legal-search.provider-route-telemetry@1` record. The record carries the task, attempt, stage, `grok_cli` / `grok-4.7`, fallback provider, policy, workspace digest, unchanged bit, decision, terminal reason, failure kind, and the canonical `ipfs_accelerate_py/provider-route@1` record when fallback is selected. `completion_authority` is false. The record contains no prompt body and no credential. Production runner receipts use `state/ptr_lane_worldwide_legal_search/provider_route_receipts/<task>/provider-route-attempt-<n>.json`.

## Admitted tiny live canary

The admitted tiny live canary is the only live process this binding starts. `run_admitted_tiny_canary` accepts only that admission. It accepts only the admission from `admitted_tiny_canary`: task `WWL-003`, attempt `1`, one fixed prompt under 64 bytes, `paid_provider_calls` 0, and `hosted_model_invocations` 0. The canary runs the real `provider_fallback_runner` once with `--primary-unavailable-kind launch_failure` and a local fallback script. It does not probe Grok or Codex, does not pass a hosted model command, and clears provider credentials and protected-state roots in the child environment so an operator profile cannot become canary authority.

The canary proves three runner facts: typed pre-effect fallback, an unchanged candidate workspace, and persisted route telemetry whose failure kind is `launch_failure`. A repeated call, a different admission, or a workspace change is refused. The canary is not task completion and does not authorize corpus work.

## Limitations

The canary is a non-charging local process. It does not prove that a hosted Grok or Codex credential is currently usable. Readiness of those binaries remains the production `--probe-route-readiness` path, which this task does not invoke. Codex is not a routine reviewer. This document does not grant campaign completion. `artifacts/worldwide-legal-search/receipts/WWL-003.json` does not grant it either.
