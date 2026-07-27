# Goose CLI Integration Objective Heap

This objective heap is the durable source of intent for adding Goose CLI
support to `ipfs_accelerate_py`. The companion todo board is the executable
projection. Model output and successful process exit are proposal evidence,
not completion evidence; every goal remains active until its acceptance
criteria are covered by current-tree implementation and validation evidence.

The integration has two separate authority profiles:

- `goose_cli` is stateless chat generation with tools, profiles, and sessions
  disabled.
- `goose_agent` is side-effecting agent execution and requires explicit local
  or worker policy authorization.

These profiles must never be silently interchanged by retry, cache, fallback,
autodiscovery, endpoint dispatch, or P2P routing.

## GOOSE-G000 Secure and unified Goose CLI support

- Status: active
- Parent:
- Fib priority: 1
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Add production-quality Goose CLI support through one reusable runtime used by llm_router, lazy installation, CLI endpoint and MCP handling, compatibility wrappers, and opt-in P2P execution while preserving strict separation between side-effect-free chat generation and authorized agent execution.
- Evidence: GOOSE-G010, GOOSE-G020, GOOSE-G030, GOOSE-G040, GOOSE-G050, GOOSE-G060, GOOSE-G070
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Acceptance: One canonical Goose adapter backs all surfaces; router chat mode cannot execute tools or load default extensions; installation is explicit, pinned, verified, noninteractive, idempotent, and atomic; endpoint registration, health, execution, streaming, cancellation, and session lifecycle are concrete and typed; agent execution requires explicit authorization and bounded roots; side-effecting requests are never cached, blindly retried, or cross-provider-fallbacked; Codex, Copilot, and existing router behavior remain compatible; operator documentation and live opt-in smoke instructions are complete.
- Gap task: Implement the highest-priority incomplete child goal without weakening install provenance, process isolation, endpoint authorization, or retry and cache safety.
- Refinement: Keep shared contracts, subprocess execution, binary installation, provider behavior, router wiring, endpoint lifecycle, ACP transport, worker policy, compatibility migration, tests, and documentation in dependency-aware lanes.
- Embedding query: goose cli llm router lazy installer endpoint MCP ACP agent safety process cancellation provider compatibility
- AST query: LLMProvider _builtin_provider_by_name CLIEndpointAdapter register_cli_endpoint _run_llm_generate BaseCLIWrapper
- Goal completion schema version: 1
- Completion confidence: 0.083333
- Uncovered criteria: ["One canonical Goose adapter backs all surfaces","router chat mode cannot execute tools or load default extensions","installation is explicit, pinned, verified, noninteractive, idempotent, and atomic","endpoint registration, health, execution, streaming, cancellation, and session lifecycle are concrete and typed","agent execution requires explicit authorization and bounded roots","side-effecting requests are never cached, blindly retried, or cross-provider-fallbacked","Codex, Copilot, and existing router behavior remain compatible","operator documentation and live opt-in smoke instructions are complete."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G010 Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G000
- Fib priority: 2
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Establish provider-neutral typed contracts, capability metadata, registry behavior, error taxonomy, and a bounded subprocess runner that can safely serve Goose and later consolidate Codex and Copilot.
- Evidence: ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py, test/test_cli_runtime_contracts.py, test/test_cli_runtime_process_runner.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Acceptance: Existing LLMProvider string generation remains compatible; rich request, result, event, capability, and error records are deterministic and bounded; execution uses argv with shell disabled and prompt stdin; synchronous, asynchronous, and streaming paths share semantics; timeout and cancellation terminate process trees; output is bounded; working directories and environment overlays are validated; secrets and prompts are absent from errors and telemetry; importing or listing providers starts no process and loads no optional provider.
- Gap task: Add or repair the smallest shared runtime component and its focused deterministic tests.
- Refinement: Separate immutable contracts and registry work from process execution so installer, endpoint, and provider lanes can build on stable interfaces.
- Embedding query: typed CLI runtime subprocess cancellation streaming capabilities side effects cache retry registry
- AST query: LLMProvider ProviderInfo _run_cli_command BaseCLIWrapper CLIEndpointAdapter
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Existing LLMProvider string generation remains compatible","rich request, result, event, capability, and error records are deterministic and bounded","execution uses argv with shell disabled and prompt stdin","synchronous, asynchronous, and streaming paths share semantics","timeout and cancellation terminate process trees","output is bounded","working directories and environment overlays are validated","secrets and prompts are absent from errors and telemetry","importing or listing providers starts no process and loads no optional provider."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G020 Pinned and verifiable Goose lazy installation

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3
- Track: goose-installer
- Priority: P0
- Bundle: llm-cli/goose/installer
- Goal: Discover or lazily install a supported Goose release without import-time effects, interactive configuration, privilege escalation, unverified archives, or shell-profile mutation.
- Evidence: ipfs_accelerate_py/cli_runtime/installers/goose.py, ipfs_accelerate_py/cli_runtime/installers/goose_release_manifest.json, test/test_goose_installer.py
- Outputs: ipfs_accelerate_py/cli_runtime/installers/__init__.py, ipfs_accelerate_py/cli_runtime/installers/goose.py, ipfs_accelerate_py/cli_runtime/installers/goose_release_manifest.json, test/test_goose_installer.py
- Validation: python -m pytest test/test_goose_installer.py -q
- Acceptance: Discovery checks explicit path, operator argv, PATH, and the managed versioned directory in deterministic order; implicit router discovery never installs; explicit provider selection may install unless policy disables it; OS, architecture, libc, and supported release variants are validated; archives are downloaded to staging, size bounded, SHA-256 checked against pinned package data, safely extracted, version probed, and atomically promoted; concurrent threads and processes collapse to one install; rollback preserves the prior executable; configuration and authentication remain separate typed readiness states; tests cover tampering, traversal, timeout, offline failure, unsupported platforms, locks, rollback, and no-install discovery without live network access.
- Gap task: Implement the next missing installer lifecycle or provenance guarantee with mocked deterministic tests.
- Refinement: Keep release-manifest maintenance separate from router and provider command construction.
- Embedding query: goose release lazy installer sha256 atomic rollback lock linux mac windows musl vulkan cuda
- AST query: ensure_mistral_vibe MistralVibeInstallResult auto_install _process_install_lock
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Discovery checks explicit path, operator argv, PATH, and the managed versioned directory in deterministic order","implicit router discovery never installs","explicit provider selection may install unless policy disables it","OS, architecture, libc, and supported release variants are validated","archives are downloaded to staging, size bounded, SHA-256 checked against pinned package data, safely extracted, version probed, and atomically promoted","concurrent threads and processes collapse to one install","rollback preserves the prior executable","configuration and authentication remain separate typed readiness states","tests cover tampering, traversal, timeout, offline failure, unsupported platforms, locks, rollback, and no-install discovery without live network access."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G030 Safe Goose provider and llm_router integration

- Status: active
- Parent: GOOSE-G000
- Fib priority: 5
- Track: goose-provider-router
- Priority: P0
- Bundle: llm-cli/goose/router
- Goal: Implement one Goose command builder and structured output parser, then expose it as `goose_cli` and `goose` through llm_router with safe chat defaults and capability-aware cache, retry, fallback, and concurrency behavior.
- Evidence: ipfs_accelerate_py/cli_runtime/providers/goose.py, ipfs_accelerate_py/llm_router.py, test/test_goose_cli_provider.py, test/test_llm_router_goose.py
- Outputs: ipfs_accelerate_py/cli_runtime/providers/__init__.py, ipfs_accelerate_py/cli_runtime/providers/goose.py, ipfs_accelerate_py/llm_router.py, test/test_goose_cli_provider.py, test/test_llm_router_goose.py
- Validation: python -m pytest test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_llm_router_integration.py -q
- Acceptance: Chat execution sets GOOSE_MODE=chat, disables sessions and default profiles, uses stdin, requests JSON, enforces low explicit turn and tool-repetition bounds, and enables no builtin or external extension; model_name maps only to the Goose model and a separate goose_provider selects the underlying provider; JSON and stream-json parsing preserves final text and bounded metadata; unsupported versions fail closed when required safety flags are missing; explicit selection can invoke the installer but generic discovery cannot; provider cache identity includes Goose configuration; agent mode is rejected by ordinary generate_text unless explicit side-effect authorization is supplied; side-effecting calls bypass response caches, default-model retry, provider fallback, and unsafe batch concurrency.
- Gap task: Implement or repair Goose provider behavior and router policy while preserving existing provider contracts.
- Refinement: Complete the standalone adapter and parser before modifying central router resolution.
- Embedding query: goose run no session no profile JSON stream chat llm router provider model retry cache
- AST query: _builtin_provider_by_name _resolve_provider_uncached generate_text _provider_cache_key _effective_model_key
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Chat execution sets GOOSE_MODE=chat, disables sessions and default profiles, uses stdin, requests JSON, enforces low explicit turn and tool-repetition bounds, and enables no builtin or external extension","model_name maps only to the Goose model and a separate goose_provider selects the underlying provider","JSON and stream-json parsing preserves final text and bounded metadata","unsupported versions fail closed when required safety flags are missing","explicit selection can invoke the installer but generic discovery cannot","provider cache identity includes Goose configuration","agent mode is rejected by ordinary generate_text unless explicit side-effect authorization is supplied","side-effecting calls bypass response caches, default-model retry, provider fallback, and unsafe batch concurrency."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G040 Concrete CLI endpoints, MCP handling, and ACP lifecycle

- Status: active
- Parent: GOOSE-G000
- Fib priority: 8
- Track: goose-endpoint
- Priority: P0
- Bundle: llm-cli/goose/endpoint
- Goal: Replace abstract endpoint construction with a canonical factory and provide Goose one-shot and persistent ACP endpoints with typed health, execution, streaming, cancellation, and session lifecycle.
- Evidence: ipfs_accelerate_py/cli_runtime/endpoints.py, ipfs_accelerate_py/cli_runtime/acp/goose_client.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, test/test_goose_cli_endpoint.py, test/test_goose_acp_client.py
- Outputs: ipfs_accelerate_py/cli_runtime/endpoints.py, ipfs_accelerate_py/cli_runtime/acp/__init__.py, ipfs_accelerate_py/cli_runtime/acp/goose_client.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py
- Validation: python -m pytest test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/mcp_server/test_cli_endpoint_tools.py -q
- Acceptance: Endpoint registration resolves a concrete factory instead of instantiating an abstract class; list and liveness are side-effect free; installed, configured, ready, and degraded states remain distinct; execute returns a typed bounded envelope without echoing the prompt; stream and cancel clean up child processes; one-shot chat uses the same Goose adapter as llm_router; persistent sessions use goose acp over stdio with bounded restart recovery; optional goose serve support, if added, binds loopback and requires a generated secret; agent endpoint requests require explicit side-effect policy, allowed roots, extension and builtin allowlists, turn limits, timeouts, and approval mode.
- Gap task: Implement the next endpoint lifecycle component without duplicating Goose command or parser logic.
- Refinement: Land the provider-neutral endpoint factory before Goose endpoint wiring, and land one-shot execution before ACP persistence.
- Embedding query: goose endpoint MCP factory health readiness stream cancel session ACP stdio lifecycle
- AST query: CLIEndpointAdapter register_cli_endpoint execute_cli_inference cli_endpoint_register goose acp
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Endpoint registration resolves a concrete factory instead of instantiating an abstract class","list and liveness are side-effect free","installed, configured, ready, and degraded states remain distinct","execute returns a typed bounded envelope without echoing the prompt","stream and cancel clean up child processes","one-shot chat uses the same Goose adapter as llm_router","persistent sessions use goose acp over stdio with bounded restart recovery","optional goose serve support, if added, binds loopback and requires a generated secret","agent endpoint requests require explicit side-effect policy, allowed roots, extension and builtin allowlists, turn limits, timeouts, and approval mode."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G050 Remote worker and side-effect safety

- Status: active
- Parent: GOOSE-G000
- Fib priority: 13
- Track: goose-worker-security
- Priority: P0
- Bundle: llm-cli/goose/security
- Goal: Make Goose safe under local batching, remote task workers, retries, fallback, cancellation, path handling, telemetry, and uncertain partial execution.
- Evidence: ipfs_accelerate_py/p2p_tasks/worker.py, test/test_goose_p2p_policy.py, test/test_goose_cli_security_e2e.py
- Outputs: ipfs_accelerate_py/p2p_tasks/worker.py, test/test_goose_p2p_policy.py, test/test_goose_cli_security_e2e.py
- Validation: python -m pytest test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py test/api/test_task_worker_session_failover.py -q
- Acceptance: Goose is absent from the default remote allowlist; safe chat and agent execution have separate enable gates; agent mode is disabled remotely by default; caller paths are resolved under configured roots; persisted sessions are sticky to one worker and isolated with an absolute GOOSE_PATH_ROOT; no cache, retry, or provider fallback occurs after output or tool activity; failures report whether side effects may have started; timeouts and cancellation kill descendants; telemetry records version, duration, status, bounded counts, and error class without prompts, credentials, command secrets, or raw model output; restart and duplicate-delivery tests prove uncertain agent work is not replayed automatically.
- Gap task: Close the highest-risk local or remote execution boundary with deterministic adversarial tests.
- Refinement: Keep worker policy separate from provider and endpoint implementation, then run one final cross-surface security matrix.
- Embedding query: goose P2P worker allowlist side effect idempotency retry fallback path root sticky session cancellation
- AST query: _allowed_llm_providers _run_llm_generate generate_text_mesh provider fallback resume_session_id
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Goose is absent from the default remote allowlist","safe chat and agent execution have separate enable gates","agent mode is disabled remotely by default","caller paths are resolved under configured roots","persisted sessions are sticky to one worker and isolated with an absolute GOOSE_PATH_ROOT","no cache, retry, or provider fallback occurs after output or tool activity","failures report whether side effects may have started","timeouts and cancellation kill descendants","telemetry records version, duration, status, bounded counts, and error class without prompts, credentials, command secrets, or raw model output","restart and duplicate-delivery tests prove uncertain agent work is not replayed automatically."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G060 Compatibility wrappers and shared Codex/Copilot runtime

- Status: active
- Parent: GOOSE-G000
- Fib priority: 21
- Track: cli-compatibility
- Priority: P1
- Bundle: llm-cli/goose/compatibility
- Goal: Provide a lazy Goose compatibility wrapper and begin consolidating stale Codex and Copilot wrapper behavior onto the shared CLI runtime without breaking public imports or getters.
- Evidence: ipfs_accelerate_py/cli_integrations/goose_cli_integration.py, ipfs_accelerate_py/cli_integrations/base_cli_wrapper.py, ipfs_accelerate_py/cli_integrations/openai_codex_cli_integration.py, ipfs_accelerate_py/cli_integrations/copilot_cli_integration.py, test/test_cli_integration_registry.py
- Outputs: ipfs_accelerate_py/cli_integrations/goose_cli_integration.py, ipfs_accelerate_py/cli_integrations/__init__.py, ipfs_accelerate_py/cli_integrations/base_cli_wrapper.py, ipfs_accelerate_py/cli_integrations/openai_codex_cli_integration.py, ipfs_accelerate_py/cli_integrations/copilot_cli_integration.py
- Validation: python -m pytest test/test_cli_integration_registry.py test/test_unified_cli_integration.py test/test_llm_router_integration.py -q
- Acceptance: Goose compatibility methods delegate to the canonical adapter; provider listing does not eagerly instantiate or probe every tool; existing import names and getters remain available; stale Codex and Copilot command paths are either migrated to the shared runner or explicitly deprecated behind compatible facades; generic caching and retry are disabled for side-effecting operations; custom commands remain argv-based and shell-free; contract tests prove existing router behavior is unchanged.
- Gap task: Remove the next duplicate or stale wrapper path while retaining compatibility and side-effect policy.
- Refinement: Add the Goose facade first, then migrate Codex and Copilot in bounded compatibility-preserving steps.
- Embedding query: goose CLI integration wrapper codex copilot shared runtime lazy registry compatibility cache retry
- AST query: BaseCLIWrapper OpenAICodexCLIIntegration CopilotCLIIntegration get_all_cli_integrations
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["Goose compatibility methods delegate to the canonical adapter","provider listing does not eagerly instantiate or probe every tool","existing import names and getters remain available","stale Codex and Copilot command paths are either migrated to the shared runner or explicitly deprecated behind compatible facades","generic caching and retry are disabled for side-effecting operations","custom commands remain argv-based and shell-free","contract tests prove existing router behavior is unchanged."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G070 Verification, documentation, and controlled rollout

- Status: active
- Parent: GOOSE-G000
- Fib priority: 34
- Track: goose-rollout
- Priority: P1
- Bundle: llm-cli/goose/rollout
- Goal: Prove cross-surface behavior with deterministic fixtures, preserve existing providers, document operator policy, and define an explicit opt-in live rollout.
- Evidence: test/test_goose_cli_security_e2e.py, docs/LLM_ROUTER.md, README.md, docs/INDEX.md
- Outputs: test/test_goose_cli_security_e2e.py, docs/LLM_ROUTER.md, README.md, docs/INDEX.md, docs/guides/QUICKSTART.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py test/test_llm_router_integration.py test/test_unified_cli_integration.py -q
- Acceptance: The default suite uses fake executables and no network; an explicit environment-gated live test validates a pinned supported Goose version and configured provider; security tests cover argv injection, archive traversal, digest mismatch, malicious output, excessive output, timeout, cancellation, orphan cleanup, profile isolation, path escape, secret redaction, duplicate delivery, and side-effect-aware retry and cache behavior; documentation covers installation, configuration, underlying provider and model mapping, chat versus agent authority, endpoints, ACP, P2P gates, state paths, health, troubleshooting, version updates, and rollback; rollout begins with explicit provider selection and opt-in discovery rather than silently changing automatic provider order.
- Gap task: Add the missing proof, regression test, or operator documentation required for controlled release.
- Refinement: Run focused tests in each producing task, then close with one cross-surface matrix and documentation pass.
- Embedding query: goose integration tests security documentation rollout live smoke backward compatibility
- AST query: test_llm_router_integration test_unified_cli_integration docs LLM_ROUTER
- Goal completion schema version: 1
- Completion confidence: 0.166667
- Uncovered criteria: ["The default suite uses fake executables and no network","an explicit environment-gated live test validates a pinned supported Goose version and configured provider","security tests cover argv injection, archive traversal, digest mismatch, malicious output, excessive output, timeout, cancellation, orphan cleanup, profile isolation, path escape, secret redaction, duplicate delivery, and side-effect-aware retry and cache behavior","documentation covers installation, configuration, underlying provider and model mapping, chat versus agent authority, endpoints, ACP, P2P gates, state paths, health, troubleshooting, version updates, and rollback","rollout begins with explicit provider selection and opt-in discovery rather than silently changing automatic provider order."]
- Stale evidence: []
- Analyzer health: {"evidence":{},"passed":false,"reason_code":"analyzer_health_missing","status":"missing"}
- Exhaustion quorum: {"evidence":{},"member_count":null,"reason_code":"exhaustion_quorum_missing","required_members":null,"satisfied":false,"stale_members":[]}
- Reopen reasons: []

## GOOSE-G071 Prove GOOSE-G010 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3000
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G010`.
- Evidence: GOOSE-G010
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G010
- AST query: GOOSE-G010
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G010` with a narrow, verifiable change.

## GOOSE-G072 Prove GOOSE-G020 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3001
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G020`.
- Evidence: GOOSE-G020
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G020
- AST query: GOOSE-G020
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G020` with a narrow, verifiable change.

## GOOSE-G073 Prove GOOSE-G030 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3002
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G030`.
- Evidence: GOOSE-G030
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G030
- AST query: GOOSE-G030
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G030` with a narrow, verifiable change.

## GOOSE-G074 Prove GOOSE-G040 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3000
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G040`.
- Evidence: GOOSE-G040
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G040
- AST query: GOOSE-G040
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G040` with a narrow, verifiable change.

## GOOSE-G075 Prove GOOSE-G050 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3000
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G050`.
- Evidence: GOOSE-G050
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G050
- AST query: GOOSE-G050
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G050` with a narrow, verifiable change.

## GOOSE-G076 Prove GOOSE-G060 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3001
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G060`.
- Evidence: GOOSE-G060
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G060
- AST query: GOOSE-G060
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G060` with a narrow, verifiable change.

## GOOSE-G077 Prove GOOSE-G070 for Secure and unified Goose CLI support

- Status: active
- Parent: GOOSE-G000
- Fib priority: 3002
- Track: goose-cli-integration
- Priority: P0
- Bundle: llm-cli/goose/root
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `GOOSE-G070`.
- Evidence: GOOSE-G070
- Outputs: ipfs_accelerate_py/cli_runtime, ipfs_accelerate_py/llm_router.py, ipfs_accelerate_py/mcp/tools/cli_endpoint_adapters.py, ipfs_accelerate_py/mcp_server/tools/cli_endpoint_tools/native_cli_endpoint_tools.py, ipfs_accelerate_py/p2p_tasks/worker.py, docs/LLM_ROUTER.md
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py test/test_goose_installer.py test/test_goose_cli_provider.py test/test_llm_router_goose.py test/test_goose_cli_endpoint.py test/test_goose_acp_client.py test/test_goose_p2p_policy.py test/test_goose_cli_security_e2e.py -q
- Refinement depth: 1
- Embedding query: GOOSE-G070
- AST query: GOOSE-G070
- Parallel lane: llm-cli/goose/root
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `GOOSE-G070` with a narrow, verifiable change.

## GOOSE-G078 Prove ipfs_accelerate_py/cli_runtime/contracts.py for Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G010
- Fib priority: 5000
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `ipfs_accelerate_py/cli_runtime/contracts.py`.
- Evidence: ipfs_accelerate_py/cli_runtime/contracts.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Refinement depth: 2
- Embedding query: ipfs_accelerate_py/cli_runtime/contracts.py
- AST query: ipfs_accelerate_py/cli_runtime/contracts.py
- Parallel lane: llm-cli/goose/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `ipfs_accelerate_py/cli_runtime/contracts.py` with a narrow, verifiable change.

## GOOSE-G079 Prove ipfs_accelerate_py/cli_runtime/errors.py for Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G010
- Fib priority: 5001
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `ipfs_accelerate_py/cli_runtime/errors.py`.
- Evidence: ipfs_accelerate_py/cli_runtime/errors.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Refinement depth: 2
- Embedding query: ipfs_accelerate_py/cli_runtime/errors.py
- AST query: ipfs_accelerate_py/cli_runtime/errors.py
- Parallel lane: llm-cli/goose/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `ipfs_accelerate_py/cli_runtime/errors.py` with a narrow, verifiable change.

## GOOSE-G080 Prove ipfs_accelerate_py/cli_runtime/registry.py for Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G010
- Fib priority: 5002
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `ipfs_accelerate_py/cli_runtime/registry.py`.
- Evidence: ipfs_accelerate_py/cli_runtime/registry.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Refinement depth: 2
- Embedding query: ipfs_accelerate_py/cli_runtime/registry.py
- AST query: ipfs_accelerate_py/cli_runtime/registry.py
- Parallel lane: llm-cli/goose/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `ipfs_accelerate_py/cli_runtime/registry.py` with a narrow, verifiable change.

## GOOSE-G081 Prove ipfs_accelerate_py/cli_runtime/process_runner.py for Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G010
- Fib priority: 5000
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `ipfs_accelerate_py/cli_runtime/process_runner.py`.
- Evidence: ipfs_accelerate_py/cli_runtime/process_runner.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Refinement depth: 2
- Embedding query: ipfs_accelerate_py/cli_runtime/process_runner.py
- AST query: ipfs_accelerate_py/cli_runtime/process_runner.py
- Parallel lane: llm-cli/goose/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `ipfs_accelerate_py/cli_runtime/process_runner.py` with a narrow, verifiable change.

## GOOSE-G082 Prove test/test_cli_runtime_contracts.py for Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G010
- Fib priority: 5001
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `test/test_cli_runtime_contracts.py`.
- Evidence: test/test_cli_runtime_contracts.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Refinement depth: 2
- Embedding query: test/test_cli_runtime_contracts.py
- AST query: test/test_cli_runtime_contracts.py
- Parallel lane: llm-cli/goose/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `test/test_cli_runtime_contracts.py` with a narrow, verifiable change.

## GOOSE-G083 Prove test/test_cli_runtime_process_runner.py for Shared CLI runtime contracts and process lifecycle

- Status: active
- Parent: GOOSE-G010
- Fib priority: 5002
- Track: cli-runtime
- Priority: P0
- Bundle: llm-cli/goose/runtime
- Goal: Create concrete implementation, tests, docs, or interface descriptors proving `test/test_cli_runtime_process_runner.py`.
- Evidence: test/test_cli_runtime_process_runner.py
- Outputs: ipfs_accelerate_py/cli_runtime/__init__.py, ipfs_accelerate_py/cli_runtime/contracts.py, ipfs_accelerate_py/cli_runtime/errors.py, ipfs_accelerate_py/cli_runtime/registry.py, ipfs_accelerate_py/cli_runtime/process_runner.py
- Validation: python -m pytest test/test_cli_runtime_contracts.py test/test_cli_runtime_process_runner.py -q
- Refinement depth: 2
- Embedding query: test/test_cli_runtime_process_runner.py
- AST query: test/test_cli_runtime_process_runner.py
- Parallel lane: llm-cli/goose/runtime
- Conflict policy: prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts
- Gap task: Close the missing objective evidence `test/test_cli_runtime_process_runner.py` with a narrow, verifiable change.
