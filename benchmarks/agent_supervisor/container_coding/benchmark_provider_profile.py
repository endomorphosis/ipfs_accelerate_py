"""Current matched-provider pin shared by native and supervisor benchmarks.

Historical setup-cache policies retain their own version and binary pins.
Changing this profile requires fresh archives and trial preparations.
"""

CLI_VERSION = "0.160.0"
CODEX_PROFILE = "codex-gpt-6.1-sol-0.160.0@1"
GROK_PROFILE = "grok-4.7-cli-1.0.46@1"
PROVIDER_PROFILES = (CODEX_PROFILE, GROK_PROFILE)


def resolve_provider_profile(name=None):
    """Select an immutable route; no discovery or implicit fallback changes it."""
    if name is None:
        name = CODEX_PROFILE
    if type(name) is not str or name not in PROVIDER_PROFILES:
        raise ValueError("explicit supported benchmark provider profile required")
    if name == CODEX_PROFILE:
        return dict(id=name, provider="codex_cli", model="gpt-6.1-sol",
                    reasoning_effort="high", cli_version=CLI_VERSION)
    return dict(id=name, provider="grok_cli", model="grok-4.7",
                reasoning_effort="high", cli_version="1.0.46")


def require_runtime_provider_profile(manifest, name=None):
    selected = resolve_provider_profile(name)
    if selected["provider"] == "codex_cli":
        require_runtime_cli_version(manifest)
    else:
        from .terminal_grok_deployment import validate_grok_binding
        validate_grok_binding(manifest, required=True)
    return selected


def require_runtime_cli_version(manifest):
    """Refuse an archive that cannot preserve the current comparison's CLI pin."""
    if type(manifest) is not dict or manifest.get("codex_version") != CLI_VERSION:
        raise ValueError("runtime archive Codex version differs from the current benchmark profile; rebuild the archive")


def prepared_provider_identity(prepared, config):
    """Recover frozen identity without relabelling historical runs as current."""
    from .benchmark_controls import _digest, validate_controls
    if type(prepared) is not dict or type(config) is not dict:
        raise ValueError("frozen provider preparation and configuration must be objects")
    fields = ("model", "reasoning_effort", "cli_version")
    controls = prepared.get("comparison_controls")
    if controls is not None:
        if not validate_controls(controls) or controls["configuration_sha256"] != _digest(config):
            raise ValueError("frozen provider profile differs from prepared configuration")
        identity = {key: controls["identity"][key] for key in fields}
        if any(key in prepared and prepared[key] != identity[key] for key in fields):
            raise ValueError("prepared provider identity differs from frozen controls")
    else:
        identity = {key: prepared.get(key) for key in fields}
    if any(type(value) is not str or not 0 < len(value) <= 128 for value in identity.values()):
        raise ValueError("frozen provider identity is unavailable")
    agents = config.get("agents")
    if (type(agents) is not list or len(agents) != 1 or type(agents[0]) is not dict
            or agents[0].get("model_name") != identity["model"]):
        raise ValueError("frozen provider model differs from configuration")
    agent = agents[0]
    kwargs = agent.get("kwargs")
    if kwargs is not None and type(kwargs) is not dict:
        raise ValueError("frozen provider kwargs must be an object")
    declared_profile = (kwargs or {}).get("provider_profile")
    if declared_profile is not None or "provider_profile" in prepared:
        selected = resolve_provider_profile(declared_profile)
        if (prepared.get("provider_profile") != declared_profile
                or any(identity[key] != selected[key] for key in fields)):
            raise ValueError("frozen provider selection differs from preparation")
    if agent.get("name") == "codex" and not agent.get("import_path"):
        if agent.get("kwargs") != {"version": identity["cli_version"], "reasoning_effort": identity["reasoning_effort"]}:
            raise ValueError("frozen native provider profile differs from configuration")
    return identity
