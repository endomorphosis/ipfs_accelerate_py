"""Current matched-provider pin shared by native and supervisor benchmarks.

Historical setup-cache policies retain their own version and binary pins.
Changing this profile requires fresh archives and trial preparations.
"""

CLI_VERSION = "0.160.0"


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
    if agent.get("name") == "codex" and not agent.get("import_path"):
        if agent.get("kwargs") != {"version": identity["cli_version"], "reasoning_effort": identity["reasoning_effort"]}:
            raise ValueError("frozen native provider profile differs from configuration")
    return identity
