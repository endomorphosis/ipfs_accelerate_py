"""Unified discover/ensure catalog for CLI coding tools.

Goose and Muse have first-class installers (pinned archive / official script).
Mistral Vibe uses uv/pip. The remaining tools are detect-first: ``ensure`` only
runs a documented official argv when ``auto_install=True``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS


@dataclass(frozen=True)
class CliInstallerSpec:
    provider: str
    binary: str
    env_names: tuple[str, ...]
    robustness: str
    official_argv: tuple[str, ...] = ()
    notes: str = ""


CLI_INSTALLERS: dict[str, CliInstallerSpec] = {
    "muse_code": CliInstallerSpec(
        provider="muse_code",
        binary="muse",
        env_names=("IPFS_ACCELERATE_MUSE_PATH", "MUSE_BIN", "MUSE_CLI_PATH"),
        robustness="official_script",
        notes="curl https://dev.meta.ai/install.sh; binary-present survives nonzero PATH exit",
    ),
    "goose_cli": CliInstallerSpec(
        provider="goose_cli",
        binary="goose",
        env_names=("IPFS_ACCELERATE_GOOSE_PATH", "GOOSE_BIN"),
        robustness="pinned_archive",
        notes="SHA-256 verified release manifest via ensure_goose",
    ),
    "mistral_vibe": CliInstallerSpec(
        provider="mistral_vibe",
        binary="vibe",
        env_names=("IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD", "VIBE_BIN", "MISTRAL_VIBE_BIN"),
        robustness="package_manager",
        notes="uv tool install mistral-vibe or pip --user",
    ),
    "codex_cli": CliInstallerSpec(
        provider="codex_cli",
        binary="codex",
        env_names=("CODEX_BIN", "IPFS_ACCELERATE_CODEX_PATH"),
        robustness="package_manager",
        official_argv=("npm", "install", "-g", "@openai/codex"),
        notes="Official npm package; detect-only unless auto_install",
    ),
    "claude_code": CliInstallerSpec(
        provider="claude_code",
        binary="claude",
        env_names=("CLAUDE_BIN", "ANTHROPIC_CLI_BIN", "IPFS_ACCELERATE_AGENT_CLAUDE_BIN"),
        robustness="official_script",
        official_argv=("npm", "install", "-g", "@anthropic-ai/claude-code"),
        notes="Official npm package; detect-only unless auto_install",
    ),
    "grok_cli": CliInstallerSpec(
        provider="grok_cli",
        binary="grok",
        env_names=("GROK_BIN", "ipfs_accelerate_py_GROK_CLI_CMD"),
        robustness="detect_only",
        official_argv=("npm", "install", "-g", "@xai/grok"),
        notes="Prefer PATH / grok login; npm ensure is best-effort",
    ),
    "copilot_cli": CliInstallerSpec(
        provider="copilot_cli",
        binary="copilot",
        env_names=("COPILOT_BIN", "GITHUB_COPILOT_BIN"),
        robustness="package_manager",
        official_argv=("npm", "install", "-g", "@github/copilot"),
        notes="Official npm package; detect-only unless auto_install",
    ),
    "gemini_cli": CliInstallerSpec(
        provider="gemini_cli",
        binary="gemini",
        env_names=("GEMINI_BIN", "ipfs_accelerate_py_GEMINI_CLI_CMD"),
        robustness="package_manager",
        official_argv=("npm", "install", "-g", "@google/gemini-cli"),
        notes="npx @google/gemini-cli also works without a global install",
    ),
}


@dataclass
class CliToolInstallResult:
    provider: str
    available: bool
    executable: str = ""
    method: str = ""
    reason: str = ""
    robustness: str = ""
    details: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "executable": self.executable,
            "method": self.method,
            "reason": self.reason,
            "robustness": self.robustness,
            "details": dict(self.details),
        }


def _first_token(raw: str) -> str:
    token = str(raw or "").strip().split()[0] if str(raw or "").strip() else ""
    return token.replace("{prompt}", "").replace("{model}", "").strip()


def _local_bin(binary: str, *, environ: Mapping[str, str]) -> Optional[str]:
    home = Path(str(environ.get("HOME") or Path.home()))
    path = home / ".local" / "bin" / binary
    try:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    except OSError:
        return None
    return None


def _executable_result(
    spec: CliInstallerSpec, executable: str, *, method: str
) -> CliToolInstallResult:
    return CliToolInstallResult(
        provider=spec.provider,
        available=True,
        executable=executable,
        method=method,
        robustness=spec.robustness,
    )


def discover_cli_tool(
    provider: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> CliToolInstallResult:
    """Detect-only lookup. Never downloads or installs."""
    key = str(provider or "").strip().lower().replace("-", "_")
    spec = CLI_INSTALLERS.get(key)
    if spec is None:
        return CliToolInstallResult(
            provider=key,
            available=False,
            reason="unknown_provider",
        )
    env = os.environ if environ is None else environ
    if key == "muse_code":
        try:
            from .muse import discover_muse

            found = discover_muse(probe_version=False, environ=env)
        except Exception as exc:
            return CliToolInstallResult(
                provider=key,
                available=False,
                method="discover",
                reason=type(exc).__name__,
                robustness=spec.robustness,
            )
        return CliToolInstallResult(
            provider=key,
            available=bool(found.available),
            executable=str(found.executable or ""),
            method=str(found.method or "discover"),
            reason=str(found.reason or ""),
            robustness=spec.robustness,
        )
    if key == "goose_cli":
        try:
            from .goose import discover_goose

            found = discover_goose(probe_version=False, environ=env)
            if found.available and found.executable:
                return CliToolInstallResult(
                    provider=key,
                    available=True,
                    executable=str(found.executable),
                    method=str(found.method or "discover"),
                    robustness=spec.robustness,
                )
        except Exception:
            pass
    if key == "mistral_vibe":
        exe = shutil.which("vibe") or shutil.which("mistral-vibe")
        if not exe:
            exe = _local_bin("vibe", environ=env) or _local_bin("mistral-vibe", environ=env)
        return CliToolInstallResult(
            provider=key,
            available=bool(exe),
            executable=str(exe or ""),
            method="discover",
            reason="" if exe else "not_installed",
            robustness=spec.robustness,
        )
    for env_name in spec.env_names:
        configured = str(env.get(env_name) or "").strip()
        token = _first_token(configured)
        if not token:
            continue
        path = Path(token).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return CliToolInstallResult(
                provider=key,
                available=True,
                executable=str(path),
                method="env",
                robustness=spec.robustness,
            )
        found = shutil.which(token)
        if found:
            return CliToolInstallResult(
                provider=key,
                available=True,
                executable=found,
                method="path",
                robustness=spec.robustness,
            )
    found = shutil.which(spec.binary)
    if found:
        return _executable_result(spec, found, method="path")
    local = _local_bin(spec.binary, environ=env)
    if local:
        return _executable_result(spec, local, method="local_bin")
    return CliToolInstallResult(
        provider=key,
        available=False,
        method="discover",
        reason="not_installed",
        robustness=spec.robustness,
    )


def ensure_cli_tool(
    provider: str,
    *,
    auto_install: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    run_fn: Any = None,
    timeout_seconds: float = 180.0,
) -> CliToolInstallResult:
    """Discover, optionally installing via the catalogued official argv."""
    key = str(provider or "").strip().lower().replace("-", "_")
    spec = CLI_INSTALLERS.get(key)
    env = os.environ if environ is None else environ
    found = discover_cli_tool(key, environ=env)
    if found.available:
        return found
    if not auto_install:
        return found
    if key == "muse_code":
        from .muse import ensure_muse

        try:
            result = ensure_muse(auto_install=True, environ=env)
        except Exception as exc:
            found.reason = type(exc).__name__
            found.method = "ensure"
            return found
        return CliToolInstallResult(
            provider=key,
            available=bool(result.available),
            executable=str(result.executable or ""),
            method=str(result.method or "ensure"),
            reason=str(result.reason or ""),
            robustness=spec.robustness if spec else "official_script",
        )
    if key == "goose_cli":
        from .goose import ensure_goose

        try:
            result = ensure_goose(auto_install=True, environ=env)
        except Exception as exc:
            found.reason = type(exc).__name__
            found.method = "ensure"
            return found
        return CliToolInstallResult(
            provider=key,
            available=bool(result.available),
            executable=str(result.executable or ""),
            method=str(result.method or "ensure"),
            reason=str(result.reason or ""),
            robustness=spec.robustness if spec else "pinned_archive",
        )
    if key == "mistral_vibe":
        from ipfs_accelerate_py.utils.mistral_vibe import ensure_mistral_vibe

        result = ensure_mistral_vibe(auto_install=True, environ=env)
        return CliToolInstallResult(
            provider=key,
            available=bool(result.available),
            executable=str(result.executable or ""),
            method=str(result.method or "ensure"),
            reason=str(result.reason or ""),
            robustness=spec.robustness if spec else "package_manager",
        )
    if spec is None or not spec.official_argv:
        found.reason = found.reason or "no_official_installer"
        return found
    runner = subprocess.run if run_fn is None else run_fn
    argv = list(spec.official_argv)
    if shutil.which(argv[0]) is None and run_fn is None:
        found.reason = f"missing_{argv[0]}"
        return found
    prefix = Path(str(env.get("HOME") or Path.home())) / ".local"
    attempts = [list(argv)]
    if argv and argv[0] == "npm":
        attempts.append(["npm", "install", "-g", "--prefix", str(prefix), argv[-1]])
    last_proc = None
    for command in attempts:
        try:
            last_proc = runner(
                command,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_seconds)),
                check=False,
                env=dict(env),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            found.reason = type(exc).__name__
            found.method = "official_argv"
            continue
        again = discover_cli_tool(key, environ=env)
        if again.available:
            again.method = "official_argv"
            return again
        prefixed = prefix / "bin" / spec.binary
        try:
            if prefixed.is_file() and os.access(prefixed, os.X_OK):
                return _executable_result(spec, str(prefixed), method="official_argv")
        except OSError:
            pass
        blob = (
            str(getattr(last_proc, "stdout", "") or "")
            + str(getattr(last_proc, "stderr", "") or "")
        ).lower()
        if "eacces" not in blob and "permission" not in blob:
            break
    found.method = "official_argv"
    found.reason = "install_did_not_produce_binary"
    found.details = {"returncode": str(getattr(last_proc, "returncode", "") if last_proc else "")}
    return found


def installer_robustness_report() -> list[dict[str, Any]]:
    """Operator-facing snapshot of installer coverage. Detect-only."""
    rows = []
    for provider in sorted(CLI_PROVIDERS):
        spec = CLI_INSTALLERS.get(provider)
        discovered = discover_cli_tool(provider)
        rows.append(
            {
                "provider": provider,
                "robustness": spec.robustness if spec else "missing",
                "binary": spec.binary if spec else "",
                "available": discovered.available,
                "executable": discovered.executable,
                "notes": spec.notes if spec else "no catalog entry",
            }
        )
    return rows


__all__ = [
    "CLI_INSTALLERS",
    "CliInstallerSpec",
    "CliToolInstallResult",
    "discover_cli_tool",
    "ensure_cli_tool",
    "installer_robustness_report",
]
