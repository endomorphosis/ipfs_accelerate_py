"""Opt-in coding guidance for explicit, bounded command-output artifacts.

The native provider owns its tool loop. This policy does not intercept it, and
the worker's artifact observations confer no validation or completion authority.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shlex
import sys
import uuid

SCHEMA = "supervisor-tool-output-policy@1"
POLICIES = ("legacy", "artifact-preview@1", "artifact-preview@2")


def validate_policy(value: str, *, purpose: str, provider: str) -> str:
    if type(value) is not str or value not in POLICIES:
        raise ValueError("explicit supported tool output policy required")
    if value != "legacy" and (purpose != "coding" or provider != "codex_cli"):
        raise ValueError("artifact preview requires a Codex coding route")
    if value != "legacy" and sys.platform != "linux":
        raise ValueError("artifact preview requires the Linux process cleanup boundary")
    return value


def _path(value):
    if (type(value) is not str or not 1 <= len(value.encode()) <= 4096
            or any(ord(char) < 32 for char in value)):
        raise ValueError("bounded absolute tool helper path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("canonical absolute tool helper path required")
    return path


def _instruction(binding: dict) -> str:
    python = shlex.quote(binding["python_executable"])
    helper = shlex.quote(binding["helper_script"])
    store = shlex.quote(binding["store_path"])
    if binding["requested_policy"] == "artifact-preview@2":
        return (
            "Optional bounded output helper:\n"
            "Keep targeted reads and existing command batches. Use normal tools for small outputs; "
            "do not add commands just to use this helper. For expected verbose, noninteractive output:\n"
            f"{python} {helper} run --store {store} --response-format compact --timeout 120 --preview-bytes 2048 -- PROGRAM ARG...\n"
            "Runs argv once without stdin/TTY; stdout and stderr stay separate. Limits: 8 MiB capture, "
            "32 MiB store, 120 seconds; exceeding limits can terminate the command. Check complete, "
            "exit_code, cleanup_complete and termination_reason. Small outputs use one exact encoding. "
            "Omitted output has head/tail and omitted_bytes; only complete omitted streams include a replay SHA256. "
            "Retrieve necessary ranges, especially missing failure details:\n"
            f"{python} {helper} read --store {store} --response-format compact --sha256 HASH --offset 0 --length 4096\n"
            "Pages contain exact UTF-8 text or base64, never both. Do not rerun commands to recover output; "
            "previews do not establish success. Artifacts expire after this invocation and carry no proof "
            "or completion authority.\n\n"
        )
    return (
        "Bounded command-output previews:\n"
        "Use targeted reads. For expected verbose, noninteractive inspection or test commands, "
        "retain full output and inspect a small preview with:\n"
        f"{python} {helper} run --store {store} --timeout 120 --preview-bytes 2048 -- PROGRAM ARG...\n"
        "This explicitly runs the given argv once with no stdin or TTY. It captures stdout and stderr "
        "under an 8 MiB combined limit, a 32 MiB store quota, and a timeout; limits can terminate the command. "
        "The JSON reports the original exit code, capture completeness, previews and omitted byte counts. "
        "Only complete captures have replayable full output. To inspect omitted evidence, use:\n"
        f"{python} {helper} read --store {store} --sha256 HASH --offset 0 --length 4096\n"
        "Retrieve further ranges as needed, especially failure details; previews do not establish success. "
        "Do not repeat commands just to obtain unshown output. Interactive commands should use normal tools. "
        "Artifacts are removed after this invocation. They are temporary worker data, "
        "not task outputs, proof, or completion authority.\n\n"
    )


def create_binding(*, workspace: Path, requested_policy: str = "artifact-preview@1") -> dict:
    """Choose one unused invocation path without creating any files yet."""
    validate_policy(requested_policy, purpose="coding", provider="codex_cli")
    if requested_policy == "legacy":
        raise ValueError("legacy policy has no tool output binding")
    helper = Path(__file__).with_name("tool_output_artifact.py").resolve(strict=True)
    if not helper.is_file() or helper.stat().st_size > 256_000:
        raise ValueError("bounded output helper source required")
    binding = {
        "schema": SCHEMA, "requested_policy": requested_policy,
        "python_executable": str(Path(sys.executable).absolute()),
        "helper_script": str(helper),
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "store_path": "/tmp/supervisor-tool-output-" + uuid.uuid4().hex,
        "native_tool_interception": False, "helper_adoption_guaranteed": False,
        "token_savings_measured": False, "execution_authority": False,
        "completion_authority": False,
    }
    instruction = _instruction(binding)
    binding.update(instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                   instruction_bytes=len(instruction.encode()))
    validate_binding(binding, workspace=workspace)
    return binding


def validate_binding(binding: dict, *, workspace: Path) -> str:
    """Reconstruct recorded guidance without reading worker files or executing it."""
    fields = {"schema", "requested_policy", "python_executable", "helper_script", "helper_sha256",
              "store_path", "native_tool_interception", "helper_adoption_guaranteed",
              "token_savings_measured", "execution_authority", "completion_authority",
              "instruction_sha256", "instruction_bytes"}
    if (type(binding) is not dict or set(binding) not in (fields, fields | {"usage"})
            or binding.get("schema") != SCHEMA
            or type(binding.get("requested_policy")) is not str
            or binding["requested_policy"] not in POLICIES[1:]
            or any(binding.get(key) is not False for key in ("native_tool_interception", "helper_adoption_guaranteed",
                "token_savings_measured", "execution_authority", "completion_authority"))):
        raise ValueError("invalid tool output policy binding")
    python, helper, store = (_path(binding[key]) for key in ("python_executable", "helper_script", "store_path"))
    if (helper.name != "tool_output_artifact.py"
            or re.fullmatch(r"/tmp/supervisor-tool-output-[0-9a-f]{32}", str(store)) is None
            or store.is_relative_to(workspace) or helper.is_relative_to(workspace)
            or type(binding["helper_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", binding["helper_sha256"]) is None):
        raise ValueError("invalid tool output helper identity")
    instruction = _instruction(binding)
    if (type(binding["instruction_bytes"]) is not int
            or binding["instruction_bytes"] != len(instruction.encode())
            or binding["instruction_bytes"] > 8192
            or binding["instruction_sha256"] != hashlib.sha256(instruction.encode()).hexdigest()):
        raise ValueError("tool output instruction binding differs")
    return instruction
