#!/usr/bin/env python3
"""Finite fixture-only native Grok qualification; execute inside a native owner.

This helper performs no canonical task mutations or real inference. A fresh
network namespace is mandatory before any Grok process can start. HOME is kept
unchanged. Runtime fixtures use their own GROK_HOME and no real credentials.
Results describe this exact binary, source and fixture configuration only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.server
import importlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time

SCHEMA = "grok-planning-loopback-confinement-qualification/v5"
FIXTURE_ROOT = Path("/tmp/pytest-of-barberb/grk17f")
FIXTURE_MAX_BYTES = 128 * 1024 * 1024
FIXTURE_MAX_ENTRIES = 8192
PREDICATE_SELECTION_EVIDENCE = [{'path': '/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-grok-rootlesskit-loopback-20261017-03/rootlesskit-loopback/canonical-planning-adversarial/case-result.json', 'sha256': '09276e7813f82a0a5de45a2e5f879d63f365339ff0fb20f2e410c1c1a9e3e26a', 'size_bytes': 14507}, {'path': '/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-grok-rootlesskit-loopback-20261017-03/rootlesskit-loopback/canonical-planning-adversarial/request-02.json', 'sha256': '31e25da55b32c704d9912b7c0902f140fa6a2ecdeefae865041478989da08cee', 'size_bytes': 7713}]
FAKE_KEY = "nonsecret-local-tool-confinement-fixture"
MODEL = "grok-4.7"
FORBIDDEN = (
    "read_file", "run_terminal_command", "search_tool", "use_tool",
    "CallMcpTool", "FetchMcpResource", "ListMcpResources", "spawn_subagent",
)
def parse_argument_object(text: str) -> dict | None:
    """Reject ambiguous, malformed or non-object argument JSON."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate argument key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("non-JSON argument constant")

    try:
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def exact_arguments(actual: dict | None, expected: dict) -> bool:
    if actual is None:
        return False
    try:
        return json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False


def forbidden_arguments(name: str, sentinel: Path) -> dict:
    if name == "run_terminal_command":
        return {"command": "printf 'fixture-only-forbidden-terminal'"}
    if name in {"use_tool", "CallMcpTool"}:
        return {"name": "fixture__sentinel", "toolName": "fixture__sentinel", "arguments": {}}
    return {"target_file": str(sentinel), "path": str(sentinel)}


def native_parse_stage_refused(message: dict, requested_tool: str, expected_arguments: dict) -> bool:
    """Accept only the observed native argument-parser refusal, bound to this call.

    This finite observation does not establish which registry the parser uses.
    In particular an inner MCP 'Error calling tool: Tool not found' is rejected.
    Both literals in this exact wrapper occur in the pinned Grok 1.0.60 binary.
    """
    if (message.get("role") != "tool"
            or message.get("tool_call_id") != "call-forbidden-" + requested_tool):
        return False
    prefix = (f"Failed to parse arguments for tool `{requested_tool}`: Tool not found: {requested_tool}"
              "\n\nYour original arguments:\n")
    content = message.get("content")
    if not isinstance(content, str) or not content.startswith(prefix):
        return False
    return exact_arguments(parse_argument_object(content[len(prefix):]), expected_arguments)


def bound_native_parse_refused(call: dict, message: dict, requested_tool: str, sentinel: Path) -> bool:
    function = call.get("function", {})
    if (call.get("id") != "call-forbidden-" + requested_tool or call.get("type") != "function"
            or not isinstance(function, dict) or function.get("name") != requested_tool):
        return False
    expected = forbidden_arguments(requested_tool, sentinel)
    return (exact_arguments(parse_argument_object(function.get("arguments")), expected)
            and native_parse_stage_refused(message, requested_tool, expected))


def qualify_refusal_predicate() -> dict:
    """Independent fixed examples prevent broad unknown-tool acceptance."""
    expected = {"target_file": "/fixture/sentinel.txt", "path": "/fixture/sentinel.txt"}
    valid = {"role": "tool", "tool_call_id": "call-forbidden-read_file", "content":
        'Failed to parse arguments for tool `read_file`: Tool not found: read_file\n\nYour original arguments:\n'
        '{"target_file": "/fixture/sentinel.txt", "path": "/fixture/sentinel.txt"}'}
    samples = [
        ("observed-exact-native-parse-wrapper", valid, True),
        ("inner-mcp-tool-error", {**valid, "content":
            'Error calling tool: Tool not found: read_file\n\nYour original arguments:\n'
            '{"target_file": "/fixture/sentinel.txt", "path": "/fixture/sentinel.txt"}'}, False),
        ("unreviewed-registry-wording", {**valid, "content": 'Tool not found in LocalRegistry: read_file'}, False),
        ("wrong-call-id", {**valid, "tool_call_id": "call-forbidden-use_tool"}, False),
        ("wrong-prefix-tool-name", {**valid, "content":
            'Failed to parse arguments for tool `use_tool`: Tool not found: read_file\n\nYour original arguments:\n'
            '{"target_file": "/fixture/sentinel.txt", "path": "/fixture/sentinel.txt"}'}, False),
        ("wrong-repeated-tool-name", {**valid, "content":
            'Failed to parse arguments for tool `read_file`: Tool not found: use_tool\n\nYour original arguments:\n'
            '{"target_file": "/fixture/sentinel.txt", "path": "/fixture/sentinel.txt"}'}, False),
        ("wrong-original-arguments", {**valid, "content":
            'Failed to parse arguments for tool `read_file`: Tool not found: read_file\n\nYour original arguments:\n'
            '{"target_file": "/other/sentinel.txt", "path": "/other/sentinel.txt"}'}, False),
        ("duplicate-original-argument-key", {**valid, "content":
            'Failed to parse arguments for tool `read_file`: Tool not found: read_file\n\nYour original arguments:\n'
            '{"path": "/other.txt", "target_file": "/fixture/sentinel.txt", "path": "/fixture/sentinel.txt"}'}, False),
        ("malformed-original-arguments", {**valid, "content":
            'Failed to parse arguments for tool `read_file`: Tool not found: read_file\n\nYour original arguments:\n{'}, False),
        ("trailing-second-json-object", {**valid, "content": valid["content"] + '{}'}, False),
        ("wrong-message-role", {**valid, "role": "assistant"}, False),
    ]
    observations = []
    for name, message, expected_outcome in samples:
        observed = native_parse_stage_refused(message, "read_file", expected)
        observations.append({"case": name, "expected": expected_outcome, "observed": observed,
                             "passed": observed == expected_outcome})
    call = {"id": "call-forbidden-read_file", "type": "function", "function": {
        "name": "read_file", "arguments": '{"path":"/fixture/sentinel.txt","target_file":"/fixture/sentinel.txt"}'}}
    echo_samples = [
        ("exact-bound-assistant-echo", call, True),
        ("wrong-assistant-call-id", {**call, "id": "call-forbidden-use_tool"}, False),
        ("wrong-assistant-tool-name", {**call, "function": {**call["function"], "name": "use_tool"}}, False),
        ("wrong-assistant-arguments", {**call, "function": {**call["function"],
            "arguments": '{"path":"/other.txt","target_file":"/other.txt"}'}}, False),
    ]
    for name, candidate, expected_outcome in echo_samples:
        observed = bound_native_parse_refused(candidate, valid, "read_file", Path("/fixture/sentinel.txt"))
        observations.append({"case": name, "expected": expected_outcome, "observed": observed,
                             "passed": observed == expected_outcome})
    return {"schema": "native-parse-refusal-predicate-fixtures/v1", "cases": observations,
            "passed": all(row["passed"] for row in observations)}


def write_new(path: Path, value) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)


def pin(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            h.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": h.hexdigest()}


def exact_head(controller: Path, expected: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=controller, text=True).strip()
    if actual != expected:
        raise ValueError("controller HEAD differs from explicit reviewed input")


def bring_loopback_up() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as descriptor:
        request = struct.pack("16sH14x", b"lo", 0)
        current = fcntl.ioctl(descriptor.fileno(), 0x8913, request)
        flags = struct.unpack("16sH14x", current)[1] | 1
        fcntl.ioctl(descriptor.fileno(), 0x8914, struct.pack("16sH14x", b"lo", flags))


def safe_import_environment(temporary_directory: Path, allocation_database: Path) -> dict[str, str]:
    """Retain native owner import limits and keep all temp writes charged."""
    if (not temporary_directory.is_dir() or temporary_directory.is_symlink()
            or not temporary_directory.is_absolute()):
        raise ValueError("exact owned fixture temporary directory required")
    if (not allocation_database.is_absolute() or not allocation_database.parent.is_dir()
            or allocation_database.is_symlink()):
        raise ValueError("exact owned fixture allocation database path required")
    return {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": os.environ["HOME"],
        "TMPDIR": str(temporary_directory), "TMP": str(temporary_directory),
        "TEMP": str(temporary_directory), "PYTHONDONTWRITEBYTECODE": "1",
        "IPFS_ACCEL_SKIP_CORE": "1", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1", "IPFS_AUTO_INSTALL": "false",
        "IPFS_DATASETS_AUTO_INSTALL": "false", "IPFS_ACCELERATE_AUTO_INSTALL": "false",
        "IPFS_ACCELERATE_GOOSE_AUTO_INSTALL": "0", "IPFS_ACCELERATE_PY_GOOSE_AUTO_INSTALL": "0",
        "IPFS_ACCELERATE_MUSE_AUTO_INSTALL": "0", "IPFS_ACCELERATE_PY_MUSE_AUTO_INSTALL": "0",
        "IPFS_ACCELERATE_MISTRAL_VIBE_AUTO_INSTALL": "0",
        "IPFS_ACCELERATE_LLAMA_CPP_AUTO_INSTALL": "0", "IPFS_ACCELERATE_LLAMA_CPP_NATIVE_AUTO_INSTALL": "0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "-1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        # Native discovery itself writes provider health/limits, and generate_text
        # records observations/sessions. Bind the supported native store before
        # canonical imports, retaining this one fresh fixture path across cases.
        "IPFS_ACCELERATE_LLM_ALLOCATION_DB": str(allocation_database),
        "IPFS_ACCELERATE_PY_ROUTER_CACHE": "0", "IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE": "0",
        "XDG_CONFIG_HOME": str(temporary_directory / "xdg-config"),
        "XDG_DATA_HOME": str(temporary_directory / "xdg-data"),
        "XDG_STATE_HOME": str(temporary_directory / "xdg-state"),
        "XDG_CACHE_HOME": str(temporary_directory / "xdg-cache"),
    }


def fixture_environment(home: Path, base_url: str, binary: Path,
                        temporary_directory: Path, allocation_database: Path) -> dict[str, str]:
    # Preserve HOME literally; no production credentials or endpoint environment
    # is forwarded. The kernel network namespace independently prevents egress.
    result = safe_import_environment(temporary_directory, allocation_database)
    result.update({"GROK_HOME": str(home), "XAI_API_KEY": FAKE_KEY,
              "GROK_MODELS_BASE_URL": base_url,
              "GROK_MODELS_LIST_URL": base_url + "/models",
              "GROK_CLI_CHAT_PROXY_BASE_URL": base_url,
              "GROK_XAI_API_BASE_URL": base_url,
              "GROK_TELEMETRY_ENABLED": "0", "GROK_FEEDBACK_ENABLED": "0",
              "GROK_TELEMETRY_TRACE_UPLOAD": "0", "GROK_AUTO_UPDATE": "0",
              "GROK_MANAGED_MCPS_ENABLED": "0", "GROK_MANAGED_MCP_GATEWAY_TOOLS_ENABLED": "0",
              "GROK_SUBAGENTS": "0", "GROK_MEMORY": "0", "GROK_WEB_FETCH": "0",
              "IPFS_ACCELERATE_PY_GROK_CLI_CMD": str(binary),
              "NO_COLOR": "1"})
    for vendor in ("CLAUDE", "CURSOR", "CODEX"):
        for component in ("SKILLS", "RULES", "AGENTS", "MCPS", "HOOKS", "SESSIONS"):
            result[f"GROK_{vendor}_{component}_ENABLED"] = "0"
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        result[name] = str(home / name.lower())
    return result


def make_config(home: Path, base_url: str) -> None:
    text = '\n'.join((
        '[cli]', 'auto_update = false', 'use_leader = false',
        '[features]', 'telemetry = false', 'feedback = false', 'lsp_tools = false',
        '[session]', 'load_envrc = false',
        '[endpoints]', f'models_base_url = "{base_url}"',
        '[model."grok-4.7"]', f'base_url = "{base_url}"',
        f'api_key = "{FAKE_KEY}"', 'temperature = 0.0',
        '[compat.claude]', 'skills = false', 'rules = false', 'agents = false',
        'mcps = false', 'hooks = false', 'sessions = false',
        '[compat.cursor]', 'skills = false', 'rules = false', 'agents = false',
        'mcps = false', 'hooks = false', 'sessions = false',
        '[compat.codex]', 'skills = false', 'rules = false', 'agents = false',
        'mcps = false', 'hooks = false', 'sessions = false', '',
    ))
    target = home / "config.toml"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)


def read_tool_arguments(definition: dict, sentinel: Path) -> dict:
    parameters = definition.get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    arguments = {}
    for name, spec in properties.items():
        if re.search(r"(?:path|file)", name, re.I) and spec.get("type") == "string":
            arguments[name] = str(sentinel)
        elif name in required:
            kind = spec.get("type")
            if kind == "boolean": arguments[name] = True
            elif kind in {"integer", "number"}: arguments[name] = 1
            elif kind == "array": arguments[name] = []
            elif kind == "object": arguments[name] = {}
            elif kind == "string": arguments[name] = str(sentinel)
            else: raise ValueError("positive fixture tool required an unsupported argument type")
    if not arguments:
        raise ValueError("positive fixture read tool had no understood path argument")
    return arguments


class FixtureServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


def request_class(path: str, body: dict) -> str:
    """Keep observed native ancillary calls out of primary tool injection."""
    endpoint = path.split('?', 1)[0]
    if endpoint == '/v1/responses':
        items = body.get('input', [])
        tools = body.get('tools', [])
        if (isinstance(items, list) and len(items) == 2
                and items[0].get('role') == 'system'
                and str(items[0].get('content', '')).startswith('You are tasked with generating the session title.')
                and items[1].get('role') == 'user'
                and 'Reply exactly QUALIFICATION_DONE after any fixture requests.' in str(items[1].get('content', ''))
                and [tool.get('name') for tool in tools] == ['session_title']):
            return 'auxiliary_session_title'
    if endpoint == '/v1/chat/completions':
        messages = body.get('messages', [])
        if isinstance(messages, list) and messages:
            first = str(messages[0].get('content', ''))
            if (messages[0].get('role') == 'system'
                    and first.startswith('Write an ultra-short dashboard line that captures')
                    and not body.get('tools') and not body.get('functions')):
                return 'auxiliary_dashboard_summary'
            if (messages[0].get('role') == 'system' and first.startswith('You are Grok released by xAI.')
                    and any(item.get('role') == 'user' and item.get('content') ==
                        'Reply exactly QUALIFICATION_DONE after any fixture requests.' for item in messages)):
                return 'primary_agent'
    return 'unsupported'


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args): pass

    def send_json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": MODEL, "object": "model", "created": 0, "owned_by": "fixture"}]})
        else:
            self.send_json(404, {"error": {"message": "fixture supports model list only"}})

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 4_194_304:
            self.send_json(413, {"error": {"message": "bounded fixture request required"}})
            return
        body = json.loads(self.rfile.read(size))
        state = self.server.state
        classification = request_class(self.path, body)
        with state["lock"]:
            sequence = len(state["requests"]) + 1
            state["requests"].append({"path": self.path, "body": body, "request_class": classification,
                "authorization_is_fixture": self.headers.get("Authorization") == "Bearer " + FAKE_KEY})
            write_new(state["directory"] / f"request-{sequence:02d}.json", state["requests"][-1])
        primary_sequence = sum(row['request_class'] == 'primary_agent' for row in state['requests'])
        if classification == 'auxiliary_session_title':
            self.send_json(400, {'error': {'message': 'fixture intentionally declines ancillary title generation'}})
            return
        if classification == 'unsupported' or sequence > 8 or primary_sequence > 3:
            self.send_json(400, {"error": {"message": "unsupported fixture protocol; qualification inconclusive"}})
            return
        calls = []
        if classification == "primary_agent" and primary_sequence == 1 and state["case"] == "positive-read-file":
            offered = body.get("tools", [])
            read = [t.get("function", {}) for t in offered if
                    t.get("function", {}).get("name", "").lower() in {"read", "read_file"}]
            if len(read) == 1:
                calls = [{"id": "call-positive-read-file", "type": "function", "function": {
                    "name": read[0]["name"], "arguments": json.dumps(read_tool_arguments(read[0], state["sentinel"]))}}]
        elif classification == "primary_agent" and primary_sequence == 1 and state["case"] == "canonical-planning-adversarial":
            for name in FORBIDDEN:
                arguments = {"target_file": str(state["sentinel"]), "path": str(state["sentinel"])}
                if "terminal" in name:
                    arguments = {"command": "printf 'fixture-only-forbidden-terminal'"}
                elif name in {"use_tool", "CallMcpTool"}:
                    arguments = {"name": "fixture__sentinel", "toolName": "fixture__sentinel", "arguments": {}}
                calls.append({"id": "call-forbidden-" + name, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments)}})
        finish = "tool_calls" if calls else "stop"
        text = "" if calls else "QUALIFICATION_DONE"
        model = body.get("model", MODEL)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            delta = {"role": "assistant", "content": text}
            if calls:
                delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(calls)]
            frames = [
                {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ]
            for frame in frames:
                self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            message = {"role": "assistant", "content": text}
            if calls: message["tool_calls"] = calls
            self.send_json(200, {"id": "chatcmpl-fixture", "object": "chat.completion", "created": 0,
                "model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})



def fixture_root_inventory(*, phase: str) -> dict:
    """Bounded descriptor-relative census with O_NOFOLLOW directory opens.

    Enumeration never opens a symlink as a directory. Link-chain resolution
    uses bounded metadata lookups rooted at the held fixture descriptor and
    rejects every lexical escape before looking up that target. This remains
    a non-atomic live snapshot: directory replacement/ABA and namespace changes
    are not fully excluded, and quiescent post-owner readback is still required.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def identity(info):
        return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))

    def link_identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_uid, info.st_gid, info.st_mtime_ns, info.st_ctime_ns)

    def open_exact_directory(parent_fd, name, expected):
        if not stat.S_ISDIR(expected.st_mode):
            raise ValueError("fixture directory name is not an exact directory")
        result = os.open(name, flags, dir_fd=parent_fd)
        try:
            if (identity(os.fstat(result)) != identity(expected)
                    or identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != identity(expected)):
                raise ValueError("fixture directory name/inode changed while opening")
            return result
        except BaseException:
            os.close(result)
            raise

    # Anchor through every named ancestor with no symlink-following opens.
    anchor = os.open("/", flags)
    try:
        for component in FIXTURE_ROOT.parts[1:]:
            info = os.stat(component, dir_fd=anchor, follow_symlinks=False)
            child = open_exact_directory(anchor, component, info)
            os.close(anchor)
            anchor = child
        root_info = os.fstat(anchor)
        if root_info.st_uid != os.getuid():
            raise ValueError("fixture root owner differs from fixture process")
        if identity(FIXTURE_ROOT.lstat()) != identity(root_info):
            raise ValueError("named fixture root differs from held directory")

        def resolve_inside(lexical: Path) -> Path:
            if not lexical.is_relative_to(FIXTURE_ROOT):
                raise ValueError("fixture symlink lexical target escapes charged root")
            pending = list(lexical.relative_to(FIXTURE_ROOT).parts)
            resolved = []
            current = os.dup(anchor)
            hops = 0
            try:
                while pending:
                    component = pending.pop(0)
                    if component == "..":
                        if not resolved:
                            raise ValueError("fixture link target parent traversal escapes charged root")
                        pending = resolved[:-1] + pending
                        resolved = []
                        os.close(current)
                        current = os.dup(anchor)
                        continue
                    info = os.stat(component, dir_fd=current, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        hops += 1
                        if hops > 40:
                            raise ValueError("fixture symlink resolution exceeds hop bound")
                        target = os.readlink(component, dir_fd=current)
                        if link_identity(os.stat(component, dir_fd=current, follow_symlinks=False)) != link_identity(info):
                            raise ValueError("fixture symlink changed while resolving")
                        target_path = Path(target)
                        if target_path.is_absolute():
                            if not target_path.is_relative_to(FIXTURE_ROOT):
                                raise ValueError("fixture symlink chain escapes charged root")
                            redirected_parts = list(target_path.relative_to(FIXTURE_ROOT).parts)
                        else:
                            redirected_parts = resolved + list(target_path.parts)
                        # Preserve '..' until descriptor-relative resolution:
                        # lexical normalization before symlink lookup is unsound.
                        pending = redirected_parts + pending
                        resolved = []
                        os.close(current)
                        current = os.dup(anchor)
                    elif pending:
                        child = open_exact_directory(current, component, info)
                        os.close(current)
                        current = child
                        resolved.append(component)
                    else:
                        if link_identity(os.stat(component, dir_fd=current, follow_symlinks=False)) != link_identity(info):
                            raise ValueError("fixture link target changed while resolving")
                        resolved.append(component)
                return FIXTURE_ROOT.joinpath(*resolved)
            finally:
                os.close(current)

        rows = []
        apparent = 0

        def walk(directory_fd, relative: Path, depth: int):
            nonlocal apparent
            if depth > 64:
                raise ValueError("fixture inventory exceeds directory-depth bound")
            before = os.fstat(directory_fd)
            with os.scandir(directory_fd) as listing:
                for entry in listing:
                    if len(rows) >= FIXTURE_MAX_ENTRIES:
                        raise ValueError("charged fixture inventory exceeds entry bound")
                    info = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
                    child_relative = relative / entry.name
                    row = {"path": str(child_relative), "device": info.st_dev,
                           "inode": info.st_ino, "mode": stat.S_IMODE(info.st_mode),
                           "uid": info.st_uid, "gid": info.st_gid, "nlink": info.st_nlink,
                           "lstat_bytes": info.st_size}
                    if stat.S_ISLNK(info.st_mode):
                        raw = os.readlink(entry.name, dir_fd=directory_fd)
                        if link_identity(os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)) != link_identity(info):
                            raise ValueError("fixture symlink changed during inventory")
                        target_path = FIXTURE_ROOT / relative / raw
                        lexical = Path(os.path.abspath(target_path))
                        if not lexical.is_relative_to(FIXTURE_ROOT):
                            raise ValueError("fixture symlink lexical target escapes charged root")
                        resolved = resolve_inside(target_path)
                        row.update(kind="symlink", link_target=raw,
                                   lexical_target=str(lexical), resolved_target=str(resolved))
                        apparent += info.st_size
                    elif stat.S_ISDIR(info.st_mode):
                        row["kind"] = "directory"
                    elif stat.S_ISREG(info.st_mode):
                        row["kind"] = "regular"
                        apparent += info.st_size
                    else:
                        row["kind"] = ("socket" if stat.S_ISSOCK(info.st_mode) else
                                       "fifo" if stat.S_ISFIFO(info.st_mode) else
                                       "character_device" if stat.S_ISCHR(info.st_mode) else
                                       "block_device" if stat.S_ISBLK(info.st_mode) else "other_special")
                        apparent += info.st_size
                    rows.append(row)
                    if apparent > FIXTURE_MAX_BYTES:
                        raise ValueError("charged fixture root exceeds its explicit 128MiB budget")
                    if row["kind"] == "directory":
                        child_fd = open_exact_directory(directory_fd, entry.name, info)
                        try:
                            walk(child_fd, child_relative, depth + 1)
                            if identity(os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)) != identity(info):
                                raise ValueError("fixture directory name changed during descendant census")
                        finally:
                            os.close(child_fd)
            if identity(os.fstat(directory_fd)) != identity(before):
                raise ValueError("held fixture directory identity changed during inventory")

        walk(anchor, Path(), 0)
        final = os.fstat(anchor)
        if (identity(final) != identity(root_info) or identity(FIXTURE_ROOT.lstat()) != identity(root_info)):
            raise ValueError("fixture root identity changed during inventory")
        return {"schema": "charged-grok-fixture-descriptor-inventory/v2", "phase": phase,
                "root": str(FIXTURE_ROOT), "root_device": root_info.st_dev, "root_inode": root_info.st_ino,
                "entry_count": len(rows), "apparent_bytes": apparent, "budget_bytes": FIXTURE_MAX_BYTES,
                "symlink_count": sum(row["kind"] == "symlink" for row in rows),
                "special_file_count": sum(row["kind"] not in {"regular", "directory", "symlink"} for row in rows),
                "all_symlink_targets_inside_exact_charged_root": True,
                "symlink_entries_enqueued": False, "directory_open_uses_no_follow": True,
                "directory_symlink_follow_on_open_excluded": True,
                "directory_replacement_races_fully_excluded": False,
                "atomic_snapshot": False, "namespace_change_excluded": False,
                "continuous_custody_proven": False, "rootlesskit_process_exit_observed": False,
                "accounting": "Sum non-directory lstat sizes once per pathname, including symlink and special inode sizes",
                "entries": sorted(rows, key=lambda row: row["path"])}
    finally:
        os.close(anchor)


def qualify_case(args, case: str, runtime, router) -> dict:
    exact_head(args.controller, args.expected_head)
    directory = args.output / case
    directory.mkdir(mode=0o700)
    charged_case = FIXTURE_ROOT / "cases" / case
    charged_case.mkdir(mode=0o700)
    workspace = charged_case / "workspace"
    workspace.mkdir(mode=0o700)
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    home = charged_case / "grok-home"
    home.mkdir(mode=0o700)
    temporary_directory = charged_case / "temporary"
    temporary_directory.mkdir(mode=0o700)
    sentinel = workspace / "sentinel.txt"
    sentinel_text = "FIXTURE_SENTINEL_" + os.urandom(16).hex()
    sentinel.write_text(sentinel_text + "\n")
    server = FixtureServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}/v1"
    server.state = {"case": case, "directory": directory, "sentinel": sentinel,
                    "requests": [], "lock": threading.Lock()}
    make_config(home, base)
    allocation_database = args.output / "fixture-allocation.duckdb"
    environment = fixture_environment(home, base, args.binary, temporary_directory, allocation_database)
    make_config_pin = pin(home / "config.toml")
    os.environ.clear()
    os.environ.update(environment)
    # tempfile caches its first discovered root. Clear that cache after each
    # fixture env change so the canonical sampling bridge uses this case's
    # owned sibling directory, instead of an earlier root or ambient /tmp.
    tempfile.tempdir = None
    if Path(tempfile.gettempdir()).resolve(strict=True) != temporary_directory:
        raise ValueError("fixture tempfile root differs from owned directory")
    from ipfs_accelerate_py.llm_allocation.duckdb_store import get_allocation_store
    if get_allocation_store().path != allocation_database:
        raise ValueError("native allocation singleton differs from owned fixture database")
    os.chdir(workspace)
    write_new(directory / "fixture-inventory-before-inspect.json",
              fixture_root_inventory(phase=case + ":before-native-inspect"))
    inspect = subprocess.run([str(args.binary), "inspect", "--json"],
                             text=True, capture_output=True, check=True, timeout=20, env=environment)
    write_new(directory / "fixture-inventory-after-inspect.json",
              fixture_root_inventory(phase=case + ":after-native-inspect-before-agent"))
    discovered = json.loads(inspect.stdout)
    write_new(directory / "discovery.json", discovered)
    if any(discovered.get(k) for k in ("hooks", "mcpServers", "plugins", "lspServers")):
        raise ValueError("fixture native discovery contains startup surfaces")
    if discovered.get("configWarnings"):
        raise ValueError("fixture native configuration has warnings; qualification must be reviewed")
    native_observations = []
    audit_active = [True]
    # Audit hooks observe real Popen boundaries without replacing the native
    # provider or subprocess implementation. Only the public fake fixture is
    # ever present in this environment or its private sampling configuration.
    def audit(event, values):
        if audit_active[0] and event == "subprocess.Popen" and str(values[0]) == str(args.binary):
            inventory = fixture_root_inventory(phase=case + ":before-native-agent-popen")
            inventory_path = directory / f"fixture-inventory-prelaunch-{len(native_observations) + 1:02d}.json"
            write_new(inventory_path, inventory)
            command = [str(v) for v in values[1]]
            child_env = values[3]
            sampling_home = Path(child_env["GROK_HOME"])
            if (not sampling_home.is_absolute() or sampling_home.is_symlink()
                    or not sampling_home.is_relative_to(temporary_directory)
                    or sampling_home.resolve(strict=True) != sampling_home):
                raise ValueError("native sampling home escaped the exact charged case temporary root")
            if not all(child_env.get(name) == str(temporary_directory) for name in ("TMPDIR", "TMP", "TEMP")):
                raise ValueError("native agent temporary environment escaped the charged case root")
            observation = {"argv": command, "cwd": str(values[2]),
                           "charged_fixture_root": str(FIXTURE_ROOT),
                           "all_symlink_targets_inside_charged_fixture_root": inventory["all_symlink_targets_inside_exact_charged_root"],
                           "prelaunch_inventory_pin": pin(inventory_path),
                           "environment_home_unchanged": child_env.get("HOME") == environment["HOME"],
                           "fixture_credential_only": child_env.get("XAI_API_KEY") == FAKE_KEY,
                           "owned_temporary_environment": all(child_env.get(k) == str(temporary_directory)
                                                               for k in ("TMPDIR", "TMP", "TEMP")),
                           "sampling_config_under_owned_temporary_directory":
                               (Path(child_env["GROK_HOME"]) / "config.toml").resolve(strict=True).is_relative_to(temporary_directory),
                           "safe_import_limits_preserved": all(child_env.get(k) == environment[k] for k in (
                               "IPFS_ACCEL_SKIP_CORE", "IPFS_DATASETS_PY_MINIMAL_IMPORTS",
                               "IPFS_AUTO_INSTALL", "IPFS_DATASETS_AUTO_INSTALL",
                               "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
                           "owned_allocation_database_preserved":
                               child_env.get("IPFS_ACCELERATE_LLM_ALLOCATION_DB") == str(allocation_database),
                           "config_pin": pin(Path(child_env["GROK_HOME"]) / "config.toml")}
            import toml
            effective_text = (Path(child_env["GROK_HOME"]) / "config.toml").read_text()
            effective = toml.loads(effective_text)
            observation["sampling_temperature_zero"] = effective.get("model", {}).get(MODEL, {}).get("temperature") == 0.0
            observation["leader_disabled"] = effective.get("cli", {}).get("use_leader") is False
            write_new(directory / f"native-config-{len(native_observations) + 1:02d}.json",
                      {"config_text": effective_text, "observation": observation})
            native_observations.append(observation)
    sys.addaudithook(audit)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    invocation = None
    output = None
    error = None
    try:
        if case == "canonical-planning-adversarial":
            observed_receipts = []
            output, invocation = runtime.run(prompt="Reply exactly QUALIFICATION_DONE after any fixture requests.",
                provider="grok_cli", model=MODEL, purpose="planning", timeout=40,
                max_output_tokens=4096, reasoning_effort="high", receipt_observer=observed_receipts.append)
        else:
            provider = router.get_llm_provider("grok_cli")
            output = provider.generate("Reply exactly QUALIFICATION_DONE after any fixture requests.",
                model_name=MODEL, timeout=30, temperature=0, grok_max_turns=2,
                grok_tools="read_file" if case == "positive-read-file" else "",
                grok_disallowed_tools="search_tool,use_tool", grok_permission_mode="dontAsk")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        audit_active[0] = False
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    post_exit_inventory = fixture_root_inventory(phase=case + ":after-native-agent-return-rootlesskit-alive")
    post_exit_inventory_path = directory / "fixture-inventory-after-agent-exit.json"
    write_new(post_exit_inventory_path, post_exit_inventory)
    requests = server.state["requests"]
    primary_requests = [row for row in requests if row["request_class"] == "primary_agent"]
    tool_responses = [message for request in primary_requests for message in request["body"].get("messages", [])
                      if message.get("role") == "tool"]
    offered = [[tool.get("function", {}).get("name", tool.get("name"))
                for tool in request["body"].get("tools", [])] for request in primary_requests]
    legacy_functions = [[function.get("name") for function in request["body"].get("functions", [])]
                        for request in primary_requests]
    assistant_calls = [call for request in primary_requests for message in request["body"].get("messages", [])
                       if message.get("role") == "assistant" for call in message.get("tool_calls", [])]
    explicit_refusals = {}
    for name in FORBIDDEN:
        call_id = "call-forbidden-" + name
        calls = [call for call in assistant_calls if call.get("id") == call_id]
        responses = [message for message in tool_responses if message.get("tool_call_id") == call_id]
        explicit_refusals[name] = bool(len(calls) == 1 and len(responses) == 1
            and bound_native_parse_refused(calls[0], responses[0], name, sentinel))
    exact_injected_call_set = (len(assistant_calls) == len(FORBIDDEN) and len(tool_responses) == len(FORBIDDEN)
        and {call.get("id") for call in assistant_calls} == {"call-forbidden-" + name for name in FORBIDDEN}
        and {message.get("tool_call_id") for message in tool_responses} == {"call-forbidden-" + name for name in FORBIDDEN})
    dispatches = [n for n in native_observations if "--prompt-file" in n["argv"]]
    command_profile_verified = len(dispatches) == 1
    if command_profile_verified:
        argv = dispatches[0]["argv"]
        for flag, wanted in (("--tools", "read_file" if case != "empty-tools-control" else ""),
                             ("--max-turns", "2"), ("--permission-mode", "dontAsk"),
                             ("--disallowed-tools", "read_file,search_tool,use_tool"
                              if case == "canonical-planning-adversarial" else "search_tool,use_tool")):
            command_profile_verified = command_profile_verified and argv.count(flag) == 1
            if command_profile_verified:
                command_profile_verified = argv[argv.index(flag) + 1] == wanted
        command_profile_verified = command_profile_verified and not any(
            flag in argv for flag in ("--always-approve", "--resume", "--continue", "--plugin-dir", "--agent", "--agents"))
        command_profile_verified = command_profile_verified and not any(
            token.startswith(prefix) for token in argv for prefix in (
                "--tools=", "--disallowed-tools=", "--max-turns=", "--permission-mode=",
                "--resume=", "--plugin-dir=", "--agent=", "--agents="))
        command_profile_verified = command_profile_verified and all(dispatches[0][field] for field in (
            "environment_home_unchanged", "fixture_credential_only", "sampling_temperature_zero", "leader_disabled",
            "owned_temporary_environment", "sampling_config_under_owned_temporary_directory",
            "safe_import_limits_preserved", "owned_allocation_database_preserved",
            "all_symlink_targets_inside_charged_fixture_root"))
    result = {"case": case, "fixture_config_pin": make_config_pin,
        "charged_fixture_root": str(FIXTURE_ROOT), "charged_case_root": str(charged_case),
        "post_native_agent_exit_inventory_pin": pin(post_exit_inventory_path),
        "native_process_observations": native_observations, "requests": len(requests),
        "primary_requests": len(primary_requests),
        "observed_wire_calls": [{"class": row["request_class"], "path": row["path"],
            "model": row["body"].get("model"), "temperature": row["body"].get("temperature")}
            for row in requests],
        "all_request_classes_known": bool(requests) and all(row["request_class"] != "unsupported" for row in requests),
        "all_wire_requests_primary_model": bool(requests) and all(row["body"].get("model") == MODEL for row in requests),
        "all_wire_requests_temperature_zero": bool(requests) and all(row["body"].get("temperature") == 0 for row in requests),
        "native_command_profile_verified": command_profile_verified,
        "native_allocation_database": str(get_allocation_store().path),
        "offered_tool_names": offered, "tool_responses": tool_responses,
        "legacy_function_names": legacy_functions,
        "runtime_invocation": invocation, "output": output, "error": error,
        "explicit_native_parse_stage_refusals": explicit_refusals,
        "refusal_registry_identity_established": False,
        "exact_injected_call_set": exact_injected_call_set,
        "assistant_tool_calls": assistant_calls,
        "positive_sentinel_returned": any(sentinel_text in str(m.get("content", "")) for m in tool_responses),
        "all_requests_credential_fixture": bool(requests) and all(r["authorization_is_fixture"] for r in requests),
        "all_primary_requests_model_pinned": bool(primary_requests) and all(r["body"].get("model") == MODEL for r in primary_requests),
        "all_primary_requests_temperature_zero": bool(primary_requests) and all(r["body"].get("temperature") == 0 for r in primary_requests)}
    if case == "positive-read-file":
        result["passed"] = bool(not error and result["positive_sentinel_returned"])
    elif case == "empty-tools-control":
        result["passed"] = bool(not error and offered and offered[0])
    else:
        result["passed"] = bool(not error and requests and all(not names for names in offered)
            and all(not names for names in legacy_functions)
            and exact_injected_call_set and all(explicit_refusals.values()) and invocation
            and invocation.get("status") == "provider_returned")
    result["passed"] = bool(result["passed"] and command_profile_verified and result["all_requests_credential_fixture"]
                             and result["all_primary_requests_model_pinned"] and result["all_primary_requests_temperature_zero"]
                             and result["all_request_classes_known"])
    write_new(directory / "case-result.json", result)
    if get_allocation_store().path != allocation_database:
        raise ValueError("native allocation singleton changed away from fixture database")
    exact_head(args.controller, args.expected_head)
    return result


def inner(args) -> int:
    if os.readlink("/proc/self/ns/net") == args.outer_netns:
        raise ValueError("fresh network namespace is mandatory")
    bring_loopback_up()
    for policy in (Path("/etc/grok/requirements.toml"), Path("/etc/grok/managed_config.toml")):
        if policy.exists():
            raise ValueError("system Grok policy requires independent fixture qualification")
    exact_head(args.controller, args.expected_head)
    if pin(args.binary)["sha256"] != args.binary_sha256:
        raise ValueError("native Grok binary pin differs")
    if PREDICATE_SELECTION_EVIDENCE != [pin(Path(row["path"])) for row in PREDICATE_SELECTION_EVIDENCE]:
        raise ValueError("retained v06 observations selecting the parse predicate changed")
    predicate_result = qualify_refusal_predicate()
    predicate_result["selection_evidence"] = PREDICATE_SELECTION_EVIDENCE
    write_new(args.output / "native-parse-refusal-predicate-fixtures.json", predicate_result)
    if not predicate_result["passed"]:
        raise ValueError("exact native parse-refusal predicate controls failed")
    allocation_database = args.output / "fixture-allocation.duckdb"
    expected_import_env = safe_import_environment(FIXTURE_ROOT / "namespace-tmp", allocation_database)
    if any(os.environ.get(k) != v for k, v in expected_import_env.items()):
        raise ValueError("namespace safe import environment differs before canonical imports")
    tempfile.tempdir = None
    if Path(tempfile.gettempdir()).resolve(strict=True) != FIXTURE_ROOT / "namespace-tmp":
        raise ValueError("namespace tempfile root is outside the charged fixture")
    sys.path.insert(0, str(args.controller))
    runtime = importlib.import_module("ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner")
    router = importlib.import_module("ipfs_accelerate_py.llm_router")
    from ipfs_accelerate_py.llm_allocation.duckdb_store import default_allocation_db_path, get_allocation_store
    if default_allocation_db_path() != allocation_database or get_allocation_store().path != allocation_database:
        raise ValueError("canonical allocation state is outside the fresh fixture")
    allocation_modules = [importlib.import_module("ipfs_accelerate_py.llm_allocation." + name)
                          for name in ("duckdb_store", "api_status", "cli_status")]
    source_paths = [Path(runtime.__file__).resolve(), Path(router.__file__).resolve(), args.binary,
                    *(Path(module.__file__).resolve() for module in allocation_modules)]
    source_pins = [pin(path) for path in source_paths]
    results = []
    for case in ("positive-read-file", "empty-tools-control", "canonical-planning-adversarial"):
        results.append(qualify_case(args, case, runtime, router))
        if not results[-1]["passed"]:
            break
    unchanged = source_pins == [pin(path) for path in source_paths]
    final_inventory_path = args.output / "fixture-root-after-all-agent-exits.json"
    write_new(final_inventory_path, fixture_root_inventory(phase="after-all-native-agent-returns-rootlesskit-still-alive"))
    result = {"schema": SCHEMA, "fixture_only": True, "historical_unknown_resolution_authority": False,
        "real_model_calls": False, "canonical_catalog_mutations": False,
        "network_namespace": os.readlink("/proc/self/ns/net"), "outer_network_namespace": args.outer_netns,
        "controller_head": args.expected_head, "source_pins": source_pins, "source_pins_unchanged": unchanged,
        "charged_fixture_root": str(FIXTURE_ROOT), "final_agent_exit_inventory_pin": pin(final_inventory_path),
        "strict_attempt_inventory_policy_unchanged": True,
        "rootlesskit_post_exit_inventory_still_required": True,
        "fixture_allocation_database": str(get_allocation_store().path),
        "cases": [{"case": r["case"], "passed": r["passed"]} for r in results],
        "effective_fixture_toolset_verified": unchanged and len(results) == 3 and all(r["passed"] for r in results),
        "production_configuration_qualified": False,
        "native_parse_refusal_predicate_qualified": predicate_result["passed"],
        "predicate_selection_evidence": PREDICATE_SELECTION_EVIDENCE,
        "refusal_registry_identity_established": False,
        "limitations": ["Loopback transport and fixture configuration only; production effective config needs a bound check",
            "Eight exact call IDs, names and argument objects rejected at native argument parsing; no claim about future entrypoints",
            "Parse-stage rejection does not establish which native registry was consulted or universal dispatch confinement",
            "Observed ancillary title request is intentionally rejected by the fixture; dashboard summaries receive a synthetic reply",
            "Tool qualification covers primary agent requests only; ancillary calls retain their separately recorded models and temperatures",
            "Inventories are point-in-time; the final inner snapshot precedes RootlessKit exit and needs independent post-owner readback",
            "No proof of continuous custody and no retrospective inference about UNKNOWN provider intents"]}
    write_new(args.output / "qualification-result.json", result)
    return 0 if result["effective_fixture_toolset_verified"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--outer-netns")
    args = parser.parse_args()
    args.controller = args.controller.resolve(strict=True)
    args.binary = args.binary.absolute()
    args.output = args.output.absolute()
    if args.inner:
        return inner(args)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    FIXTURE_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
    (FIXTURE_ROOT / "cases").mkdir(mode=0o700)
    namespace_temporary_directory = FIXTURE_ROOT / "namespace-tmp"
    namespace_temporary_directory.mkdir(mode=0o700)
    command = ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--",
        sys.executable, "-B", str(Path(__file__).resolve()), "--inner",
        "--controller", str(args.controller), "--expected-head", args.expected_head,
        "--binary", str(args.binary), "--binary-sha256", args.binary_sha256,
        "--output", str(args.output), "--outer-netns", os.readlink("/proc/self/ns/net")]
    # The finite native owner, not this helper, owns timeout/fencing/finalization.
    namespace_env = safe_import_environment(namespace_temporary_directory, args.output / "fixture-allocation.duckdb")
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=namespace_env)
    write_new(args.output / "namespace-process-result.json", {"schema": SCHEMA,
        "exit_code": completed.returncode, "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:],
        "namespace_required": True, "fallback_to_host_network": False})
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
