"""Pure, explicit source-display policy shared by inventory and owner adapters.

This module chooses codec arguments only. Callers must independently admit the
path/source population and bind policy identities to their source generation.
Existing codec manifests and their exact restoration rules remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import uuid

from .reversible_source_view import build_source_view, current_codec_binding

SCHEMA = "supervisor-source-view-policy@1"
BINDING_SCHEMA = "supervisor-source-view-policy-binding@1"
PROFILES = frozenset({"identity@1", "python-whitespace@1", "python-comments@1"})
MAX_PATHS = 256
MAX_LINES = 4096
MAX_PATH_HEX_CHARS = 8192
MAX_LINE = 1048576
MAX_POLICY_JSON_BYTES = 65536
SOURCE_VIEW_POLICY_FLAG = "--source-view-policy-json"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _profile(value):
    _need(type(value) is str and value in PROFILES, "unsupported source-view policy profile")


def _raw_path_hex(value):
    _need(type(value) is str and 0 < len(value) <= MAX_PATH_HEX_CHARS,
        "invalid protected raw path")
    try:
        canonical = bytes.fromhex(value).hex() == value
    except ValueError:
        canonical = False
    _need(canonical, "invalid protected raw path")


def _lines(value):
    _need(type(value) is tuple and 0 < len(value) <= MAX_LINES
        and all(type(line) is int and 1 <= line <= MAX_LINE for line in value),
        "invalid protected source lines")
    _need(value == tuple(sorted(set(value))), "noncanonical protected source lines")


@dataclass(frozen=True)
class ResolvedSourceViewPolicy:
    """Codec arguments for one path; this grants no path or source admission."""

    python_eligible: bool
    profile: str
    protected_lines: tuple[int, ...]

    def __post_init__(self):
        _need(type(self) is ResolvedSourceViewPolicy
            and type(self.python_eligible) is bool, "invalid resolved source-view policy")
        _profile(self.profile)
        _need(type(self.protected_lines) is tuple, "invalid resolved protected source lines")
        if self.protected_lines:
            _lines(self.protected_lines)
        _need(self.python_eligible or (self.profile == "identity@1" and not self.protected_lines),
            "non-Python source requires unprotected identity policy")


@dataclass(frozen=True)
class SourceViewPolicy:
    """Canonical immutable caller protection and Python display selection."""

    profile: str = "python-whitespace@1"
    protected_lines: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self):
        self._validate()

    def _validate(self):
        _need(type(self) is SourceViewPolicy, "exact source-view policy required")
        _profile(self.profile)
        records = self.protected_lines
        _need(type(records) is tuple and len(records) <= MAX_PATHS,
            "protected source policy exceeds bounds")
        previous, total = None, 0
        for record in records:
            _need(type(record) is tuple and len(record) == 2, "invalid protected source policy entry")
            path, lines = record
            _raw_path_hex(path)
            _lines(lines)
            _need(previous is None or previous < path, "noncanonical protected source paths")
            previous = path
            total += len(lines)
            _need(total <= MAX_LINES, "protected source lines exceed bounds")

    @classmethod
    def from_mapping(cls, *, profile="python-whitespace@1", protected_lines=None):
        """Detach the existing raw-path-hex mapping without sorting line values."""
        _need(cls is SourceViewPolicy, "exact source-view policy class required")
        _profile(profile)
        if protected_lines is None:
            return cls(profile=profile)
        _need(type(protected_lines) is dict and len(protected_lines) <= MAX_PATHS,
            "protected source policy exceeds bounds")
        records, total = [], 0
        for path, values in protected_lines.items():
            _raw_path_hex(path)
            _need(type(values) in {list, tuple} and 0 < len(values) <= MAX_LINES,
                "invalid protected source lines")
            lines = tuple(values)
            _lines(lines)
            total += len(lines)
            _need(total <= MAX_LINES, "protected source lines exceed bounds")
            records.append((path, lines))
        return cls(profile=profile, protected_lines=tuple(sorted(records)))

    @classmethod
    def from_dict(cls, value):
        """Read the closed versioned recipe, detaching caller-owned containers."""
        _need(cls is SourceViewPolicy, "exact source-view policy class required")
        _need(type(value) is dict and set(value) == {"schema", "profile", "protected_lines"}
            and type(value["schema"]) is str and value["schema"] == SCHEMA,
            "closed source-view policy recipe required")
        _need(type(value["protected_lines"]) is dict, "protected source mapping required")
        return cls.from_mapping(profile=value["profile"], protected_lines=value["protected_lines"])

    @classmethod
    def from_json(cls, value):
        """Read bounded strict JSON; repeated keys and nonfinite values refuse."""
        _need(cls is SourceViewPolicy, "exact source-view policy class required")
        _need(type(value) is str, "exact source-view policy JSON text required")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError as error:
            raise ValueError("source-view policy JSON must be UTF-8") from error
        _need(0 < size <= MAX_POLICY_JSON_BYTES, "source-view policy JSON exceeds bounds")

        def unique(pairs):
            result = {}
            for key, item in pairs:
                _need(key not in result, "duplicate source-view policy JSON key")
                result[key] = item
            return result

        def refuse_constant(_):
            raise ValueError("nonfinite source-view policy JSON value")

        try:
            return cls.from_dict(json.loads(value, object_pairs_hook=unique,
                parse_constant=refuse_constant))
        except (ValueError, RecursionError) as error:
            raise ValueError("invalid source-view policy JSON") from error

    def to_dict(self):
        """Return a detached versioned policy, with no source or task authority."""
        self._validate()
        return dict(schema=SCHEMA, profile=self.profile,
            protected_lines={path: list(lines) for path, lines in self.protected_lines})

    def to_json(self):
        """Return one canonical, bounded representation for configuration/argv."""
        result = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        _need(len(result.encode("utf-8")) <= MAX_POLICY_JSON_BYTES,
            "source-view policy JSON exceeds bounds")
        return result

    def binding(self):
        """Bind caller configuration and local resolver/codec implementation bytes."""
        self._validate()
        policy = self.to_dict()
        result = dict(schema=BINDING_SCHEMA, policy=policy,
            resolver_sha256=hashlib.sha256(Path(__file__).resolve(strict=True).read_bytes()).hexdigest(),
            codec_binding=current_codec_binding())
        _need(self.to_dict() == policy, "source-view policy changed during binding")
        return result

    def resolve(self, *, raw_path: bytes, regular_file: bool) -> ResolvedSourceViewPolicy:
        """Resolve exact raw path bytes; caller-supplied file classification is not verified."""
        self._validate()
        _need(type(raw_path) is bytes and 0 < len(raw_path) <= MAX_PATH_HEX_CHARS // 2,
            "bounded exact raw source path required")
        _need(type(regular_file) is bool, "exact regular-file classification required")
        python_eligible = regular_file and raw_path.endswith((b".py", b".pyi"))
        lines = dict(self.protected_lines).get(raw_path.hex(), ())
        _need(python_eligible or not lines,
            "protected policy differs from regular Python source scope")
        return ResolvedSourceViewPolicy(python_eligible=python_eligible,
            profile=self.profile if python_eligible else "identity@1", protected_lines=lines)

    def project(self, source: bytes, *, raw_path: bytes, regular_file: bool):
        """Project with the existing codec, refusing configuration or implementation drift."""
        self._validate()
        _need(type(source) is bytes, "source must be exact bytes")
        before = self.binding()
        resolved = self.resolve(raw_path=raw_path, regular_file=regular_file)
        result = build_source_view(source, profile=resolved.profile,
            protected_lines=resolved.protected_lines)
        _need(self.binding() == before, "source-view policy or implementation changed during projection")
        return result


__all__ = ["SourceViewPolicy", "ResolvedSourceViewPolicy", "SCHEMA", "BINDING_SCHEMA",
           "MAX_POLICY_JSON_BYTES", "SOURCE_VIEW_POLICY_FLAG"]


# Optional checkout-helper advice; separate from the structured codec recipe above.
POLICIES = ("legacy", "source-view-raw@1", "source-view-compact@1")


def validate_policy(value: str, *, purpose: str, provider: str) -> str:
    if type(value) is not str or value not in POLICIES:
        raise ValueError("explicit supported source view policy required")
    if value != "legacy" and (purpose != "coding" or provider != "codex_cli"):
        raise ValueError("source view requires a Codex coding route")
    return value


def _path(value):
    if (type(value) is not str or not 1 <= len(value.encode()) <= 4096
            or any(ord(char) < 32 for char in value)):
        raise ValueError("bounded absolute source helper path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("canonical absolute source helper path required")
    return path


def _instruction(binding: dict) -> str:
    command = " ".join(shlex.quote(binding[key]) for key in ("python_executable", "helper_script"))
    command += " --root " + shlex.quote(binding["workspace"])
    command += " --audit-store " + shlex.quote(binding["audit_store"])
    representation = "raw" if binding["requested_policy"] == "source-view-raw@1" else "compact"
    return (
        "Optional source views and hash-checked edits:\n"
        "For explicit visible Python files, inspect symbols with the helper; execution uses the original checkout.\n"
        f"{command} index PATH...\n"
        f"{command} read PATH --symbol NAME --format {representation}\n"
        "Use symbol module for whole-file reads. Source views are bounded; retrieve necessary original detail "
        "with normal tools if unavailable. They do not prove behavior. To replace a supported symbol, "
        "write the complete replacement definition to a visible relative file, then use the source_sha256 from your read:\n"
        f"{command} replace PATH --symbol NAME --expected-sha256 HASH --replacement-file PATH --format {representation}\n"
        "Stale edits are rejected. Inspect the canonical diff, refresh changed views, and run the required checks. "
        "This does not expand allowed files or establish task completion. Helper use and token savings are measured separately.\n\n"
    )


def create_binding(*, workspace: Path, requested_policy: str) -> dict:
    validate_policy(requested_policy, purpose="coding", provider="codex_cli")
    if requested_policy == "legacy":
        raise ValueError("legacy policy has no source view binding")
    helper = Path(__file__).with_name("code_source_view.py").resolve(strict=True)
    if not helper.is_file() or helper.stat().st_size > 256_000:
        raise ValueError("bounded source helper source required")
    binding = {
        "schema": SCHEMA, "requested_policy": requested_policy,
        "python_executable": str(Path(sys.executable).absolute()),
        "helper_script": str(helper), "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "workspace": str(workspace),
        "audit_store": "/tmp/supervisor-source-view-" + uuid.uuid4().hex + "/operations.jsonl",
        "native_tool_interception": False, "helper_adoption_guaranteed": False,
        "token_savings_measured": False, "execution_authority": False, "completion_authority": False,
    }
    instruction = _instruction(binding)
    binding.update(instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                   instruction_bytes=len(instruction.encode()))
    validate_binding(binding, workspace=workspace)
    return binding


def validate_binding(binding: dict, *, workspace: Path) -> str:
    fields = {"schema", "requested_policy", "python_executable", "helper_script", "helper_sha256",
              "workspace", "audit_store", "native_tool_interception", "helper_adoption_guaranteed",
              "token_savings_measured", "execution_authority", "completion_authority",
              "instruction_sha256", "instruction_bytes"}
    if (type(binding) is not dict or set(binding) not in (fields, fields | {"usage"})
            or binding.get("schema") != SCHEMA
            or type(binding.get("requested_policy")) is not str
            or binding["requested_policy"] not in POLICIES[1:]
            or any(binding.get(key) is not False for key in ("native_tool_interception", "helper_adoption_guaranteed",
                "token_savings_measured", "execution_authority", "completion_authority"))):
        raise ValueError("invalid source view policy binding")
    python, helper, recorded_workspace, audit = (_path(binding[key]) for key in
        ("python_executable", "helper_script", "workspace", "audit_store"))
    if (helper.name != "code_source_view.py" or recorded_workspace != workspace
            or helper.is_relative_to(workspace) or audit.is_relative_to(workspace)
            or re.fullmatch(r"/tmp/supervisor-source-view-[0-9a-f]{32}/operations.jsonl", str(audit)) is None
            or type(binding["helper_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", binding["helper_sha256"]) is None):
        raise ValueError("invalid source view helper identity")
    instruction = _instruction(binding)
    if (type(binding["instruction_bytes"]) is not int
            or binding["instruction_bytes"] != len(instruction.encode())
            or binding["instruction_bytes"] > 8192
            or binding["instruction_sha256"] != hashlib.sha256(instruction.encode()).hexdigest()):
        raise ValueError("source view instruction binding differs")
    return instruction


def create_audit_store(binding: dict) -> tuple[int, int]:
    """Create a private invocation directory outside the task output tree."""
    import os
    directory = Path(binding["audit_store"]).parent
    directory.mkdir(mode=0o700)
    info = directory.lstat()
    return info.st_dev, info.st_ino


def finish_audit_store(binding: dict, *, workspace: Path, expected_identity) -> dict:
    """Export bounded content-free observations and remove only the known log."""
    import os
    import stat
    usage = {"status": "unavailable", "helper_operations_observed": None,
             "store_cleanup_complete": False, "artifacts_retained_after_invocation": None}
    if expected_identity is None:
        return usage
    audit = Path(binding["audit_store"])
    descriptor = None
    try:
        descriptor = os.open(audit.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if ((info.st_dev, info.st_ino) != expected_identity or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("source view audit directory identity changed")
        try:
            from .code_source_view import summarize_audit_store
            usage = summarize_audit_store(audit, expected_root=workspace)
            usage["helper_operations_observed"] = (bool(usage["operations"])
                if usage.get("status") in {"observed", "not_observed"}
                and type(usage.get("operations")) is int else None)
        except Exception:
            pass
        # Never recurse through worker-written files. unlink does not follow links.
        try:
            os.unlink(audit.name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        current = audit.parent.lstat()
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ValueError("source view audit directory identity changed")
        audit.parent.rmdir()
        usage.update(store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    except Exception:
        usage.update(store_cleanup_complete=False, artifacts_retained_after_invocation=None,
                     store_cleanup_error="cleanup_unavailable")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return usage
