"""Pure, explicit source-display policy shared by inventory and owner adapters.

This module chooses codec arguments only. Callers must independently admit the
path/source population and bind policy identities to their source generation.
Existing codec manifests and their exact restoration rules remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .reversible_source_view import build_source_view, current_codec_binding

SCHEMA = "supervisor-source-view-policy@1"
BINDING_SCHEMA = "supervisor-source-view-policy-binding@1"
PROFILES = frozenset({"identity@1", "python-whitespace@1", "python-comments@1"})
MAX_PATHS = 256
MAX_LINES = 4096
MAX_PATH_HEX_CHARS = 8192
MAX_LINE = 1048576


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

    def to_dict(self):
        """Return a detached versioned policy, with no source or task authority."""
        self._validate()
        return dict(schema=SCHEMA, profile=self.profile,
            protected_lines={path: list(lines) for path, lines in self.protected_lines})

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


__all__ = ["SourceViewPolicy", "ResolvedSourceViewPolicy", "SCHEMA", "BINDING_SCHEMA"]
