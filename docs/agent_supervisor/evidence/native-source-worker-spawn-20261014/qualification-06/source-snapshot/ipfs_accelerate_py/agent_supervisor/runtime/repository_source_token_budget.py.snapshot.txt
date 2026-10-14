"""Choose a repository reply using the owner-selected local tokenizer.

The caller supplies complete, already validated MCP JSON-RPC reply frames,
including their final newline. This module never rewrites content or loads a
tokenizer. Source equivalence and currentness remain the repository owner's
responsibility; local wire counts are not provider token usage.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .local_prompt_tokenizer import MAX_PROMPT_BYTES, OwnerPromptTokenizer

SCHEMA = "repository-source-token-selection@1"
MAX_REPLY_BYTES = 2 * 1024 * 1024 + 1
MAX_REPLY_TOKENS = 1_000_000
TOKENIZER_FIELDS = frozenset({"artifact", "expected_sha256", "provider", "model"})
COUNT_SCOPE = "complete_mcp_jsonrpc_reply_utf8_with_trailing_newline"
AUDIT_SCHEMA = "repository-source-token-selection-audit@1"
SUMMARY_SCHEMA = "repository-source-token-selection-summary@1"
MAX_AUDIT_RECORDS = 512
MAX_AUDIT_BYTES = 2 * 1024 * 1024
MAX_AUDIT_RECORD_BYTES = 32768


def validate_tokenizer_binding(value):
    """Validate inert child configuration without opening local assets.

    Loading and runtime revalidation belong to ``load_owner_prompt_tokenizer``.
    A configuration digest or this shape check cannot authenticate the files.
    """
    if type(value) is not dict or set(value) != TOKENIZER_FIELDS:
        raise ValueError("closed repository tokenizer binding required")
    artifact = value["artifact"]
    if (type(artifact) is not str or not 0 < len(artifact) <= 4096
            or any(ord(character) < 32 for character in artifact)
            or ":" in artifact or "\\" in artifact):
        raise ValueError("canonical absolute local tokenizer descriptor required")
    try:
        encoded = artifact.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("canonical absolute local tokenizer descriptor required") from error
    path = Path(artifact)
    if (len(encoded) > 4096 or not path.is_absolute() or str(path) != artifact
            or any(part in {".", ".."} for part in artifact.split("/"))):
        raise ValueError("canonical absolute local tokenizer descriptor required")
    for key in ("provider", "model"):
        item = value[key]
        if (type(item) is not str or not 0 < len(item) <= 256 or item.strip() != item
                or any(ord(character) < 32 for character in item)):
            raise ValueError("bounded tokenizer provider and model required")
        try:
            if len(item.encode("utf-8")) > 1024:
                raise ValueError("bounded tokenizer provider and model required")
        except UnicodeError as error:
            raise ValueError("bounded tokenizer provider and model required") from error
    if (type(value["expected_sha256"]) is not str
            or re.fullmatch(r"[a-f0-9]{64}", value["expected_sha256"]) is None):
        raise ValueError("exact repository tokenizer descriptor SHA256 required")
    return dict(value)


def _frame(text):
    if (type(text) is not str or not text.endswith("\n") or text.count("\n") != 1
            or not 1 < len(text) <= MAX_REPLY_BYTES):
        raise ValueError("one bounded complete repository reply frame required")
    try:
        raw = text.encode("utf-8", "strict")
    except UnicodeError as error:
        raise ValueError("repository reply must be exact UTF8") from error
    if len(raw) > MAX_REPLY_BYTES:
        raise ValueError("repository reply exceeds its wire byte bound")
    return {"reply_sha256": hashlib.sha256(raw).hexdigest(), "reply_bytes": len(raw), "tokens": None}


def select_repository_reply(*, original_reply: str, compact_reply: str | None,
                            requested_representation: str,
                            tokenizer: OwnerPromptTokenizer | None) -> dict:
    """Prefer compact only for a strict decrease in complete local wire tokens.

    Explicit original reads do not measure or inspect the unused compact input.
    Missing/invalid counters, changed assets and inputs outside the tokenizer's
    existing 256,000-byte bound retain original source with no token claim.
    At most two counts run per selection. Each uses the existing tokenizer's
    bounded pre/post asset verification; there is no discovery or download.
    Selection metadata must remain outside the counted model reply itself.
    """
    if type(requested_representation) is not str or requested_representation not in {"original", "compact"}:
        raise ValueError("explicit original or compact representation required")
    candidates = {"original": _frame(original_reply)}
    selected = "original"
    status, reason = "not_requested", "original_requested"
    binding = provenance = None
    if requested_representation == "compact":
        reason = "compact_unavailable"
        if compact_reply is not None:
            candidates["compact"] = _frame(compact_reply)
            status, reason = "unavailable", "tokenizer_unavailable"
            if tokenizer is not None:
                if type(tokenizer) is not OwnerPromptTokenizer:
                    status, reason = "failed", "invalid_tokenizer"
                elif any(row["reply_bytes"] > MAX_PROMPT_BYTES for row in candidates.values()):
                    reason = "tokenizer_input_bound"
                else:
                    try:
                        # Do not publish either count unless the complete pair
                        # succeeds. A first count cannot justify partial savings.
                        original_tokens = tokenizer.count(original_reply)
                        compact_tokens = tokenizer.count(compact_reply)
                        if any(type(count) is not int or not 1 <= count <= MAX_REPLY_TOKENS
                               for count in (original_tokens, compact_tokens)):
                            raise ValueError("invalid bounded local token count")
                        binding, provenance = tokenizer.binding(), tokenizer.provenance()
                    except Exception:
                        # Backend details may contain model input or local paths.
                        # Retain only a fixed content-free failure classification.
                        status, reason = "failed", "tokenizer_failed"
                        binding = provenance = None
                    else:
                        status, reason = "measured", "original_not_larger"
                        candidates["original"]["tokens"] = original_tokens
                        candidates["compact"]["tokens"] = compact_tokens
                        if compact_tokens < original_tokens:
                            selected, reason = "compact", "compact_strictly_fewer_tokens"
    chosen = candidates[selected]
    return {"schema": SCHEMA, "requested_representation": requested_representation,
        "selected_representation": selected, "selection_reason": reason,
        "measurement_status": status, "candidates": candidates,
        "selected_tokens": chosen["tokens"], "selected_reply_sha256": chosen["reply_sha256"],
        "selected_reply_bytes": chosen["reply_bytes"], "tokenizer_binding": binding,
        "tokenizer_provenance": provenance,
        "count_scope": COUNT_SCOPE,
        "max_tokenizer_input_bytes": MAX_PROMPT_BYTES, "max_reply_tokens": MAX_REPLY_TOKENS,
        "provider_usage_measured": False, "provider_framing_measured": False,
        "provider_tokenizer_equivalence_verified": False, "total_session_token_savings": None,
        "proof_authority": False, "completion_authority": False}


_SELECTION_FIELDS = frozenset({"schema", "requested_representation", "selected_representation",
    "selection_reason", "measurement_status", "candidates", "selected_tokens", "selected_reply_sha256",
    "selected_reply_bytes", "tokenizer_binding", "tokenizer_provenance", "count_scope",
    "max_tokenizer_input_bytes", "max_reply_tokens", "provider_usage_measured", "provider_framing_measured",
    "provider_tokenizer_equivalence_verified", "total_session_token_savings", "proof_authority",
    "completion_authority"})
_BINDING_FIELDS = frozenset({"provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"})
_PROVENANCE_FIELDS = frozenset({"schema", "backend", "descriptor_path", "descriptor_sha256", "asset_path",
    "asset_sha256", "tokenizers_version", "compatibility_basis", "completion_authority"})
_SUMMARY_COUNTERS = ("read_count", "compact_selected", "original_selected", "measured_comparisons",
    "unmeasured_fallbacks", "explicit_original", "selected_reply_tokens")
_SUMMARY_FIELDS = frozenset({"schema", "status", "audit_closed", "provider_usage_measured", *_SUMMARY_COUNTERS})


def _sha(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _bounded_text(value):
    return (type(value) is str and 0 < len(value) <= 4096 and value == value.strip()
            and not any(ord(character) < 32 for character in value)
            and len(value.encode("utf-8")) <= 16384)


def _validate_selection(value):
    """Check a closed local receipt; this does not reproduce tokenization."""
    if (type(value) is not dict or set(value) != _SELECTION_FIELDS or value["schema"] != SCHEMA
            or type(value["requested_representation"]) is not str
            or value["requested_representation"] not in {"original", "compact"}
            or type(value["selected_representation"]) is not str
            or value["selected_representation"] not in {"original", "compact"}
            or value["count_scope"] != COUNT_SCOPE
            or type(value["max_tokenizer_input_bytes"]) is not int
            or value["max_tokenizer_input_bytes"] != MAX_PROMPT_BYTES
            or type(value["max_reply_tokens"]) is not int or value["max_reply_tokens"] != MAX_REPLY_TOKENS
            or value["total_session_token_savings"] is not None
            or any(value[key] is not False for key in ("provider_usage_measured", "provider_framing_measured",
                "provider_tokenizer_equivalence_verified", "proof_authority", "completion_authority"))):
        raise ValueError("invalid token selection receipt")
    candidates = value["candidates"]
    if type(candidates) is not dict or set(candidates) not in ({"original"}, {"original", "compact"}):
        raise ValueError("invalid token selection candidates")
    for row in candidates.values():
        if (type(row) is not dict or set(row) != {"reply_sha256", "reply_bytes", "tokens"}
                or not _sha(row["reply_sha256"]) or not _integer(row["reply_bytes"], 2, MAX_REPLY_BYTES)
                or row["tokens"] is not None and not _integer(row["tokens"], 1, MAX_REPLY_TOKENS)):
            raise ValueError("invalid token selection candidate")
    selected = candidates.get(value["selected_representation"])
    if (selected is None or value["selected_reply_sha256"] != selected["reply_sha256"]
            or not _integer(value["selected_reply_bytes"], 2, MAX_REPLY_BYTES)
            or value["selected_reply_bytes"] != selected["reply_bytes"]
            or type(value["selected_tokens"]) is not type(selected["tokens"])
            or value["selected_tokens"] != selected["tokens"]):
        raise ValueError("selected token receipt differs from candidate")
    status, reason = value["measurement_status"], value["selection_reason"]
    if type(status) is not str or type(reason) is not str:
        raise ValueError("invalid token measurement classification")
    if status == "measured":
        if (value["requested_representation"] != "compact" or "compact" not in candidates
                or any(row["tokens"] is None or row["reply_bytes"] > MAX_PROMPT_BYTES for row in candidates.values())):
            raise ValueError("incomplete bounded token comparison")
        expected = "compact" if candidates["compact"]["tokens"] < candidates["original"]["tokens"] else "original"
        if (value["selected_representation"] != expected or reason != (
                "compact_strictly_fewer_tokens" if expected == "compact" else "original_not_larger")):
            raise ValueError("token selection contradicts complete comparison")
        binding, provenance = value["tokenizer_binding"], value["tokenizer_provenance"]
        if (type(binding) is not dict or set(binding) != _BINDING_FIELDS
                or not all(_bounded_text(item) for item in binding.values())
                or binding["special_token_policy"] not in {"tokenizers.encode:add_special_tokens=true",
                    "tokenizers.encode:add_special_tokens=false"}
                or type(provenance) is not dict or set(provenance) != _PROVENANCE_FIELDS
                or provenance["schema"] != "supervisor-local-prompt-tokenizer-provenance@1"
                or provenance["backend"] != "tokenizers-json@1"
                or provenance["compatibility_basis"] != "owner_declaration"
                or provenance["completion_authority"] is not False
                or not _sha(provenance["asset_sha256"])
                or not _bounded_text(provenance["tokenizers_version"])
                or binding["tokenizer_revision"] != "tokenizers/" + provenance["tokenizers_version"]
                    + "/sha256:" + provenance["asset_sha256"]):
            raise ValueError("invalid local token measurement provenance")
        descriptor = {"artifact": provenance["descriptor_path"], "expected_sha256": provenance["descriptor_sha256"],
            "provider": binding["provider"], "model": binding["model"]}
        validate_tokenizer_binding(descriptor)
        validate_tokenizer_binding({**descriptor, "artifact": provenance["asset_path"],
            "expected_sha256": provenance["asset_sha256"]})
    else:
        if (value["selected_representation"] != "original" or value["tokenizer_binding"] is not None
                or value["tokenizer_provenance"] is not None
                or any(row["tokens"] is not None for row in candidates.values())):
            raise ValueError("unmeasured selection contains a token claim")
        expected = {
            ("not_requested", "original_requested"): ("original", {"original"}),
            ("not_requested", "compact_unavailable"): ("compact", {"original"}),
            ("unavailable", "tokenizer_unavailable"): ("compact", {"original", "compact"}),
            ("unavailable", "tokenizer_input_bound"): ("compact", {"original", "compact"}),
            ("failed", "invalid_tokenizer"): ("compact", {"original", "compact"}),
            ("failed", "tokenizer_failed"): ("compact", {"original", "compact"}),
        }.get((status, reason))
        if expected != (value["requested_representation"], set(candidates)):
            raise ValueError("invalid unmeasured fallback classification")
        if reason == "tokenizer_input_bound" and not any(row["reply_bytes"] > MAX_PROMPT_BYTES for row in candidates.values()):
            raise ValueError("tokenizer bound fallback lacks oversized candidate")


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate selection audit field")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("nonfinite selection audit field")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _private_parent(path):
    """Walk with no-follow descriptors, retaining the private parent handle."""
    text = os.fspath(path)
    path = Path(text)
    if (not path.is_absolute() or text != str(path) or len(text.encode("utf-8")) > 4096
            or any(part in {".", ".."} for part in text.split("/"))
            or len(path.parts) > 64 or not path.name):
        raise ValueError("canonical private selection audit path required")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("selection audit parent must be private")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _file_info(fd):
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_AUDIT_BYTES):
        raise ValueError("invalid private selection audit file")
    return info


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class SelectionAudit:
    """Bounded private journal of successful read selections, closed by a footer.

    The server records a receipt only after a successful read/currentness check.
    The journal hash detects population changes; it is not a signature or proof
    that a provider consumed a reply. Consumers must join read counts to the
    separate server audit and bind the tokenizer to their invocation config.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.parent_fd = _private_parent(path)
        self.fd = None
        try:
            self.fd = os.open(self.path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                              | os.O_CLOEXEC, 0o600, dir_fd=self.parent_fd)
            _file_info(self.fd)
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
            os.close(self.parent_fd)
            raise
        self.sequence = self.bytes_written = 0
        self.digest = hashlib.sha256()
        self.failed = False

    def _write(self, raw):
        info = _file_info(self.fd)
        current = os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if (info.st_size != self.bytes_written or _identity(info) != _identity(current)
                or info.st_size + len(raw) > MAX_AUDIT_BYTES):
            raise ValueError("selection audit changed or exceeded its bound")
        view = memoryview(raw)
        while view:
            size = os.write(self.fd, view)
            if size <= 0:
                raise OSError("selection audit write failed")
            view = view[size:]
        self.bytes_written += len(raw)

    def record(self, selection):
        if self.fd is None or self.failed:
            raise ValueError("selection audit is closed or failed")
        try:
            if self.sequence >= MAX_AUDIT_RECORDS:
                raise ValueError("selection audit record bound exceeded")
            _validate_selection(selection)
            raw = _json({"schema": AUDIT_SCHEMA, "sequence": self.sequence + 1, "selection": selection}) + b"\n"
            if len(raw) > MAX_AUDIT_RECORD_BYTES or self.bytes_written + len(raw) + 512 > MAX_AUDIT_BYTES:
                raise ValueError("selection audit byte bound exceeded")
            self._write(raw)
            self.sequence += 1
            self.digest.update(raw)
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.fd is None:
            return
        try:
            if not self.failed:
                self._write(_json({"schema": AUDIT_SCHEMA, "event": "closed", "read_count": self.sequence,
                    "records_sha256": self.digest.hexdigest()}) + b"\n")
        finally:
            os.close(self.fd)
            os.close(self.parent_fd)
            self.fd = self.parent_fd = None


def _empty_summary(status):
    return {"schema": SUMMARY_SCHEMA, "status": status, "audit_closed": status == "valid",
        **{key: 0 if status == "valid" else None for key in _SUMMARY_COUNTERS},
        "provider_usage_measured": False}


def validate_selection_summary(value):
    """Accept a complete consistent summary, without inferring task success."""
    if (type(value) is not dict or set(value) != _SUMMARY_FIELDS or value["schema"] != SUMMARY_SCHEMA
            or value["status"] != "valid" or value["audit_closed"] is not True
            or value["provider_usage_measured"] is not False
            or any(not _integer(value[key], 0, MAX_AUDIT_RECORDS) for key in _SUMMARY_COUNTERS[:-1])
            or not _integer(value["selected_reply_tokens"], 0, MAX_AUDIT_RECORDS * MAX_REPLY_TOKENS)
            or value["compact_selected"] + value["original_selected"] != value["read_count"]
            or value["measured_comparisons"] + value["unmeasured_fallbacks"] + value["explicit_original"] != value["read_count"]
            or value["compact_selected"] > value["measured_comparisons"]
            or not value["measured_comparisons"] <= value["selected_reply_tokens"] <= value["measured_comparisons"] * MAX_REPLY_TOKENS):
        raise ValueError("invalid complete repository token selection summary")
    return dict(value)


def summarize_selection_audit(path, *, tokenizer_binding=None):
    """Validate the complete bounded population before returning any counters.

    No source or reply bodies are retained. Missing/invalid journals return no
    partial totals, and successful replay does not establish provider usage.
    """
    parent_fd = None
    try:
        if tokenizer_binding is not None:
            tokenizer_binding = validate_tokenizer_binding(tokenizer_binding)
        path = Path(path)
        parent_fd = _private_parent(path)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as stream:
            before = _file_info(stream.fileno())
            raw = stream.read(MAX_AUDIT_BYTES + 1)
            after = _file_info(stream.fileno())
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (_identity(before) != _identity(after) or _identity(after) != _identity(current)
                    or len(raw) != before.st_size):
                raise ValueError("selection audit changed during read")
        if not raw or len(raw) > MAX_AUDIT_BYTES or not raw.endswith(b"\n"):
            raise ValueError("incomplete selection audit")
        lines = raw.splitlines(keepends=True)
        if len(lines) > MAX_AUDIT_RECORDS + 1 or any(len(line) > MAX_AUDIT_RECORD_BYTES for line in lines):
            raise ValueError("selection audit population bound exceeded")
        footer = _loads(lines[-1])
        if (type(footer) is not dict or set(footer) != {"schema", "event", "read_count", "records_sha256"}
                or footer["schema"] != AUDIT_SCHEMA or footer["event"] != "closed"
                or not _integer(footer["read_count"], 0, MAX_AUDIT_RECORDS)
                or footer["read_count"] != len(lines) - 1 or not _sha(footer["records_sha256"])
                or footer["records_sha256"] != hashlib.sha256(b"".join(lines[:-1])).hexdigest()):
            raise ValueError("invalid selection audit closure")
        result = _empty_summary("valid")
        owner = None
        for sequence, line in enumerate(lines[:-1], 1):
            row = _loads(line)
            if (type(row) is not dict or set(row) != {"schema", "sequence", "selection"}
                    or row["schema"] != AUDIT_SCHEMA or type(row["sequence"]) is not int or row["sequence"] != sequence):
                raise ValueError("invalid selection audit sequence")
            selection = row["selection"]
            _validate_selection(selection)
            measured = selection["measurement_status"] == "measured"
            explicit = selection["requested_representation"] == "original"
            if measured:
                binding, provenance = selection["tokenizer_binding"], selection["tokenizer_provenance"]
                if owner is not None and owner != (binding, provenance):
                    raise ValueError("selection audit mixes tokenizer owners")
                owner = binding, provenance
                if tokenizer_binding is not None and tokenizer_binding != {
                        "artifact": provenance["descriptor_path"], "expected_sha256": provenance["descriptor_sha256"],
                        "provider": binding["provider"], "model": binding["model"]}:
                    raise ValueError("selection audit tokenizer differs from configuration")
            result["read_count"] += 1
            result["compact_selected"] += selection["selected_representation"] == "compact"
            result["original_selected"] += selection["selected_representation"] == "original"
            result["measured_comparisons"] += measured
            result["explicit_original"] += explicit
            result["unmeasured_fallbacks"] += not measured and not explicit
            result["selected_reply_tokens"] += selection["selected_tokens"] if measured else 0
        return validate_selection_summary(result)
    except FileNotFoundError:
        return _empty_summary("missing")
    except (ValueError, OSError, TypeError, UnicodeError, RecursionError):
        return _empty_summary("invalid")
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
