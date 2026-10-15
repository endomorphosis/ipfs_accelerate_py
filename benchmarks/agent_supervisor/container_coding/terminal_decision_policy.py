"""Explicit decision launch controls and current selected-source custody.

Local input budgets describe the chosen counter; they are not native charges
or a guarantee about the provider's cumulative session budget. This module
performs no filesystem work unless an archive/source check is requested.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import tarfile


def _artifact(value):
    if not isinstance(value, (str, Path)):
        raise ValueError("literal absolute decision artifact required")
    text = str(value)
    path = Path(text)
    if (not 1 <= len(text.encode("utf-8")) <= 4096 or not path.is_absolute()
            or str(path) != text or text.startswith("//") or ":" in text or "\\" in text
            or any(part in {".", ".."} for part in text.split("/"))
            or any(ord(char) < 32 for char in text)):
        raise ValueError("canonical bounded absolute decision artifact required")
    return text


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("independent lowercase decision SHA-256 required")
    return value


def decision_selection(*, semantic_decision_policy_artifact=None,
                       semantic_decision_policy_sha256=None,
                       semantic_decision_nomination_artifact=None,
                       semantic_decision_nomination_sha256=None,
                       semantic_decision_nomination_source=None,
                       disable_semantic_typed_nomination=False,
                       semantic_retrieval_query_reference=False,
                       semantic_decision_max_calls=3,
                       semantic_decision_max_total_input_tokens=None):
    """Normalize one independently selected policy/nomination; omit defaults."""
    if (type(disable_semantic_typed_nomination) is not bool
            or type(semantic_retrieval_query_reference) is not bool):
        raise ValueError("decision nomination switches must be Boolean")
    if type(semantic_decision_max_calls) is not int or not 1 <= semantic_decision_max_calls <= 4:
        raise ValueError("decision call limit must be an integer from 1 to 4")
    if (semantic_decision_max_total_input_tokens is not None
            and (type(semantic_decision_max_total_input_tokens) is not int
                 or semantic_decision_max_total_input_tokens < 1)):
        raise ValueError("positive complete decision input budget required")
    policy = semantic_decision_policy_artifact is not None or semantic_decision_policy_sha256 is not None
    nomination = (semantic_decision_nomination_artifact is not None
                  or semantic_decision_nomination_source is not None
                  or semantic_decision_nomination_sha256 is not None)
    if policy and nomination:
        raise ValueError("choose decision policy or nomination, not both")
    if semantic_decision_nomination_artifact is not None and semantic_decision_nomination_source is not None:
        raise ValueError("choose child nomination or host nomination source")
    if not policy and not nomination:
        if (disable_semantic_typed_nomination or semantic_retrieval_query_reference
                or semantic_decision_max_calls != 3 or semantic_decision_max_total_input_tokens is not None):
            raise ValueError("decision controls require an explicit policy or nomination")
        return {}
    if disable_semantic_typed_nomination and not nomination:
        raise ValueError("typed nomination switch requires a nomination")
    if semantic_retrieval_query_reference and not nomination:
        raise ValueError("retrieval reference requires an independently pinned owner nomination")
    if policy:
        result = {"semantic_decision_policy_artifact": _artifact(semantic_decision_policy_artifact),
                  "semantic_decision_policy_sha256": _sha(semantic_decision_policy_sha256)}
    else:
        name = ("semantic_decision_nomination_source" if semantic_decision_nomination_source is not None
                else "semantic_decision_nomination_artifact")
        value = semantic_decision_nomination_source if name.endswith("source") else semantic_decision_nomination_artifact
        result = {name: _artifact(value),
                  "semantic_decision_nomination_sha256": _sha(semantic_decision_nomination_sha256)}
    result.update(semantic_decision_max_calls=semantic_decision_max_calls,
                  semantic_decision_max_total_input_tokens=semantic_decision_max_total_input_tokens)
    if disable_semantic_typed_nomination:
        result["disable_semantic_typed_nomination"] = True
    if semantic_retrieval_query_reference:
        result["semantic_retrieval_query_reference"] = True
    return result


_MODE_DEFAULTS = {
    "planning_mode": "model", "semantic_transport_schema": "supervisor-semantic-router-input@1",
    "coding_reply_mode": "legacy", "tool_output_policy": "legacy",
    "source_view_advisory_policy": "legacy", "repository_source_policy": "legacy",
    "source_input_policy": "legacy", "query_reference_policy": "legacy",
    "semantic_metadata_view": "legacy", "semantic_working_set": "legacy",
    "source_view_profile": "disabled", "source_edit_mode": "disabled", "source_view_policy": None,
    "semantic_context_profile": "legacy-bounded@1", "capsule_selection": "lexical-query-symbol-name@1",
    "capsule_budget": "legacy", "numerical_formalization_profile": None,
    "numerical_selection_config": None, "source384_config": None,
    "metadata_census_only": False, "grok_coding_max_turns": 128,
}


def validate_decision_route(selection, *, arm, provider, **options):
    """Refuse unsupported selected routes without touching ordinary profiles."""
    if type(selection) is not dict:
        raise ValueError("decision selection must be an object")
    if not selection:
        return
    if decision_selection(**selection) != selection:
        raise ValueError("canonical decision selection required")
    if provider != "codex_cli":
        raise ValueError("semantic decision policy requires the qualified Codex provider")
    if arm != "full":
        raise ValueError("decision dialogue currently requires full indexed Codex coding")
    ignored = {"resource_profile", "task_profile", "intent_requirement_contract", "semantic_repository", "doctor"}
    if set(options) - set(_MODE_DEFAULTS) - ignored - {"timeout", "purpose"}:
        raise ValueError("unrecognized selected decision route option")
    for name, default in _MODE_DEFAULTS.items():
        if name in options and (type(options[name]) is not type(default) or options[name] != default):
            raise ValueError("decision dialogue does not support selected " + name)
    if options.get("purpose", "coding") != "coding":
        raise ValueError("decision dialogue requires coding purpose")
    if "timeout" in options and (type(options["timeout"]) is not int or not 1 <= options["timeout"] <= 600):
        raise ValueError("decision worker timeout must be an integer from 1 to 600")


def semantic_decision_arguments(*, owner_binding=False, **options):
    """Encode child-local controls; task/bundle custody belongs to the driver."""
    if type(owner_binding) is not bool:
        raise ValueError("explicit owner binding switch required")
    selected = decision_selection(**options)
    if not selected:
        if owner_binding:
            raise ValueError("owner binding requires a nomination")
        return []
    if "semantic_decision_nomination_source" in selected:
        raise ValueError("host nomination must be delivered before child launch")
    nomination = "semantic_decision_nomination_artifact" in selected
    if owner_binding and not nomination:
        raise ValueError("owner binding requires a nomination")
    kind = "nomination" if nomination else "policy"
    result = ["--semantic-decision-" + kind + "-artifact", selected["semantic_decision_" + kind + "_artifact"],
              "--semantic-decision-" + kind + "-sha256", selected["semantic_decision_" + kind + "_sha256"]]
    if owner_binding:
        result += ["--semantic-decision-typed-nomination",
                   "disabled" if selected.get("disable_semantic_typed_nomination") else "enabled"]
    elif selected.get("disable_semantic_typed_nomination"):
        result += ["--disable-semantic-typed-nomination"]
    if selected.get("semantic_retrieval_query_reference") and not owner_binding:
        result += ["--semantic-retrieval-query-reference"]
    result += ["--semantic-decision-max-calls", str(selected["semantic_decision_max_calls"])]
    if selected["semantic_decision_max_total_input_tokens"] is not None:
        result += ["--semantic-decision-max-total-input-tokens", str(selected["semantic_decision_max_total_input_tokens"])]
    return result


_BENCHMARK_SOURCES = (
    "terminal_decision_policy.py", "full_supervisor_benchmark.py", "full_supervisor_harbor_agent.py",
    "terminal_container_supervisor.py", "terminal_doctor_dispatch.py", "container_worker_deployment.py",
    "terminal_deployment.py", "benchmark_controls.py", "semantic_decision_nomination_delivery.py", "terminal_context_audit.py",
    "terminal_context_evidence.py", "semantic_decision_input_audit.py", "automatic_tokenizer_audit.py",
    "terminal_decision_usage_projection.py", "native_usage_accounting.py", "terminal_worker_capability.py",
)
_RUNTIME_SOURCES = (
    "semantic_decision_worker.py", "semantic_decision_owner.py", "semantic_decision_provider.py",
    "semantic_decision_dialogue.py", "semantic_decision_packet.py", "semantic_decision_usage.py",
    "semantic_projection_dispatch.py", "semantic_router_translation.py", "semantic_context_projection.py",
    "router_implementation_runner.py", "automatic_prompt_tokenizer.py", "local_prompt_tokenizer.py",
    "container_worker_boundary.py", "doctor_residual_context.py", "router_public_instruction.py",
    "cumulative_usage_reconciliation.py", "multi_supervisor_runner.py",
)
_TYPED_SOURCES = (
    "semantic_typed_owner_policy.py", "semantic_typed_source_context.py", "semantic_typed_dependency_proposal.py",
    "semantic_typed_nomination_lineage.py", "semantic_selection_relation_audit.py", "semantic_selection_dependency_audit.py",
)


def decision_runtime_sources(selection):
    """Declare the selected consumer/producer chain, not all imported modules."""
    if not selection:
        return {}
    if decision_selection(**selection) != selection:
        raise ValueError("canonical decision selection required")
    root = Path(__file__).resolve().parents[3]
    prefix = "ipfs_accelerate_py/agent_supervisor/runtime/"
    names = ["benchmarks/agent_supervisor/container_coding/" + n for n in _BENCHMARK_SOURCES]
    names += [prefix + n for n in _RUNTIME_SOURCES]
    nomination = ("semantic_decision_nomination_artifact" in selection
                  or "semantic_decision_nomination_source" in selection)
    if nomination and not selection.get("disable_semantic_typed_nomination"):
        names += [prefix + n for n in _TYPED_SOURCES]
        names += ["ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py"]
    if selection.get("semantic_retrieval_query_reference"):
        names += [prefix + n for n in ("semantic_retrieval_query_reference.py", "code_retrieval_context.py", "task_context_bundle.py")]
    return {"source/" + n: root / n for n in names}


def _source_row(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 2 * 1024 * 1024:
            raise ValueError("bounded regular selected decision source required")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode, value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.lstat()):
        raise ValueError("selected decision source changed while reading")
    return {"sha256": digest, "bytes": before.st_size,
            "mode": 0o755 if os.access(path, os.X_OK) else 0o644}


def require_decision_source_pins(pins, selection):
    sources = decision_runtime_sources(selection)
    if type(pins) is not dict or set(pins) != set(sources):
        raise ValueError("decision source pins differ; prepare a new trial")
    if any(pins[name] != _source_row(path)["sha256"] for name, path in sources.items()):
        raise ValueError("decision source pins differ; prepare a new trial")


def require_decision_archive(archive, manifest, selection):
    """Join exact current selected bytes to unique manifest and real tar rows."""
    sources = decision_runtime_sources(selection)
    if not sources:
        return {}
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("selected decision archive inventory required")
    held = {name: _source_row(path) for name, path in sources.items()}
    for name, expected in held.items():
        matches = [row for row in rows if type(row) is dict and row.get("path") == name]
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in expected.items()):
            raise ValueError("selected decision archive source inventory differs")
        if any(type(matches[0].get(key)) is not int for key in ("bytes", "mode")):
            raise ValueError("selected decision archive integer metadata required")
    archive = Path(archive)
    if archive.is_dir():
        archive = archive / "runtime.tar.gz"
    descriptor = os.open(archive, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    seen = set()
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 4 * 1024**3:
            raise ValueError("bounded regular selected decision archive required")
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest.get("archive_sha256"):
            raise ValueError("selected decision archive digest differs")
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode="r|gz") as bundle:
            total = 0
            for ordinal, member in enumerate(bundle):
                total += member.size
                if ordinal >= 250_000 or total > 8 * 1024**3 or member.size < 0:
                    raise ValueError("selected decision archive exceeds inventory bounds")
                if member.name not in held:
                    continue
                expected = held[member.name]
                if (member.name in seen or not member.isfile() or member.issym() or member.islnk()
                        or member.size != expected["bytes"] or member.mode != expected["mode"]):
                    raise ValueError("selected decision archive member differs")
                content = bundle.extractfile(member)
                if content is None:
                    raise ValueError("selected decision archive member unavailable")
                with content:
                    raw = content.read(2 * 1024 * 1024 + 1)
                if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                    raise ValueError("selected decision archive payload differs")
                seen.add(member.name)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode, value.st_mtime_ns, value.st_ctime_ns)
    if (seen != set(held) or identity(before) != identity(after)
            or identity(after) != identity(archive.lstat())
            or any(_source_row(sources[name]) != row for name, row in held.items())):
        raise ValueError("selected decision archive or source changed during verification")
    return {name: row["sha256"] for name, row in held.items()}
