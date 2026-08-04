"""Tests for evidence-bound program contract profiles (LPR-022)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_contract_profile import (
    MAX_TEXT_BYTES,
    PROFILE_AUTHORIZES_REPAIR,
    PROFILE_IS_COMPLETION_EVIDENCE,
    PROFILE_IS_CORRECTNESS_EVIDENCE,
    PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA,
    PROGRAM_CONTRACT_PROFILE_SCHEMA,
    PROGRAM_CONTRACT_PROFILE_VERSION,
    CanonicalVector,
    ContractSourceKind,
    ContractVocabulary,
    DataMode,
    ExecutionMode,
    ExpectationIssue,
    ExpectationState,
    FacadeCompatibility,
    FacadeExample,
    InvariantContract,
    IssueKind,
    OperationContract,
    OperationSupport,
    ProgramContractProfile,
    ProgramContractProfileCompiler,
    ProgramContractProfileError,
    PublicSurfaceContract,
    SourceContract,
    SurfaceOperationContract,
    assert_contract_profile_complete,
    discover_profile_schemas,
    publish_contract_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_MODULE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "analysis"
    / "program_contract_profile.py"
)
LOCK_PATH = REPO_ROOT / "config" / "agent_supervisor_vfs_generalization_sources.lock.json"

_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife|"
    r"ipfs_kit|board[_-]?id|board[_-]?namespace)\b"
)


def _canonical_digest(record: dict[str, object]) -> str:
    unsigned = dict(record)
    unsigned.pop("content_id")
    payload = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Unrelated RPC / key-value profile — exercises every generic record type
# ---------------------------------------------------------------------------


def _rpc_vocabulary() -> ContractVocabulary:
    return ContractVocabulary(
        operations=("kv.get", "kv.put", "kv.delete", "rpc.call"),
        invariant_kinds=(
            "key_identity",
            "value_bytes",
            "rpc_timeout",
            "delete_idempotent",
        ),
        error_codes=(
            "invalid_argument",
            "not_found",
            "already_exists",
            "deadline_exceeded",
            "permission_denied",
        ),
        surfaces=("python_client", "http_gateway", "cli"),
    )


def _rpc_source(
    source_id: str,
    *,
    available: bool = True,
    reviewed: bool = True,
    authority: bool = True,
    kind: ContractSourceKind = ContractSourceKind.REVIEWED_INTERFACE,
) -> SourceContract:
    return SourceContract(
        source_id=source_id,
        kind=kind,
        locator=f"task://rpc-kv/{source_id}",
        revision="rev:rpc-kv-1",
        summary=f"Independent source evidence for {source_id}.",
        reviewed=reviewed,
        available=available,
        expectation_authority=authority,
    )


def build_rpc_key_value_profile() -> ProgramContractProfile:
    """Domain-unrelated profile that fills every generic record kind."""

    vocab = _rpc_vocabulary()
    acceptance = _rpc_source("source:rpc-kv-acceptance")
    surface_sources = tuple(
        _rpc_source(f"source:rpc-kv-surface:{surface}") for surface in vocab.surfaces
    )
    missing = SourceContract(
        source_id="source:rpc-kv-missing-batch",
        kind=ContractSourceKind.REVIEWED_INTERFACE,
        locator="missing://rpc-kv/batch-contract",
        revision="unavailable",
        summary="Batch multi-key atomicity contract was not supplied.",
        reviewed=False,
        available=False,
        expectation_authority=False,
    )
    conflict_a = _rpc_source("source:rpc-kv-conflict-a")
    conflict_b = _rpc_source("source:rpc-kv-conflict-b")
    sources = (acceptance, *surface_sources, missing, conflict_a, conflict_b)

    def inv(
        kind: str,
        statement: str,
        applies_to: tuple[str, ...],
        errors: tuple[str, ...] = (),
    ) -> InvariantContract:
        return InvariantContract(
            invariant_id=f"invariant:{kind}",
            kind=kind,
            statement=statement,
            applies_to=applies_to,
            source_contract_ids=("source:rpc-kv-acceptance",),
            error_codes=errors,
            postconditions=(f"holds for {kind}",),
        )

    invariants = (
        inv(
            "key_identity",
            "Keys are exact byte strings; equal keys address one entry.",
            ("kv.get", "kv.put", "kv.delete"),
            ("invalid_argument",),
        ),
        inv(
            "value_bytes",
            "Values are opaque bytes with exact length accounting.",
            ("kv.get", "kv.put"),
            ("invalid_argument",),
        ),
        inv(
            "rpc_timeout",
            "rpc.call respects its deadline and never hangs past it.",
            ("rpc.call",),
            ("deadline_exceeded",),
        ),
        inv(
            "delete_idempotent",
            "kv.delete on a missing key is a no-op success.",
            ("kv.delete",),
            ("not_found",),
        ),
    )

    def op(
        name: str,
        summary: str,
        *,
        inputs: tuple[DataMode, ...],
        outputs: tuple[DataMode, ...],
        mutates: bool,
        idempotent: bool | None,
        inv_ids: tuple[str, ...],
        errors: tuple[str, ...],
    ) -> OperationContract:
        return OperationContract(
            operation=name,
            summary=summary,
            input_modes=inputs,
            output_modes=outputs,
            execution_modes=(ExecutionMode.SYNC, ExecutionMode.ASYNC),
            invariant_ids=inv_ids,
            error_codes=errors,
            source_contract_ids=("source:rpc-kv-acceptance",),
            mutates=mutates,
            idempotent=idempotent,
        )

    operations = (
        op(
            "kv.get",
            "Fetch value bytes for an exact key.",
            inputs=(DataMode.BYTES, DataMode.METADATA),
            outputs=(DataMode.BYTES, DataMode.METADATA),
            mutates=False,
            idempotent=True,
            inv_ids=("invariant:key_identity", "invariant:value_bytes"),
            errors=("invalid_argument", "not_found", "permission_denied"),
        ),
        op(
            "kv.put",
            "Store exact value bytes under a key.",
            inputs=(DataMode.BYTES, DataMode.METADATA),
            outputs=(DataMode.METADATA,),
            mutates=True,
            idempotent=False,
            inv_ids=("invariant:key_identity", "invariant:value_bytes"),
            errors=("invalid_argument", "already_exists", "permission_denied"),
        ),
        op(
            "kv.delete",
            "Remove a key; missing keys succeed idempotently.",
            inputs=(DataMode.BYTES, DataMode.METADATA),
            outputs=(DataMode.NONE,),
            mutates=True,
            idempotent=True,
            inv_ids=("invariant:key_identity", "invariant:delete_idempotent"),
            errors=("invalid_argument", "not_found", "permission_denied"),
        ),
        op(
            "rpc.call",
            "Invoke a remote procedure with a hard deadline.",
            inputs=(DataMode.BYTES, DataMode.METADATA),
            outputs=(DataMode.BYTES, DataMode.METADATA),
            mutates=False,
            idempotent=None,
            inv_ids=("invariant:rpc_timeout",),
            errors=("invalid_argument", "deadline_exceeded", "permission_denied"),
        ),
    )

    surfaces: list[PublicSurfaceContract] = []
    for surface in vocab.surfaces:
        source_id = f"source:rpc-kv-surface:{surface}"
        modes = (
            (ExecutionMode.SYNC, ExecutionMode.ASYNC)
            if surface == "python_client"
            else (ExecutionMode.SYNC,)
            if surface == "cli"
            else (ExecutionMode.ASYNC,)
        )
        support_map = {
            "kv.get": OperationSupport.SUPPORTED,
            "kv.put": OperationSupport.SUPPORTED,
            "kv.delete": OperationSupport.SUPPORTED,
            "rpc.call": (
                OperationSupport.SUPPORTED
                if surface != "cli"
                else OperationSupport.UNSUPPORTED
            ),
        }
        bindings = tuple(
            SurfaceOperationContract(
                operation=operation,
                support=support_map[operation],
                source_contract_ids=(source_id,),
                entrypoint=f"{surface}:{operation}",
                note=(
                    ""
                    if support_map[operation] is OperationSupport.SUPPORTED
                    else "CLI does not expose generic RPC call."
                ),
            )
            for operation in vocab.operations
        )
        surfaces.append(
            PublicSurfaceContract(
                surface=surface,
                contract_name=f"{surface} contract",
                execution_modes=modes,
                operations=bindings,
                source_contract_ids=(source_id,),
            )
        )

    issues = (
        ExpectationIssue(
            issue_id="issue:rpc-kv-missing-batch",
            kind=IssueKind.MISSING,
            subject="Batch multi-key atomicity is absent until a reviewed contract exists.",
            source_contract_ids=("source:rpc-kv-missing-batch",),
            positions=(),
            state=ExpectationState.UNRESOLVED,
        ),
        ExpectationIssue(
            issue_id="issue:rpc-kv-conflict-put-create",
            kind=IssueKind.CONFLICT,
            subject="Two reviewed sources disagree on put create-only semantics.",
            source_contract_ids=(
                "source:rpc-kv-conflict-a",
                "source:rpc-kv-conflict-b",
            ),
            positions=(
                "put fails when key exists",
                "put overwrites existing key",
            ),
            state=ExpectationState.CONFLICTING,
        ),
    )

    vectors = (
        CanonicalVector(
            vector_id="vector:kv:get-exact",
            operation="kv.get",
            description="Exact key returns stored bytes.",
            request={"key": "alpha"},
            expected={"value_hex": "6869", "size": 2},
            invariant_ids=("invariant:key_identity", "invariant:value_bytes"),
            source_contract_ids=("source:rpc-kv-acceptance",),
            exact_semantics="get(key) returns the exact bytes last put under that key.",
        ),
        CanonicalVector(
            vector_id="vector:kv:delete-missing",
            operation="kv.delete",
            description="Deleting a missing key succeeds without effects.",
            request={"key": "missing"},
            expected={"ok": True, "effects": "none"},
            invariant_ids=("invariant:delete_idempotent",),
            source_contract_ids=("source:rpc-kv-acceptance",),
            exact_semantics="delete of absent key is idempotent success.",
        ),
        CanonicalVector(
            vector_id="vector:rpc:deadline",
            operation="rpc.call",
            description="Past-deadline call fails closed.",
            request={"method": "echo", "deadline_ms": 0},
            expected={"error": "deadline_exceeded"},
            invariant_ids=("invariant:rpc_timeout",),
            source_contract_ids=("source:rpc-kv-acceptance",),
            exact_semantics="deadline_ms=0 always yields deadline_exceeded.",
        ),
    )

    facade_examples = (
        FacadeExample(
            example_id="facade:python_client:compatible-get",
            surface="python_client",
            compatibility=FacadeCompatibility.COMPATIBLE,
            description="Python client returns bytes for get.",
            operation="kv.get",
            example={"call": "client.get(b'k')", "returns": "bytes"},
            rationale="Matches value_bytes invariant.",
            source_contract_ids=("source:rpc-kv-acceptance",),
        ),
        FacadeExample(
            example_id="facade:http_gateway:incompatible-text",
            surface="http_gateway",
            compatibility=FacadeCompatibility.INCOMPATIBLE,
            description="HTTP gateway decodes bytes as text without opt-in.",
            operation="kv.get",
            example={"accept": "text/plain"},
            rationale="Silent text decoding violates value_bytes.",
            source_contract_ids=("source:rpc-kv-acceptance",),
        ),
        FacadeExample(
            example_id="facade:cli:unresolved-rpc",
            surface="cli",
            compatibility=FacadeCompatibility.UNRESOLVED,
            description="CLI rpc exposure is not yet reviewed.",
            operation="rpc.call",
            example={"argv": ["rpc", "call", "echo"]},
            rationale="No reviewed CLI RPC contract.",
            source_contract_ids=("source:rpc-kv-missing-batch",),
        ),
    )

    compiler = ProgramContractProfileCompiler(
        schema=PROGRAM_CONTRACT_PROFILE_SCHEMA,
        operation_matrix_schema=PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA,
        contract_version=PROGRAM_CONTRACT_PROFILE_VERSION,
        goal_id="RPC-KV-001",
        profile_id="profile:rpc-key-value@1",
    )
    return compiler.compile(
        vocab,
        sources=sources,
        invariants=invariants,
        operations=operations,
        surfaces=tuple(surfaces),
        issues=issues,
        vectors=vectors,
        facade_examples=facade_examples,
    )


# ---------------------------------------------------------------------------
# VFS-equivalent projection (domain data lives only in the test profile)
# ---------------------------------------------------------------------------


# Locked identities from source blob 9acc4ceb… (schemas and vocabularies only).
_LOCKED_VFS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/vfs-contract-pack@1"
_LOCKED_VFS_MATRIX_SCHEMA = "vfs/canonical-operation-matrix@1"
_LOCKED_VFS_VERSION = "vfs-contract-pack/v1"
_LOCKED_VFS_GOAL = "VFS-026"
_LOCKED_VFS_OPERATIONS = (
    "path.resolve",
    "mount",
    "read",
    "write",
    "open",
    "close",
    "seek",
    "stat",
    "list",
    "mkdir",
    "remove",
    "rename",
    "copy",
)
_LOCKED_VFS_INVARIANTS = (
    "versioned_path",
    "unicode",
    "root",
    "traversal",
    "mount",
    "read_write",
    "handle_lifecycle",
    "seek",
    "stat_list",
    "directory_mutation",
    "namespace_mutation",
    "bytes_text",
    "sync_async",
    "error",
    "cid_size",
    "atomicity",
    "journal_replay",
    "versioning",
    "cache_pin_coherence",
    "backend_negotiation",
    "authorization",
    "resource",
    "degradation",
)
_LOCKED_VFS_ERRORS = (
    "invalid_argument",
    "invalid_path",
    "traversal_denied",
    "not_found",
    "already_exists",
    "not_a_file",
    "not_a_directory",
    "directory_not_empty",
    "permission_denied",
    "authentication_required",
    "conflict",
    "stale_version",
    "unsupported",
    "capability_unavailable",
    "resource_exhausted",
    "integrity_failure",
    "io_failure",
    "cancelled",
    "deadline_exceeded",
)
_LOCKED_VFS_SURFACES = (
    "python",
    "cli",
    "mcp",
    "mcp++",
    "http",
    "libp2p",
)


def build_vfs_equivalent_profile() -> ProgramContractProfile:
    """Project the locked VFS operation/invariant/schema identities as profile data.

    Semantic identity is schema + vocabulary + content_id — never a module path.
    """

    vocab = ContractVocabulary(
        operations=_LOCKED_VFS_OPERATIONS,
        invariant_kinds=_LOCKED_VFS_INVARIANTS,
        error_codes=_LOCKED_VFS_ERRORS,
        surfaces=_LOCKED_VFS_SURFACES,
    )
    acceptance = SourceContract(
        source_id="source:vfs-026-acceptance",
        kind=ContractSourceKind.REVIEWED_INTERFACE,
        locator="task://VFS-026/acceptance",
        revision="baguqeerauopqqkevmksjwate5nvprfxnz3bgci3kmqfcx2ckbmv6al66xfeq",
        summary="Reviewed VFS-026 operation and invariant acceptance contract.",
        reviewed=True,
    )
    surface_sources = tuple(
        SourceContract(
            source_id=f"source:vfs-026-surface:{surface}",
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            locator=f"task://VFS-026/acceptance#surface-{surface}",
            revision=acceptance.revision,
            summary=f"Reviewed facade mapping for {surface}.",
            reviewed=True,
        )
        for surface in vocab.surfaces
    )
    missing = SourceContract(
        source_id="source:missing-backend-atomicity-contract",
        kind=ContractSourceKind.REVIEWED_INTERFACE,
        locator="missing://backend/atomicity-capability-contract",
        revision="unavailable",
        summary="Backend-specific atomicity capability contract was not provided.",
        reviewed=False,
        available=False,
        expectation_authority=False,
    )
    sources = (acceptance, *surface_sources, missing)

    path_ops = (
        "path.resolve",
        "mount",
        "read",
        "write",
        "open",
        "stat",
        "list",
        "mkdir",
        "remove",
        "rename",
        "copy",
    )
    mutations = ("mount", "write", "mkdir", "remove", "rename", "copy")
    handle_ops = ("open", "close", "seek")
    content_ops = ("read", "write", "open", "stat", "copy")
    namespace_ops = ("mkdir", "remove", "rename", "copy")

    def inv(
        kind: str,
        statement: str,
        applies_to: tuple[str, ...],
        errors: tuple[str, ...] = (),
    ) -> InvariantContract:
        return InvariantContract(
            invariant_id=f"invariant:{kind}",
            kind=kind,
            statement=statement,
            applies_to=applies_to,
            source_contract_ids=("source:vfs-026-acceptance",),
            error_codes=errors,
        )

    invariants = (
        inv("versioned_path", "Path plus optional version selector.", path_ops, ("invalid_path",)),
        inv("unicode", "Valid Unicode NFC for paths and text.", path_ops, ("invalid_path", "invalid_argument")),
        inv("root", "Canonical root is '/' and cannot be removed.", path_ops, ("invalid_path", "permission_denied")),
        inv("traversal", "'..' never escapes mount root.", path_ops, ("traversal_denied",)),
        inv("mount", "Longest component-boundary mount match.", path_ops, ("capability_unavailable", "unsupported")),
        inv("read_write", "Exact byte ranges for read/write.", ("read", "write"), ("io_failure",)),
        inv("handle_lifecycle", "Open/close/seek lifecycle is explicit.", handle_ops, ("invalid_argument",)),
        inv("seek", "Seek counts bytes, not characters.", ("seek",), ("invalid_argument",)),
        inv("stat_list", "Stat/list bind type and size coherently.", ("stat", "list"), ("not_found",)),
        inv("directory_mutation", "Directory mutations are explicit.", ("mkdir", "remove"), ("directory_not_empty", "not_a_directory")),
        inv("namespace_mutation", "Namespace mutations preserve identity.", namespace_ops, ("already_exists", "not_found")),
        inv("bytes_text", "Text adapters are opt-in over exact bytes.", content_ops, ("invalid_argument",)),
        inv("sync_async", "Sync and async share semantics.", _LOCKED_VFS_OPERATIONS, ()),
        inv("error", "Transport-neutral error codes.", _LOCKED_VFS_OPERATIONS, ("io_failure",)),
        inv("cid_size", "CID and size bind the same bytes.", ("stat", "read", "write", "copy"), ("integrity_failure",)),
        inv("atomicity", "Committed ops do not partially apply.", mutations, ("conflict",)),
        inv("journal_replay", "Replay does not duplicate effects.", mutations, ("conflict",)),
        inv("versioning", "Stale base versions fail closed.", ("write", "remove", "rename"), ("stale_version",)),
        inv("cache_pin_coherence", "Cache cannot bypass authorization.", ("read", "stat", "list"), ("permission_denied",)),
        inv("backend_negotiation", "Cross-backend ops need capability.", path_ops, ("capability_unavailable", "unsupported")),
        inv("authorization", "Auth precedes content exposure.", _LOCKED_VFS_OPERATIONS, ("permission_denied", "authentication_required")),
        inv("resource", "Resource bounds are explicit.", ("list", "read", "write", "copy"), ("resource_exhausted",)),
        inv("degradation", "No silent capability fallback.", mutations + ("read",), ("capability_unavailable",)),
    )

    # Build per-operation invariant/error coverage from applicability.
    inv_by_id = {item.invariant_id: item for item in invariants}
    operations: list[OperationContract] = []
    mutation_set = set(mutations)
    for operation in vocab.operations:
        inv_ids = tuple(
            item.invariant_id
            for item in invariants
            if operation in item.applies_to
        )
        errors = sorted(
            {
                error
                for inv_id in inv_ids
                for error in inv_by_id[inv_id].error_codes
            }
        )
        if operation in ("read",):
            inputs: tuple[DataMode, ...] = (DataMode.METADATA,)
            outputs: tuple[DataMode, ...] = (DataMode.BYTES,)
        elif operation in ("write",):
            inputs = (DataMode.BYTES, DataMode.METADATA)
            outputs = (DataMode.METADATA,)
        elif operation == "open":
            inputs = (DataMode.METADATA,)
            outputs = (DataMode.HANDLE, DataMode.METADATA)
        elif operation in ("close",):
            inputs = (DataMode.HANDLE,)
            outputs = (DataMode.NONE,)
        elif operation == "seek":
            inputs = (DataMode.HANDLE, DataMode.METADATA)
            outputs = (DataMode.METADATA,)
        else:
            inputs = (DataMode.METADATA,)
            outputs = (DataMode.METADATA,)
        operations.append(
            OperationContract(
                operation=operation,
                summary=f"Canonical {operation} contract.",
                input_modes=inputs,
                output_modes=outputs,
                execution_modes=(ExecutionMode.SYNC, ExecutionMode.ASYNC),
                invariant_ids=inv_ids,
                error_codes=tuple(errors),
                source_contract_ids=("source:vfs-026-acceptance",),
                mutates=operation in mutation_set,
                idempotent=None if operation in mutation_set else True,
            )
        )

    no_handles = {"open", "close", "seek"}
    stateful = {"python", "mcp", "mcp++", "libp2p"}
    modes = {
        "python": (ExecutionMode.SYNC, ExecutionMode.ASYNC),
        "cli": (ExecutionMode.SYNC,),
        "mcp": (ExecutionMode.ASYNC,),
        "mcp++": (ExecutionMode.ASYNC,),
        "http": (ExecutionMode.ASYNC,),
        "libp2p": (ExecutionMode.ASYNC,),
    }
    surfaces: list[PublicSurfaceContract] = []
    for surface in vocab.surfaces:
        source_id = f"source:vfs-026-surface:{surface}"
        bindings = tuple(
            SurfaceOperationContract(
                operation=operation,
                support=(
                    OperationSupport.SUPPORTED
                    if surface in stateful or operation not in no_handles
                    else OperationSupport.UNSUPPORTED
                ),
                source_contract_ids=(source_id,),
                entrypoint=f"{surface}:{operation}",
            )
            for operation in vocab.operations
        )
        surfaces.append(
            PublicSurfaceContract(
                surface=surface,
                contract_name=f"{surface} facade",
                execution_modes=modes[surface],
                operations=bindings,
                source_contract_ids=(source_id,),
            )
        )

    issues = (
        ExpectationIssue(
            issue_id="issue:backend-specific-atomicity",
            kind=IssueKind.MISSING,
            subject=(
                "Backend-specific atomicity strength and cross-backend transaction "
                "support are absent until a reviewed capability contract is supplied."
            ),
            source_contract_ids=("source:missing-backend-atomicity-contract",),
            positions=(),
            state=ExpectationState.UNRESOLVED,
        ),
    )

    vectors = (
        CanonicalVector(
            vector_id="vector:path:nfc-dot-segments",
            operation="path.resolve",
            description="Unicode and dot segments canonicalize without changing identity.",
            request={"path": "/cafe\u0301//draft/../data", "version": "v7"},
            expected={"path": "/café/data", "version": "v7"},
            invariant_ids=(
                "invariant:versioned_path",
                "invariant:unicode",
                "invariant:traversal",
            ),
            source_contract_ids=("source:vfs-026-acceptance",),
            exact_semantics="NFC + collapse // + resolve . / .. without selector change.",
        ),
        CanonicalVector(
            vector_id="vector:path:root-traversal-denied",
            operation="path.resolve",
            description="Traversal above the selected root is rejected.",
            request={"path": "/../../etc/passwd"},
            expected={"error": {"code": "traversal_denied", "effects": "none"}},
            invariant_ids=("invariant:root", "invariant:traversal", "invariant:error"),
            source_contract_ids=("source:vfs-026-acceptance",),
            exact_semantics="path.resolve never escapes '/'; returns traversal_denied.",
        ),
        CanonicalVector(
            vector_id="vector:write:utf8-byte-accounting",
            operation="write",
            description="Explicit UTF-8 text adapter reports encoded byte size.",
            request={"path": "/café.txt", "text": "é", "encoding": "utf-8"},
            expected={"committed_bytes_hex": "c3a9", "size": 2, "written": 2},
            invariant_ids=(
                "invariant:bytes_text",
                "invariant:read_write",
                "invariant:cid_size",
            ),
            source_contract_ids=("source:vfs-026-acceptance",),
            exact_semantics="UTF-8 'é' commits exactly two bytes c3a9.",
        ),
    )

    facade_examples = (
        FacadeExample(
            example_id="facade:python:compatible-bytes",
            surface="python",
            compatibility=FacadeCompatibility.COMPATIBLE,
            description="Python returns bytes for read.",
            operation="read",
            example={"returns": "bytes"},
            rationale="Matches bytes_text.",
            source_contract_ids=("source:vfs-026-acceptance",),
        ),
        FacadeExample(
            example_id="facade:cli:incompatible-handle",
            surface="cli",
            compatibility=FacadeCompatibility.INCOMPATIBLE,
            description="CLI cannot expose persistent handles.",
            operation="open",
            example={"subcommand": "open"},
            rationale="Handle lifecycle unsupported on CLI.",
            source_contract_ids=("source:vfs-026-surface:cli",),
        ),
        FacadeExample(
            example_id="facade:http:unresolved-atomicity",
            surface="http",
            compatibility=FacadeCompatibility.UNRESOLVED,
            description="HTTP cross-backend atomicity not reviewed.",
            operation="copy",
            example={"atomic": True},
            rationale="Missing backend atomicity contract.",
            source_contract_ids=("source:missing-backend-atomicity-contract",),
        ),
    )

    compiler = ProgramContractProfileCompiler(
        schema=_LOCKED_VFS_SCHEMA,
        operation_matrix_schema=_LOCKED_VFS_MATRIX_SCHEMA,
        contract_version=_LOCKED_VFS_VERSION,
        goal_id=_LOCKED_VFS_GOAL,
        profile_id="profile:locked-vfs-projection@1",
    )
    return compiler.compile(
        vocab,
        sources=sources,
        invariants=invariants,
        operations=tuple(operations),
        surfaces=tuple(surfaces),
        issues=issues,
        vectors=vectors,
        facade_examples=facade_examples,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generic_module_has_no_domain_literals() -> None:
    text = PROFILE_MODULE.read_text(encoding="utf-8")
    sanitized = text.replace(
        "ipfs_accelerate_py/agent-supervisor/program-contract-profile@1",
        "<SCHEMA>",
    )
    sanitized = sanitized.replace("ipfs_accelerate_py", "<PKG>")
    hits = _FORBIDDEN_GENERIC.findall(sanitized)
    assert hits == [], f"generic module contains domain literals: {hits[:10]}"


def test_generic_module_has_no_implicit_provider_imports() -> None:
    text = PROFILE_MODULE.read_text(encoding="utf-8")
    assert "llm_router" not in text
    assert "importlib" not in text
    assert "integrations" not in text
    assert "todo_daemon" not in text


def test_rpc_key_value_profile_exercises_every_generic_record() -> None:
    profile = build_rpc_key_value_profile()
    record = profile.to_record()

    assert profile.vocabulary.operations
    assert profile.sources
    assert profile.invariants
    assert profile.operations
    assert profile.surfaces
    assert profile.issues
    assert profile.vectors
    assert profile.facade_examples

    # Every generic record type appears with evidence and exact semantics.
    assert all(item.source_contract_ids for item in profile.operations)
    assert all(item.source_contract_ids for item in profile.invariants)
    assert all(
        item.exact_semantics and item.source_contract_ids for item in profile.vectors
    )
    assert any(item.kind is IssueKind.MISSING for item in profile.issues)
    assert any(item.kind is IssueKind.CONFLICT for item in profile.issues)
    assert {item.compatibility for item in profile.facade_examples} == {
        FacadeCompatibility.COMPATIBLE,
        FacadeCompatibility.INCOMPATIBLE,
        FacadeCompatibility.UNRESOLVED,
    }

    # Unresolved / conflicting expectations are preserved, not popular-resolved.
    unresolved = profile.unresolved_expectations
    assert len(unresolved) == 2
    assert all(item.resolution is None for item in unresolved)
    conflict = next(item for item in unresolved if item.kind is IssueKind.CONFLICT)
    assert conflict.state is ExpectationState.CONFLICTING
    assert len(conflict.positions) == 2

    assert record["authority"] == {
        "completion_evidence": False,
        "correctness_evidence": False,
        "authorizes_repair": False,
    }
    assert not PROFILE_IS_COMPLETION_EVIDENCE
    assert not PROFILE_IS_CORRECTNESS_EVIDENCE
    assert not PROFILE_AUTHORIZES_REPAIR
    assert profile.content_id == _canonical_digest(record)
    assert profile.to_record() == build_rpc_key_value_profile().to_record()
    assert json.loads(profile.to_json()) == record


def test_vfs_profile_projection_preserves_locked_identities_not_module_paths() -> None:
    profile = build_vfs_equivalent_profile()
    record = profile.to_record()

    assert profile.schema == _LOCKED_VFS_SCHEMA
    assert profile.operation_matrix_schema == _LOCKED_VFS_MATRIX_SCHEMA
    assert profile.contract_version == _LOCKED_VFS_VERSION
    assert profile.goal_id == _LOCKED_VFS_GOAL
    assert tuple(profile.vocabulary.operations) == _LOCKED_VFS_OPERATIONS
    assert tuple(profile.vocabulary.invariant_kinds) == _LOCKED_VFS_INVARIANTS
    assert {item.kind for item in profile.invariants} == set(_LOCKED_VFS_INVARIANTS)
    assert {item.operation for item in profile.operations} == set(_LOCKED_VFS_OPERATIONS)
    assert {item.surface for item in profile.surfaces} == set(_LOCKED_VFS_SURFACES)

    # Semantic identity is schema/vocabulary/content_id — not a module path.
    assert not record["schema"].endswith(".py")
    assert "vfs_contract_pack.py" not in json.dumps(record)
    assert "program_contract_profile.py" not in json.dumps(record)
    assert record["vocabulary"]["vocabulary_identity"].startswith("sha256:")
    assert profile.content_id.startswith("sha256:")

    vectors = {item.vector_id: item for item in profile.vectors}
    assert vectors["vector:path:nfc-dot-segments"].expected["path"] == "/café/data"
    assert (
        vectors["vector:path:root-traversal-denied"].expected["error"]["code"]
        == "traversal_denied"
    )
    assert vectors["vector:write:utf8-byte-accounting"].expected["size"] == 2

    # Missing expectation remains unresolved (no popular backend selection).
    assert len(profile.unresolved_expectations) == 1
    assert profile.unresolved_expectations[0].state is ExpectationState.UNRESOLVED
    assert profile.unresolved_expectations[0].resolution is None

    # Handle lifecycle unsupported on stateless facades matches locked mapping.
    for surface_name in ("cli", "http"):
        surface = profile.surface_contract(surface_name)
        unsupported = {
            item.operation
            for item in surface.operations
            if item.support is OperationSupport.UNSUPPORTED
        }
        assert unsupported == {"open", "close", "seek"}


def test_source_lock_declares_contract_pack_generalization() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    modules = lock.get("modules") or lock.get("source_modules") or lock
    if isinstance(modules, dict):
        entries = modules.get("entries") or modules.get("modules") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
    else:
        entries = modules
    pack = next(
        item
        for item in entries
        if item.get("source_path", "").endswith("vfs_contract_pack.py")
        or item.get("planned_path", "").endswith("program_contract_profile.py")
    )
    assert pack["source_blob"] == "9acc4ceba42b8767f5b4e4b6ce7d4bc55893bcf2"
    assert pack["planned_path"].endswith("program_contract_profile.py")
    assert pack["schemas_and_identity"]["VFS_CONTRACT_PACK_SCHEMA"] == _LOCKED_VFS_SCHEMA


def test_vocabulary_rejects_duplicates_and_unknown_entries() -> None:
    with pytest.raises(ProgramContractProfileError, match="must be unique"):
        ContractVocabulary(
            operations=("a", "a"),
            invariant_kinds=("i",),
            error_codes=("e",),
            surfaces=("s",),
        )
    with pytest.raises(ProgramContractProfileError, match="must be non-empty"):
        ContractVocabulary(
            operations=(),
            invariant_kinds=("i",),
            error_codes=("e",),
            surfaces=("s",),
        )

    profile = build_rpc_key_value_profile()
    with pytest.raises(
        ProgramContractProfileError,
        match="cover the vocabulary|not in the closed vocabulary",
    ):
        ProgramContractProfileCompiler().compile(
            profile.vocabulary,
            sources=profile.sources,
            invariants=profile.invariants,
            operations=(
                replace(profile.operations[0], operation="kv.unknown"),
                *profile.operations[1:],
            ),
            surfaces=profile.surfaces,
            issues=profile.issues,
            vectors=profile.vectors,
            facade_examples=profile.facade_examples,
        )


def test_unknown_vocabulary_entry_fails_closed_via_compiler() -> None:
    profile = build_rpc_key_value_profile()
    bad_ops = (
        replace(profile.operations[0], operation="kv.unknown"),
        *profile.operations[1:],
    )
    with pytest.raises(ProgramContractProfileError) as excinfo:
        ProgramContractProfileCompiler(
            schema=profile.schema,
            operation_matrix_schema=profile.operation_matrix_schema,
            contract_version=profile.contract_version,
            goal_id=profile.goal_id,
            profile_id=profile.profile_id,
        ).compile(
            profile.vocabulary,
            sources=profile.sources,
            invariants=profile.invariants,
            operations=bad_ops,
            surfaces=profile.surfaces,
            issues=profile.issues,
            vectors=profile.vectors,
            facade_examples=profile.facade_examples,
        )
    assert "unknown" in str(excinfo.value).lower() or "cover" in str(excinfo.value).lower()


def test_unbounded_fields_are_rejected() -> None:
    huge = "x" * (MAX_TEXT_BYTES + 8)
    with pytest.raises(ProgramContractProfileError, match="exceeds"):
        SourceContract(
            source_id="source:huge",
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            locator="task://x",
            revision="r1",
            summary=huge,
            reviewed=True,
        )


def test_self_authority_and_observation_authority_are_rejected() -> None:
    common = {
        "source_id": "source:test",
        "locator": "test://source",
        "revision": "r1",
        "summary": "Test source.",
        "reviewed": True,
    }
    with pytest.raises(ProgramContractProfileError, match="observation-only"):
        SourceContract(
            kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
            **common,
        )
    with pytest.raises(ProgramContractProfileError, match="self-authority"):
        SourceContract(
            source_id="self",
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            locator="task://x",
            revision="r1",
            summary="Bad self id.",
            reviewed=True,
        )
    with pytest.raises(ProgramContractProfileError, match="self-authority"):
        SourceContract(
            source_id="source:ok",
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            locator="self://authority",
            revision="r1",
            summary="Bad self locator.",
            reviewed=True,
        )
    with pytest.raises(ProgramContractProfileError, match="available and reviewed"):
        SourceContract(
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            available=False,
            **common,
        )


def test_forged_content_id_and_schema_module_path_are_rejected() -> None:
    profile = build_rpc_key_value_profile()
    with pytest.raises(ProgramContractProfileError, match="forged or drifted"):
        ProgramContractProfileCompiler(
            schema=profile.schema,
            operation_matrix_schema=profile.operation_matrix_schema,
            contract_version=profile.contract_version,
            goal_id=profile.goal_id,
            profile_id=profile.profile_id,
        ).compile(
            profile.vocabulary,
            sources=profile.sources,
            invariants=profile.invariants,
            operations=profile.operations,
            surfaces=profile.surfaces,
            issues=profile.issues,
            vectors=profile.vectors,
            facade_examples=profile.facade_examples,
            expected_content_id="sha256:" + ("0" * 64),
        )

    with pytest.raises(ProgramContractProfileError, match="module path"):
        ProgramContractProfileCompiler(
            schema="analysis/program_contract_profile.py",
            operation_matrix_schema=profile.operation_matrix_schema,
            contract_version=profile.contract_version,
        ).compile(
            profile.vocabulary,
            sources=profile.sources,
            invariants=profile.invariants,
            operations=profile.operations,
            surfaces=profile.surfaces,
            issues=profile.issues,
            vectors=profile.vectors,
            facade_examples=profile.facade_examples,
        )


def test_missing_and_conflicting_expectations_fail_closed() -> None:
    profile = build_rpc_key_value_profile()
    with pytest.raises(ProgramContractProfileError, match="must stay unresolved"):
        replace(profile.issues[0], state=ExpectationState.RESOLVED)
    with pytest.raises(ProgramContractProfileError, match="at least two sources"):
        ExpectationIssue(
            issue_id="issue:one-sided",
            kind=IssueKind.CONFLICT,
            subject="One source cannot establish a conflict.",
            source_contract_ids=("source:rpc-kv-acceptance",),
            positions=("a", "b"),
            state=ExpectationState.CONFLICTING,
        )
    with pytest.raises(ProgramContractProfileError, match="cannot select a resolution"):
        ExpectationIssue(
            issue_id="issue:resolved-conflict",
            kind=IssueKind.CONFLICT,
            subject="Must not pick a winner.",
            source_contract_ids=(
                "source:rpc-kv-conflict-a",
                "source:rpc-kv-conflict-b",
            ),
            positions=("a", "b"),
            state=ExpectationState.CONFLICTING,
            resolution="pick-a",
        )


def test_resolved_vectors_require_source_and_exact_semantics() -> None:
    with pytest.raises(ProgramContractProfileError, match="exact semantics"):
        CanonicalVector(
            vector_id="vector:bad",
            operation="kv.get",
            description="Missing semantics.",
            request={"key": "k"},
            expected={"ok": True},
            invariant_ids=("invariant:key_identity",),
            source_contract_ids=("source:rpc-kv-acceptance",),
            exact_semantics="",
        )
    with pytest.raises(ProgramContractProfileError, match="needs a source"):
        CanonicalVector(
            vector_id="vector:unbacked",
            operation="kv.get",
            description="No source.",
            request={},
            expected={},
            invariant_ids=(),
            source_contract_ids=(),
            exact_semantics="must have source",
        )


def test_pack_rejects_unbacked_resolved_contracts_and_incomplete_matrix() -> None:
    profile = build_rpc_key_value_profile()
    unbacked = replace(
        profile.operations[0],
        source_contract_ids=("source:rpc-kv-missing-batch",),
    )
    with pytest.raises(
        ProgramContractProfileError,
        match="without reviewed authority",
    ):
        ProgramContractProfileCompiler(
            schema=profile.schema,
            operation_matrix_schema=profile.operation_matrix_schema,
            contract_version=profile.contract_version,
            goal_id=profile.goal_id,
            profile_id=profile.profile_id,
        ).compile(
            profile.vocabulary,
            sources=profile.sources,
            invariants=profile.invariants,
            operations=(unbacked, *profile.operations[1:]),
            surfaces=profile.surfaces,
            issues=profile.issues,
            vectors=profile.vectors,
            facade_examples=profile.facade_examples,
        )

    incomplete_surface = replace(
        profile.surfaces[0],
        operations=profile.surfaces[0].operations[:-1],
    )
    with pytest.raises(ProgramContractProfileError, match="incomplete"):
        ProgramContractProfileCompiler(
            schema=profile.schema,
            operation_matrix_schema=profile.operation_matrix_schema,
            contract_version=profile.contract_version,
            goal_id=profile.goal_id,
            profile_id=profile.profile_id,
        ).compile(
            profile.vocabulary,
            sources=profile.sources,
            invariants=profile.invariants,
            operations=profile.operations,
            surfaces=(incomplete_surface, *profile.surfaces[1:]),
            issues=profile.issues,
            vectors=profile.vectors,
            facade_examples=profile.facade_examples,
        )


def test_assert_complete_and_atomic_publication(tmp_path: Path) -> None:
    profile = build_rpc_key_value_profile()
    assert_contract_profile_complete(profile)
    assert isinstance(profile, ProgramContractProfile)

    destination = tmp_path / "nested" / "rpc-kv-profile.json"
    published = publish_contract_profile(destination, profile)
    assert published == destination.resolve()
    assert json.loads(destination.read_text(encoding="utf-8")) == profile.to_record()
    assert not list(destination.parent.glob("*.tmp"))

    first_bytes = destination.read_bytes()
    assert publish_contract_profile(destination, profile) == destination.resolve()
    assert destination.read_bytes() == first_bytes


def test_discover_profile_schemas_are_domain_neutral() -> None:
    schemas = discover_profile_schemas()
    assert PROGRAM_CONTRACT_PROFILE_SCHEMA in schemas
    assert PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA in schemas
    joined = " ".join(schemas).lower()
    assert "vfs" not in joined
    assert "fsspec" not in joined
    assert "swissknife" not in joined


def test_vocabulary_and_profile_identities_are_stable() -> None:
    a = build_rpc_key_value_profile()
    b = build_rpc_key_value_profile()
    assert a.vocabulary.identity() == b.vocabulary.identity()
    assert a.content_id == b.content_id
    vfs_a = build_vfs_equivalent_profile()
    vfs_b = build_vfs_equivalent_profile()
    assert vfs_a.content_id == vfs_b.content_id
    # Distinct domains produce distinct identities.
    assert a.content_id != vfs_a.content_id
    assert a.vocabulary.identity() != vfs_a.vocabulary.identity()
