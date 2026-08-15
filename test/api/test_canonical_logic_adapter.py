"""LPC-090 regression: supervisor compatibility maps from the catalog.

The durable generated artifact is
``data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md``.
This module parses every ``supervisor-map`` fence, expands the catalog-root
binding against ``DEFAULT_CANONICAL_CATALOG_SNAPSHOT``, cross-checks live
adapter projections, and enforces fail-closed lookup for unknown values.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    CacheScope,
    LogicFamily,
)
from ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter import (
    SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
    CanonicalLogicAdapterError,
    SupervisorCanonicalLogicAdapter,
    VocabularyProjection,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderIsolation,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_translation_validation import (
    LogicForm,
    TranslationClass,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import PropertyKind
from ipfs_datasets_py.logic.families.canonical_catalog import (
    DEFAULT_CANONICAL_CATALOG_SNAPSHOT,
)


# ---------------------------------------------------------------------------
# Errors / constants
# ---------------------------------------------------------------------------


class SupervisorMapCutoverError(ValueError):
    """Raised when a supervisor legacy value cannot be mapped fail-closed."""


REQUIRED_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "canonical_identity",
        "disposition",
        "residual",
        "deprecation",
        "catalog_root",
    }
)

REQUIRED_MAP_DOMAINS: Final[tuple[str, ...]] = (
    "analysis_family",
    "property_kind",
    "logic_form",
    "translation_class",
    "cache_scope",
    "provider_route",
    "provider_isolation",
    "operation_status",
    "semantic_verdict",
    "availability",
    "evidence_kind",
    "evidence_authority",
    "assurance_authority",
    "translation_preservation",
    "reject_merge",
    "operational_only",
)

MAP_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "map",
        "residual_collapse",
        "compatibility_alias",
        "reject_merge",
        "operational_only",
    }
)

ADAPTER_PROJECTABLE_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "analysis_family",
        "property_kind",
        "logic_form",
        "translation_class",
        "cache_scope",
        "provider_route",
        "provider_isolation",
    }
)

_SUPERVISOR_MAP_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```supervisor-map\n(.*?)\n```",
    re.DOTALL,
)

_SUPERVISOR_MAP_META_RE: Final[re.Pattern[str]] = re.compile(
    r"```supervisor-map-meta\n(.*?)\n```",
    re.DOTALL,
)

_META_KEYS: Final[frozenset[str]] = frozenset(
    {
        "domain",
        "surface",
        "disposition",
        "fail_closed",
        "catalog_root",
        "target_axis",
    }
)

CUTOVER_NOTE_RELATIVE: Final[Path] = Path(
    "data/agent_supervisor/logic_platform_canonicalization/notes/"
    "supervisor_map_cutover.md"
)


# ---------------------------------------------------------------------------
# Note loading / parsing
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / CUTOVER_NOTE_RELATIVE).is_file() and (
            parent / "ipfs_accelerate_py"
        ).is_dir():
            return parent
    return here.parents[2]


def cutover_note_path() -> Path:
    return _repo_root() / CUTOVER_NOTE_RELATIVE


def sealed_catalog_root() -> str:
    root = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
    if not isinstance(root, str) or not root:
        raise SupervisorMapCutoverError("sealed catalog root is missing")
    return root


def sealed_catalog_digest() -> str:
    digest = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise SupervisorMapCutoverError("sealed catalog digest is missing")
    return digest


def _expand_catalog_root(raw: str, *, live_root: str) -> str:
    token = raw.strip()
    if token in {"$catalog_root_binding", "DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root"}:
        return live_root
    if not token:
        raise SupervisorMapCutoverError("catalog_root must be non-empty")
    return token


def _parse_residual(raw: str) -> Mapping[str, str]:
    residual: dict[str, str] = {}
    for part in raw.split("|"):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise SupervisorMapCutoverError(
                f"residual must be k=v pairs joined by |; got {raw!r}"
            )
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SupervisorMapCutoverError(f"empty residual pair in {raw!r}")
        residual[key] = value
    if not residual:
        raise SupervisorMapCutoverError(f"empty residual: {raw!r}")
    return MappingProxyType(residual)


def _parse_row_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in raw.split(";"):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise SupervisorMapCutoverError(
                f"row field must be key=value; got {piece!r} in {raw!r}"
            )
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SupervisorMapCutoverError(f"empty field in row {raw!r}")
        if key in fields:
            raise SupervisorMapCutoverError(f"duplicate field {key!r} in row {raw!r}")
        fields[key] = value
    return fields


def parse_supervisor_map_meta(text: str) -> Mapping[str, str]:
    match = _SUPERVISOR_MAP_META_RE.search(text)
    if match is None:
        raise SupervisorMapCutoverError("missing supervisor-map-meta fence")
    meta: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SupervisorMapCutoverError(
                f"meta line must be key: value; got {raw_line!r}"
            )
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    required = {
        "schema",
        "task",
        "goal",
        "adapter_interface",
        "catalog_interface",
        "catalog_root_binding",
        "fail_closed",
    }
    missing = required - set(meta)
    if missing:
        raise SupervisorMapCutoverError(f"supervisor-map-meta missing {sorted(missing)}")
    if meta["task"] != "LPC-090":
        raise SupervisorMapCutoverError(f"unexpected task id {meta['task']!r}")
    if meta["adapter_interface"] != SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE:
        raise SupervisorMapCutoverError(
            f"adapter interface mismatch: {meta['adapter_interface']!r}"
        )
    if meta["fail_closed"].lower() != "true":
        raise SupervisorMapCutoverError("cutover must declare fail_closed: true")
    return MappingProxyType(meta)


def parse_supervisor_map_blocks(
    text: str,
    *,
    live_catalog_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse every ``supervisor-map`` fence into domain records."""

    live_root = live_catalog_root or sealed_catalog_root()
    domains: dict[str, dict[str, Any]] = {}
    reject_surfaces: list[dict[str, Any]] = []
    operational_surfaces: list[dict[str, Any]] = []

    for match in _SUPERVISOR_MAP_FENCE_RE.finditer(text):
        body = match.group(1)
        meta: dict[str, str] = {}
        labels: dict[str, dict[str, Any]] = {}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise SupervisorMapCutoverError(
                    f"supervisor-map line must be key: value; got {raw_line!r}"
                )
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in _META_KEYS:
                meta[key] = value
                continue
            fields = _parse_row_fields(value)
            missing = REQUIRED_ROW_FIELDS - set(fields)
            if missing:
                raise SupervisorMapCutoverError(
                    f"legacy value {key!r} missing fields {sorted(missing)}"
                )
            disposition = fields["disposition"]
            if disposition not in MAP_DISPOSITIONS:
                raise SupervisorMapCutoverError(
                    f"unknown disposition {disposition!r} for {key!r}"
                )
            catalog_root = _expand_catalog_root(
                fields["catalog_root"], live_root=live_root
            )
            if catalog_root != live_root:
                raise SupervisorMapCutoverError(
                    f"catalog_root for {key!r} does not match sealed snapshot "
                    f"({catalog_root!r} != {live_root!r})"
                )
            residual = _parse_residual(fields["residual"])
            if "supervisor_id" not in residual:
                raise SupervisorMapCutoverError(
                    f"residual for {key!r} must include supervisor_id"
                )
            if disposition != "compatibility_alias" and residual["supervisor_id"] != key:
                raise SupervisorMapCutoverError(
                    f"residual supervisor_id for {key!r} must equal the legacy "
                    f"value unless disposition is compatibility_alias"
                )
            labels[key] = {
                "legacy_value": key,
                "canonical_identity": fields["canonical_identity"],
                "disposition": disposition,
                "residual": residual,
                "deprecation": fields["deprecation"],
                "catalog_root": catalog_root,
            }

        domain = meta.get("domain")
        surface = meta.get("surface")
        if not domain or not surface:
            raise SupervisorMapCutoverError(
                "supervisor-map block requires domain and surface"
            )
        fail_closed = meta.get("fail_closed", "true").lower() == "true"
        if not fail_closed:
            raise SupervisorMapCutoverError(
                f"surface {surface!r} must declare fail_closed: true"
            )
        block_catalog_root = _expand_catalog_root(
            meta.get("catalog_root", "$catalog_root_binding"),
            live_root=live_root,
        )
        if block_catalog_root != live_root:
            raise SupervisorMapCutoverError(
                f"block catalog_root for {surface!r} does not match sealed snapshot"
            )

        disposition = meta.get("disposition", "map")
        if disposition in {"reject_merge", "operational_only"}:
            if labels:
                raise SupervisorMapCutoverError(
                    f"surface {surface!r} disposition {disposition} must not "
                    f"declare label maps; got {sorted(labels)!r}"
                )
            record = {
                "domain": domain,
                "surface": surface,
                "disposition": disposition,
                "fail_closed": True,
                "catalog_root": block_catalog_root,
                "labels": MappingProxyType({}),
            }
            if disposition == "reject_merge":
                reject_surfaces.append(record)
            else:
                operational_surfaces.append(record)
            # Index by surface for lookup helpers.
            domains[f"{domain}:{surface}"] = record
            continue

        if not labels:
            raise SupervisorMapCutoverError(
                f"surface {surface!r} map disposition requires at least one label"
            )
        if domain in domains and domain not in {
            # translation_class appears once; translation_preservation is distinct
        }:
            # Multiple blocks with same domain are allowed only when surfaces differ
            # and we key by domain:surface for axis dual views.
            pass
        key_name = domain if domain not in domains else f"{domain}:{surface}"
        if key_name in domains:
            raise SupervisorMapCutoverError(
                f"duplicate supervisor-map domain key {key_name!r}"
            )
        domains[key_name] = {
            "domain": domain,
            "surface": surface,
            "disposition": disposition,
            "fail_closed": True,
            "catalog_root": block_catalog_root,
            "labels": MappingProxyType(labels),
        }

    # Attach grouped views for convenience.
    domains["__reject_merge__"] = {
        "domain": "reject_merge",
        "surfaces": tuple(item["surface"] for item in reject_surfaces),
        "records": tuple(reject_surfaces),
    }
    domains["__operational_only__"] = {
        "domain": "operational_only",
        "surfaces": tuple(item["surface"] for item in operational_surfaces),
        "records": tuple(operational_surfaces),
    }
    return domains


def load_supervisor_maps(
    note_path: Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    path = note_path if note_path is not None else cutover_note_path()
    text = path.read_text(encoding="utf-8")
    parse_supervisor_map_meta(text)
    return MappingProxyType(parse_supervisor_map_blocks(text))


def _label_value(raw: object) -> str:
    if isinstance(raw, Enum):
        value = raw.value
    else:
        value = raw
    if not isinstance(value, str) or not value or value != value.strip():
        raise SupervisorMapCutoverError(
            f"legacy label must be a non-empty trimmed string; got {raw!r}"
        )
    return value


def map_supervisor_legacy(domain: str, label: object) -> Mapping[str, Any]:
    """Fail-closed lookup of one legacy value against the cutover artifact."""

    maps = load_supervisor_maps()
    if domain not in maps:
        # Allow domain:surface keys only for reject/operational; primary domains
        # are keyed by domain name.
        raise SupervisorMapCutoverError(f"unknown supervisor map domain {domain!r}")
    record = maps[domain]
    disposition = record.get("disposition", "map")
    if disposition == "reject_merge":
        raise SupervisorMapCutoverError(
            f"domain {domain!r} surface {record.get('surface')!r} is reject_merge"
        )
    if disposition == "operational_only":
        raise SupervisorMapCutoverError(
            f"domain {domain!r} surface {record.get('surface')!r} is operational_only"
        )
    key = _label_value(label)
    labels: Mapping[str, Mapping[str, Any]] = record["labels"]
    if key not in labels:
        allowed = ", ".join(sorted(labels))
        raise SupervisorMapCutoverError(
            f"unknown label {key!r} for domain {domain!r}; allowed: {allowed}"
        )
    return labels[key]


# ---------------------------------------------------------------------------
# Live adapter helpers
# ---------------------------------------------------------------------------


def _adapter() -> SupervisorCanonicalLogicAdapter:
    return SupervisorCanonicalLogicAdapter()


def _unique_enum_values(enum_type: type[Enum]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for member in enum_type:
        value = str(member.value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _project_adapter(domain: str, legacy: str) -> str:
    adapter = _adapter()
    if domain == "analysis_family":
        return adapter.project_analysis_family(legacy).canonical_id
    if domain == "property_kind":
        return adapter.project_property_kind(legacy).canonical_id
    if domain == "logic_form":
        return adapter.project_logic_form(legacy).canonical_id
    if domain == "translation_class":
        return adapter.project_translation_class(legacy).canonical_id
    if domain == "cache_scope":
        return adapter.project_cache_scope(legacy).canonical_id
    if domain == "provider_route":
        return adapter.map_prover_id_to_canonical_provider(legacy)
    if domain == "provider_isolation":
        # Isolation projection is embedded in capability projection maps.
        from ipfs_accelerate_py.agent_supervisor.proof import canonical_logic_adapter as mod

        mapped = getattr(mod, "_ISOLATION_TO_RUNTIME")
        if legacy not in mapped:
            raise CanonicalLogicAdapterError(f"unsupported isolation: {legacy}")
        return str(mapped[legacy])
    raise SupervisorMapCutoverError(f"domain {domain!r} is not adapter-projectable")


# ---------------------------------------------------------------------------
# Structural / catalog-root tests
# ---------------------------------------------------------------------------


def test_cutover_note_exists_and_declares_lpc_090() -> None:
    path = cutover_note_path()
    assert path.is_file(), f"missing cutover artifact: {path}"
    text = path.read_text(encoding="utf-8")
    assert "LPC-090" in text
    assert "SupervisorCanonicalLogicAdapter@1" in text
    meta = parse_supervisor_map_meta(text)
    assert meta["catalog_root_binding"] == (
        "DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root"
    )
    assert meta["catalog_interface"] == "CanonicalLogicCatalogSnapshot@1"


def test_every_required_domain_is_present() -> None:
    maps = load_supervisor_maps()
    present_domains = {
        record["domain"]
        for key, record in maps.items()
        if not key.startswith("__") and isinstance(record, Mapping) and "domain" in record
    }
    missing = set(REQUIRED_MAP_DOMAINS) - present_domains
    assert not missing, f"missing supervisor-map domains: {sorted(missing)}"


def test_every_mapped_row_has_required_fields_and_catalog_root() -> None:
    live_root = sealed_catalog_root()
    maps = load_supervisor_maps()
    row_count = 0
    for key, record in maps.items():
        if key.startswith("__"):
            continue
        if record.get("disposition") in {"reject_merge", "operational_only"}:
            assert record["catalog_root"] == live_root
            assert record["fail_closed"] is True
            continue
        labels = record["labels"]
        assert labels, f"domain {key} has no labels"
        for legacy, row in labels.items():
            row_count += 1
            for field in REQUIRED_ROW_FIELDS:
                assert field in row, f"{legacy} missing {field}"
            assert row["catalog_root"] == live_root
            assert row["disposition"] in MAP_DISPOSITIONS
            assert row["canonical_identity"]
            assert "supervisor_id" in row["residual"]
            assert row["deprecation"]
    assert row_count >= 50, f"expected exhaustive rows, found {row_count}"


def test_catalog_root_matches_sealed_snapshot_identity() -> None:
    live_root = sealed_catalog_root()
    live_digest = sealed_catalog_digest()
    assert live_root.startswith("b")
    assert live_digest.startswith("sha256:")
    # Rebuild identity is stable (LPC-020 / LPC-023).
    rebuilt = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_identity()
    assert rebuilt.cid == live_root
    assert rebuilt.digest == live_digest


# ---------------------------------------------------------------------------
# Fail-closed lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain",
    [
        "analysis_family",
        "property_kind",
        "logic_form",
        "translation_class",
        "cache_scope",
        "provider_route",
        "provider_isolation",
        "operation_status",
        "semantic_verdict",
        "availability",
        "evidence_kind",
        "evidence_authority",
        "assurance_authority",
        "translation_preservation",
    ],
)
def test_unknown_legacy_values_fail_closed(domain: str) -> None:
    with pytest.raises(SupervisorMapCutoverError):
        map_supervisor_legacy(domain, "__not_a_supervisor_legacy_value__")
    with pytest.raises(SupervisorMapCutoverError):
        map_supervisor_legacy(domain, "")


def test_reject_merge_surfaces_fail_closed() -> None:
    maps = load_supervisor_maps()
    reject = maps["__reject_merge__"]
    assert set(reject["surfaces"]) == {
        "goal_quality.EvidenceAuthority",
        "prompt_workflow.EvidenceAuthority",
        "plan_analysis.EvidenceAuthority",
        "planner_doctor.EvidenceAuthorityClass",
        "repository_surface.EvidenceKind",
    }
    for record in reject["records"]:
        key = f"reject_merge:{record['surface']}"
        with pytest.raises(SupervisorMapCutoverError):
            map_supervisor_legacy(key, "proof")


def test_operational_only_resource_budget_fails_closed() -> None:
    with pytest.raises(SupervisorMapCutoverError):
        map_supervisor_legacy("operational_only:supervisor.ResourceBudget", "wall_time_ms")


# ---------------------------------------------------------------------------
# Adapter consistency for projectable domains
# ---------------------------------------------------------------------------


def test_analysis_family_rows_match_adapter_and_round_trip() -> None:
    maps = load_supervisor_maps()
    labels = maps["analysis_family"]["labels"]
    adapter = _adapter()
    live = set(_unique_enum_values(LogicFamily))
    assert set(labels) == live
    for legacy, row in labels.items():
        projection = adapter.project_analysis_family(legacy)
        assert projection.canonical_id == row["canonical_identity"]
        assert projection.domain == "analysis_family"
        assert projection.residual["supervisor_id"] == legacy
        restored = adapter.restore_analysis_family(projection)
        assert restored.value == legacy
    # Unknown fails closed on adapter / family normalizer.
    with pytest.raises((CanonicalLogicAdapterError, ValueError)):
        adapter.project_analysis_family("not_a_family")


def test_flogic_and_frame_residual_collapse_is_lossless() -> None:
    maps = load_supervisor_maps()
    labels = maps["analysis_family"]["labels"]
    assert labels["flogic"]["canonical_identity"] == "frame_logic"
    assert labels["frame"]["canonical_identity"] == "frame_logic"
    assert labels["flogic"]["disposition"] == "residual_collapse"
    assert labels["frame"]["disposition"] == "residual_collapse"
    adapter = _adapter()
    flogic = adapter.project_analysis_family(LogicFamily.FLOGIC)
    frame = adapter.project_analysis_family(LogicFamily.FRAME)
    assert flogic.canonical_id == frame.canonical_id == "frame_logic"
    assert adapter.restore_analysis_family(flogic) is LogicFamily.FLOGIC
    assert adapter.restore_analysis_family(frame) is LogicFamily.FRAME


def test_property_kind_rows_match_adapter_and_round_trip() -> None:
    maps = load_supervisor_maps()
    labels = maps["property_kind"]["labels"]
    adapter = _adapter()
    live = set(_unique_enum_values(PropertyKind))
    assert set(labels) == live
    for legacy, row in labels.items():
        projection = adapter.project_property_kind(legacy)
        assert projection.canonical_id == row["canonical_identity"]
        restored = adapter.restore_property_kind(projection)
        assert restored.value == legacy
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_property_kind("not_a_property")


def test_protocol_and_runtime_trace_share_canonical_with_residual() -> None:
    maps = load_supervisor_maps()
    labels = maps["property_kind"]["labels"]
    assert labels["protocol"]["canonical_identity"] == "trace_conformance"
    assert labels["runtime_trace"]["canonical_identity"] == "trace_conformance"
    adapter = _adapter()
    protocol = adapter.project_property_kind(PropertyKind.PROTOCOL)
    runtime = adapter.project_property_kind(PropertyKind.RUNTIME_TRACE)
    assert protocol.canonical_id == runtime.canonical_id == "trace_conformance"
    assert adapter.restore_property_kind(protocol) is PropertyKind.PROTOCOL
    assert adapter.restore_property_kind(runtime) is PropertyKind.RUNTIME_TRACE


def test_logic_form_and_translation_class_rows_match_adapter() -> None:
    maps = load_supervisor_maps()
    adapter = _adapter()
    form_labels = maps["logic_form"]["labels"]
    assert set(form_labels) == set(_unique_enum_values(LogicForm))
    for legacy, row in form_labels.items():
        projection = adapter.project_logic_form(legacy)
        assert projection.canonical_id == row["canonical_identity"]
        assert adapter.restore_logic_form(projection).value == legacy

    class_labels = maps["translation_class"]["labels"]
    assert set(class_labels) == set(_unique_enum_values(TranslationClass))
    for legacy, row in class_labels.items():
        projection = adapter.project_translation_class(legacy)
        assert projection.canonical_id == row["canonical_identity"]
        assert "taxonomy_translation_kind" in projection.residual
        assert projection.residual["taxonomy_translation_kind"] == row["residual"][
            "taxonomy_translation_kind"
        ]
        assert adapter.restore_translation_class(projection).value == legacy

    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_logic_form("not_a_form")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_translation_class("not_a_class")


def test_cache_scope_rows_match_adapter() -> None:
    maps = load_supervisor_maps()
    labels = maps["cache_scope"]["labels"]
    adapter = _adapter()
    assert set(labels) == set(_unique_enum_values(CacheScope))
    for legacy, row in labels.items():
        projection = adapter.project_cache_scope(legacy)
        assert projection.canonical_id == row["canonical_identity"]
        assert adapter.restore_cache_scope(projection).value == legacy
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_cache_scope("not_a_scope")


def test_provider_route_and_isolation_rows_match_adapter() -> None:
    maps = load_supervisor_maps()
    adapter = _adapter()
    routes = maps["provider_route"]["labels"]
    assert routes["coq"]["canonical_identity"] == "rocq"
    assert routes["e"]["canonical_identity"] == "eprover"
    for legacy, row in routes.items():
        assert adapter.map_prover_id_to_canonical_provider(legacy) == row[
            "canonical_identity"
        ]

    isolations = maps["provider_isolation"]["labels"]
    assert set(isolations) == set(_unique_enum_values(ProofProviderIsolation))
    for legacy, row in isolations.items():
        assert _project_adapter("provider_isolation", legacy) == row[
            "canonical_identity"
        ]


def test_adapter_projectable_domains_are_exhaustive_against_rows() -> None:
    maps = load_supervisor_maps()
    for domain in sorted(ADAPTER_PROJECTABLE_DOMAINS):
        labels = maps[domain]["labels"]
        for legacy, row in labels.items():
            if row["disposition"] == "compatibility_alias":
                continue
            assert _project_adapter(domain, legacy) == row["canonical_identity"]


# ---------------------------------------------------------------------------
# Axis surfaces + alias deprecation
# ---------------------------------------------------------------------------


def test_attempt_status_and_proof_verdict_rows_are_closed() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        AttemptStatus,
        ProofVerdict,
    )

    maps = load_supervisor_maps()
    assert set(maps["operation_status"]["labels"]) == set(
        _unique_enum_values(AttemptStatus)
    )
    assert set(maps["semantic_verdict"]["labels"]) == set(
        _unique_enum_values(ProofVerdict)
    )
    # Success never becomes a semantic verdict in the cutover artifact.
    assert (
        maps["operation_status"]["labels"]["succeeded"]["canonical_identity"]
        == "succeeded"
    )
    assert "proved" not in maps["operation_status"]["labels"]["succeeded"]["residual"]


def test_evidence_kind_alias_and_assurance_alias_rows() -> None:
    maps = load_supervisor_maps()
    evidence = maps["evidence_kind"]["labels"]
    assert evidence["cryptographic_attestation"]["canonical_identity"] == "attestation"
    assert evidence["zkp_attestation"]["disposition"] == "compatibility_alias"
    assert evidence["zkp_attestation"]["deprecation"] == "compatibility_alias"
    assert evidence["zkp_attestation"]["residual"]["alias_of"] == (
        "cryptographic_attestation"
    )

    assurance = maps["assurance_authority"]["labels"]
    assert assurance["unverified"]["canonical_identity"] == "none"
    assert assurance["none"]["disposition"] == "compatibility_alias"
    assert assurance["solver_verified"]["disposition"] == "compatibility_alias"
    assert assurance["attested"]["canonical_identity"] == "authoritative"


def test_lookup_returns_all_five_cutover_fields() -> None:
    row = map_supervisor_legacy("analysis_family", "tdfol")
    assert row["canonical_identity"] == "tdfol"
    assert row["disposition"] == "map"
    assert row["residual"]["supervisor_id"] == "tdfol"
    assert row["deprecation"] == "active_legacy"
    assert row["catalog_root"] == sealed_catalog_root()


def test_vocabulary_projection_residual_retains_supervisor_identity() -> None:
    projection = VocabularyProjection(
        domain="analysis_family",
        supervisor_id="flogic",
        canonical_id="frame_logic",
        residual={"supervisor_enum": "LogicFamily", "supervisor_member": "FLOGIC"},
    )
    assert projection.residual["supervisor_id"] == "flogic"
    assert projection.residual["domain"] == "analysis_family"
    payload = projection.to_dict()
    restored = VocabularyProjection.from_dict(payload)
    assert restored.supervisor_id == "flogic"
    assert restored.canonical_id == "frame_logic"


def test_adapter_interface_inventory_covers_map_domains() -> None:
    adapter = _adapter()
    inventory = adapter.vocabulary_inventory()
    assert adapter.interface == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    for domain in (
        "analysis_family",
        "property_kind",
        "logic_form",
        "translation_class",
        "matrix_entry",
        "capability_probe",
        "provider",
        "route",
        "resource",
        "cache",
        "receipt",
    ):
        assert domain in inventory["domains"]


def test_kg_is_namespaced_extension_not_baseline_collapse() -> None:
    row = map_supervisor_legacy("analysis_family", "kg")
    assert row["canonical_identity"] == "supervisor.kg"
    assert row["deprecation"] == "namespaced_extension"
    # Catalog taxonomy may or may not list supervisor.kg; the cutover residual
    # must still restore the supervisor LogicFamily member.
    adapter = _adapter()
    projection = adapter.project_analysis_family("kg")
    assert projection.canonical_id == "supervisor.kg"
    assert adapter.restore_analysis_family(projection).value == "kg"
