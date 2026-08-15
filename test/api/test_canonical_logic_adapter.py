"""Regression contracts for supervisor compatibility maps (LPC-090).

The durable generated inventory is
``data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md``.
This module parses every ``supervisor-map`` fence, binds rows to the sealed
catalog root, cross-checks live ``SupervisorCanonicalLogicAdapter@1``
projections, and enforces fail-closed lookup for unknown legacy values.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    AnalysisOperationRegistryError,
    CacheScope,
    LogicFamily,
)
from ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter import (
    SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
    CanonicalLogicAdapterError,
    SupervisorCanonicalLogicAdapter,
    VocabularyProjection,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_translation_validation import (
    LogicForm,
    TranslationClass,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    PropertyKind,
)
from ipfs_datasets_py.logic.families.canonical_catalog import (
    DEFAULT_CANONICAL_CATALOG_SNAPSHOT,
)


# ---------------------------------------------------------------------------
# Errors and constants
# ---------------------------------------------------------------------------


class SupervisorMapError(ValueError):
    """Raised when a supervisor map note entry is missing or unknown."""


CATALOG_ROOT_AUTHORITY: Final = "DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root"
CATALOG_DIGEST_AUTHORITY: Final = "DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest"

REQUIRED_DOMAINS: Final[tuple[str, ...]] = (
    "analysis_family",
    "property_kind",
    "logic_form",
    "translation_class",
    "cache_scope",
    "prover_id",
)

REQUIRED_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "canonical",
        "disposition",
        "residual",
        "deprecation",
        "catalog_root",
    }
)

ALLOWED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "map",
        "map_with_residual",
        "supervisor_extension",
        "provider_alias",
    }
)

ALLOWED_DEPRECATIONS: Final[frozenset[str]] = frozenset(
    {
        "active",
        "legacy_via_adapter",
        "none",
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


# ---------------------------------------------------------------------------
# Note loading / parsing
# ---------------------------------------------------------------------------


def _cutover_note_path() -> Path:
    note_relative = Path(
        "data/agent_supervisor/logic_platform_canonicalization/notes/"
        "supervisor_map_cutover.md"
    )
    for parent in Path(__file__).resolve().parents:
        candidate = parent / note_relative
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / note_relative


def _parse_kv_lines(body: str) -> tuple[dict[str, str], dict[str, str]]:
    meta: dict[str, str] = {}
    labels: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SupervisorMapError(
                f"supervisor-map line must be key: value; got {raw_line!r}"
            )
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {
            "domain",
            "supervisor_enum",
            "catalog_root",
            "fail_closed",
            "task",
            "goal",
            "interface",
            "catalog_snapshot",
            "catalog_digest",
            "hand_maintained_family_lists",
            "unknown_policy",
        }:
            meta[key] = value
        else:
            labels[key] = value
    return meta, labels


def _parse_entry_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in raw.split(";"):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise SupervisorMapError(
                f"map entry fields must be key=value pairs; got {raw!r}"
            )
        field_name, field_value = piece.split("=", 1)
        field_name = field_name.strip()
        field_value = field_value.strip()
        if not field_name or not field_value:
            raise SupervisorMapError(f"empty field in map entry: {raw!r}")
        if field_name in fields:
            raise SupervisorMapError(
                f"duplicate field {field_name!r} in map entry: {raw!r}"
            )
        fields[field_name] = field_value
    missing = REQUIRED_ENTRY_FIELDS - set(fields)
    if missing:
        raise SupervisorMapError(
            f"map entry missing required fields {sorted(missing)}: {raw!r}"
        )
    return fields


def _parse_residual(raw: str) -> Mapping[str, str]:
    residual: dict[str, str] = {}
    if not raw:
        return MappingProxyType(residual)
    for part in raw.split("|"):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise SupervisorMapError(
                f"residual must be k=v pairs joined by |; got {raw!r}"
            )
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SupervisorMapError(f"empty residual field in {raw!r}")
        residual[key] = value
    return MappingProxyType(residual)


def parse_supervisor_map_meta(text: str) -> Mapping[str, str]:
    match = _SUPERVISOR_MAP_META_RE.search(text)
    if match is None:
        raise SupervisorMapError("supervisor-map-meta block missing")
    meta, labels = _parse_kv_lines(match.group(1))
    if labels:
        raise SupervisorMapError(
            f"supervisor-map-meta must not declare legacy labels; got {sorted(labels)!r}"
        )
    required = {
        "task",
        "goal",
        "interface",
        "catalog_snapshot",
        "catalog_root",
        "fail_closed",
    }
    missing = required - set(meta)
    if missing:
        raise SupervisorMapError(
            f"supervisor-map-meta missing fields {sorted(missing)}"
        )
    return MappingProxyType(meta)


def parse_supervisor_map_blocks(text: str) -> dict[str, dict[str, Any]]:
    """Parse every ``supervisor-map`` fence into a domain mapping record."""

    domains: dict[str, dict[str, Any]] = {}
    for match in _SUPERVISOR_MAP_FENCE_RE.finditer(text):
        meta, labels = _parse_kv_lines(match.group(1))
        domain = meta.get("domain")
        if not domain:
            raise SupervisorMapError("supervisor-map block missing domain")
        if domain in domains:
            raise SupervisorMapError(f"duplicate supervisor-map domain {domain!r}")
        if meta.get("fail_closed", "true").lower() != "true":
            raise SupervisorMapError(
                f"domain {domain!r} must declare fail_closed: true"
            )
        if meta.get("catalog_root") != CATALOG_ROOT_AUTHORITY:
            raise SupervisorMapError(
                f"domain {domain!r} catalog_root must be {CATALOG_ROOT_AUTHORITY}"
            )

        entries: dict[str, Mapping[str, Any]] = {}
        for legacy, raw_fields in labels.items():
            fields = _parse_entry_fields(raw_fields)
            disposition = fields["disposition"]
            deprecation = fields["deprecation"]
            if disposition not in ALLOWED_DISPOSITIONS:
                raise SupervisorMapError(
                    f"domain {domain!r} legacy {legacy!r} has unknown "
                    f"disposition {disposition!r}"
                )
            if deprecation not in ALLOWED_DEPRECATIONS:
                raise SupervisorMapError(
                    f"domain {domain!r} legacy {legacy!r} has unknown "
                    f"deprecation {deprecation!r}"
                )
            if fields["catalog_root"] != CATALOG_ROOT_AUTHORITY:
                raise SupervisorMapError(
                    f"domain {domain!r} legacy {legacy!r} catalog_root must be "
                    f"{CATALOG_ROOT_AUTHORITY}"
                )
            residual = _parse_residual(fields["residual"])
            entries[legacy] = MappingProxyType(
                {
                    "legacy": legacy,
                    "canonical_identity": fields["canonical"],
                    "disposition": disposition,
                    "residual": residual,
                    "deprecation": deprecation,
                    "catalog_root": fields["catalog_root"],
                }
            )

        if not entries:
            raise SupervisorMapError(
                f"domain {domain!r} must declare at least one legacy mapping"
            )

        domains[domain] = {
            "domain": domain,
            "supervisor_enum": meta.get("supervisor_enum", ""),
            "fail_closed": True,
            "catalog_root": meta["catalog_root"],
            "entries": MappingProxyType(entries),
        }
    return domains


def load_supervisor_maps(
    note_path: Path | None = None,
) -> tuple[Mapping[str, str], Mapping[str, Mapping[str, Any]]]:
    path = note_path if note_path is not None else _cutover_note_path()
    text = path.read_text(encoding="utf-8")
    meta = parse_supervisor_map_meta(text)
    domains = MappingProxyType(parse_supervisor_map_blocks(text))
    return meta, domains


def map_supervisor_legacy(domain: str, legacy: object) -> Mapping[str, Any]:
    """Map one supervisor legacy value, failing closed on unknowns."""

    _meta, domains = load_supervisor_maps()
    if domain not in domains:
        raise SupervisorMapError(f"unknown supervisor map domain {domain!r}")
    if isinstance(legacy, Enum):
        key = str(legacy.value)
    else:
        key = str(legacy)
    if not key or key != key.strip():
        raise SupervisorMapError(
            f"legacy value must be a non-empty trimmed string; got {legacy!r}"
        )
    entries: Mapping[str, Mapping[str, Any]] = domains[domain]["entries"]
    if key not in entries:
        allowed = ", ".join(sorted(entries))
        raise SupervisorMapError(
            f"unknown legacy {key!r} for domain {domain!r}; allowed: {allowed}"
        )
    return entries[key]


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


def _projectors() -> Mapping[str, Callable[[SupervisorCanonicalLogicAdapter, str], VocabularyProjection]]:
    def analysis(adapter: SupervisorCanonicalLogicAdapter, legacy: str) -> VocabularyProjection:
        return adapter.project_analysis_family(legacy)

    def property_kind(
        adapter: SupervisorCanonicalLogicAdapter, legacy: str
    ) -> VocabularyProjection:
        return adapter.project_property_kind(legacy)

    def logic_form(
        adapter: SupervisorCanonicalLogicAdapter, legacy: str
    ) -> VocabularyProjection:
        return adapter.project_logic_form(legacy)

    def translation_class(
        adapter: SupervisorCanonicalLogicAdapter, legacy: str
    ) -> VocabularyProjection:
        return adapter.project_translation_class(legacy)

    def cache_scope(
        adapter: SupervisorCanonicalLogicAdapter, legacy: str
    ) -> VocabularyProjection:
        return adapter.project_cache_scope(legacy)

    return MappingProxyType(
        {
            "analysis_family": analysis,
            "property_kind": property_kind,
            "logic_form": logic_form,
            "translation_class": translation_class,
            "cache_scope": cache_scope,
        }
    )


# ---------------------------------------------------------------------------
# Note structure and catalog root
# ---------------------------------------------------------------------------


def test_cutover_note_exists_and_declares_required_domains() -> None:
    path = _cutover_note_path()
    assert path.is_file(), f"missing cutover note at {path}"
    meta, domains = load_supervisor_maps(path)
    assert meta["task"] == "LPC-090"
    assert meta["goal"] == "LPC-G090"
    assert meta["interface"] == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    assert meta["catalog_root"] == CATALOG_ROOT_AUTHORITY
    assert meta["fail_closed"].lower() == "true"
    assert meta.get("catalog_digest") == CATALOG_DIGEST_AUTHORITY
    for domain in REQUIRED_DOMAINS:
        assert domain in domains, f"missing domain {domain}"


def test_every_entry_maps_required_fields_and_catalog_root() -> None:
    live_root = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
    live_digest = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
    assert live_root.startswith("b")
    assert live_digest.startswith("sha256:")

    _meta, domains = load_supervisor_maps()
    for domain, record in domains.items():
        assert record["catalog_root"] == CATALOG_ROOT_AUTHORITY
        # Authority resolves to the sealed snapshot root for every domain.
        resolved_root = live_root
        assert resolved_root == live_root
        for legacy, entry in record["entries"].items():
            assert entry["legacy"] == legacy
            assert entry["canonical_identity"]
            assert entry["disposition"] in ALLOWED_DISPOSITIONS
            assert entry["deprecation"] in ALLOWED_DEPRECATIONS
            assert entry["catalog_root"] == CATALOG_ROOT_AUTHORITY
            residual = entry["residual"]
            assert isinstance(residual, Mapping)
            assert residual, f"{domain}/{legacy} residual must be non-empty"
            # Residual always retains exact supervisor identity for reverse map.
            assert residual.get("supervisor_id") == legacy, (
                f"{domain}/{legacy} residual.supervisor_id must equal legacy"
            )
            assert residual.get("domain") == domain, (
                f"{domain}/{legacy} residual.domain must equal domain"
            )
            # Binding is identical across the generated artifact.
            assert entry["catalog_root"] == domains[domain]["catalog_root"]


def test_shared_canonical_ids_keep_distinct_residuals() -> None:
    _meta, domains = load_supervisor_maps()

    families = domains["analysis_family"]["entries"]
    assert families["flogic"]["canonical_identity"] == "frame_logic"
    assert families["frame"]["canonical_identity"] == "frame_logic"
    assert families["flogic"]["disposition"] == "map_with_residual"
    assert families["frame"]["disposition"] == "map_with_residual"
    assert (
        families["flogic"]["residual"]["supervisor_member"]
        != families["frame"]["residual"]["supervisor_member"]
    )

    properties = domains["property_kind"]["entries"]
    assert properties["protocol"]["canonical_identity"] == "trace_conformance"
    assert properties["runtime_trace"]["canonical_identity"] == "trace_conformance"
    assert (
        properties["protocol"]["residual"]["supervisor_member"]
        != properties["runtime_trace"]["residual"]["supervisor_member"]
    )


# ---------------------------------------------------------------------------
# Adapter parity
# ---------------------------------------------------------------------------


def test_note_entries_match_live_adapter_projections() -> None:
    adapter = _adapter()
    _meta, domains = load_supervisor_maps()
    projectors = _projectors()

    for domain, projector in projectors.items():
        record = domains[domain]
        for legacy, entry in record["entries"].items():
            projection = projector(adapter, legacy)
            assert projection.domain == domain
            assert projection.supervisor_id == legacy
            assert projection.canonical_id == entry["canonical_identity"]
            # Note residual fields must be present on the live residual.
            for key, value in entry["residual"].items():
                if key in {"supervisor_prover_id"}:
                    continue
                assert key in projection.residual, (
                    f"{domain}/{legacy} missing residual key {key!r}"
                )
                assert str(projection.residual[key]) == value
            # Live residual always retains supervisor_id / domain.
            assert projection.residual["supervisor_id"] == legacy
            assert projection.residual["domain"] == domain
            # Note residual supervisor_id / domain are the reverse-map keys.
            assert entry["residual"]["supervisor_id"] == legacy
            assert entry["residual"]["domain"] == domain


def test_adapter_inventory_is_fully_covered_by_note() -> None:
    adapter = _adapter()
    inventory = adapter.vocabulary_inventory()
    _meta, domains = load_supervisor_maps()

    expected = {
        "analysis_family": set(inventory["analysis_families"]),
        "property_kind": set(inventory["property_kinds"]),
        "logic_form": set(inventory["logic_forms"]),
        "translation_class": set(inventory["translation_classes"]),
        "cache_scope": set(inventory["cache_scopes"]),
    }
    for domain, expected_ids in expected.items():
        noted = set(domains[domain]["entries"])
        assert noted == expected_ids, (
            f"domain {domain} note/adapter drift: "
            f"only_in_note={sorted(noted - expected_ids)} "
            f"only_in_adapter={sorted(expected_ids - noted)}"
        )


def test_prover_aliases_match_adapter() -> None:
    adapter = _adapter()
    _meta, domains = load_supervisor_maps()
    for legacy, entry in domains["prover_id"]["entries"].items():
        assert (
            adapter.map_prover_id_to_canonical_provider(legacy)
            == entry["canonical_identity"]
        )
        assert entry["disposition"] == "provider_alias"
        residual = entry["residual"]
        assert residual["supervisor_id"] == legacy
        assert residual["domain"] == "prover_id"
        assert residual["supervisor_prover_id"] == legacy
        # Empty / unknown prover tokens fail closed at the note boundary.
    with pytest.raises(SupervisorMapError, match="unknown legacy"):
        map_supervisor_legacy("prover_id", "not-a-mapped-prover-alias")


def test_analysis_family_round_trip_preserves_residual_identity() -> None:
    adapter = _adapter()
    for family in (LogicFamily.FLOGIC, LogicFamily.FRAME):
        projection = adapter.project_analysis_family(family)
        restored = adapter.restore_analysis_family(projection)
        assert restored is family
        note_entry = map_supervisor_legacy("analysis_family", family.value)
        assert note_entry["canonical_identity"] == projection.canonical_id


def test_property_kind_round_trip_preserves_residual_identity() -> None:
    adapter = _adapter()
    for kind in (PropertyKind.PROTOCOL, PropertyKind.RUNTIME_TRACE):
        projection = adapter.project_property_kind(kind)
        restored = adapter.restore_property_kind(projection)
        assert restored is kind
        note_entry = map_supervisor_legacy("property_kind", kind.value)
        assert note_entry["canonical_identity"] == projection.canonical_id


def test_logic_form_and_translation_class_and_cache_round_trips() -> None:
    adapter = _adapter()
    for form in LogicForm:
        projection = adapter.project_logic_form(form)
        assert adapter.restore_logic_form(projection) is form
        assert (
            map_supervisor_legacy("logic_form", form.value)["canonical_identity"]
            == projection.canonical_id
        )
    for translation in TranslationClass:
        projection = adapter.project_translation_class(translation)
        assert adapter.restore_translation_class(projection) is translation
        note = map_supervisor_legacy("translation_class", translation.value)
        assert note["canonical_identity"] == projection.canonical_id
        assert (
            note["residual"]["taxonomy_translation_kind"]
            == projection.residual["taxonomy_translation_kind"]
        )
    for scope in CacheScope:
        # CacheScope has alias members that share values; project by value once.
        projection = adapter.project_cache_scope(scope.value)
        assert adapter.restore_cache_scope(projection).value == scope.value
        assert (
            map_supervisor_legacy("cache_scope", scope.value)["canonical_identity"]
            == projection.canonical_id
        )


# ---------------------------------------------------------------------------
# Fail-closed unknown values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain",
    (
        "analysis_family",
        "property_kind",
        "logic_form",
        "translation_class",
        "cache_scope",
        "prover_id",
    ),
)
def test_note_lookup_unknown_legacy_fails_closed(domain: str) -> None:
    with pytest.raises(SupervisorMapError, match="unknown legacy"):
        map_supervisor_legacy(domain, "__not_a_supervisor_legacy__")


def test_note_lookup_unknown_domain_fails_closed() -> None:
    with pytest.raises(SupervisorMapError, match="unknown supervisor map domain"):
        map_supervisor_legacy("not_a_domain", "tdfol")


def test_adapter_unknown_tokens_fail_closed() -> None:
    adapter = _adapter()
    # normalize_logic_family fails closed before the projection residual is built.
    with pytest.raises(
        (CanonicalLogicAdapterError, AnalysisOperationRegistryError)
    ):
        adapter.project_analysis_family("not-a-family")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_property_kind("not-a-property-kind")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_logic_form("not-a-form")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_translation_class("not-a-class")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_cache_scope("not-a-scope")


def test_empty_legacy_token_fails_closed() -> None:
    with pytest.raises(SupervisorMapError):
        map_supervisor_legacy("analysis_family", "")
    with pytest.raises(SupervisorMapError):
        map_supervisor_legacy("analysis_family", "  ")


def test_cross_domain_label_does_not_silently_map() -> None:
    # A valid analysis family label is not a valid translation class.
    with pytest.raises(SupervisorMapError):
        map_supervisor_legacy("translation_class", "tdfol")
    adapter = _adapter()
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_translation_class("tdfol")


# ---------------------------------------------------------------------------
# Catalog / generated projection relationship
# ---------------------------------------------------------------------------


def test_mapped_family_canonical_ids_exist_in_catalog_or_supervisor_namespace() -> None:
    taxonomy_ids = set(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.taxonomy.families)
    _meta, domains = load_supervisor_maps()
    for legacy, entry in domains["analysis_family"]["entries"].items():
        canonical = entry["canonical_identity"]
        if entry["disposition"] == "supervisor_extension":
            assert canonical.startswith("supervisor."), legacy
            continue
        assert canonical in taxonomy_ids, (
            f"analysis family {legacy!r} maps to {canonical!r} absent from taxonomy"
        )


def test_map_with_residual_disposition_used_when_canonical_is_shared() -> None:
    _meta, domains = load_supervisor_maps()
    for domain in ("analysis_family", "property_kind"):
        by_canonical: dict[str, list[str]] = {}
        for legacy, entry in domains[domain]["entries"].items():
            by_canonical.setdefault(entry["canonical_identity"], []).append(legacy)
        for canonical, legacies in by_canonical.items():
            if len(legacies) < 2:
                continue
            for legacy in legacies:
                entry = domains[domain]["entries"][legacy]
                assert entry["disposition"] == "map_with_residual", (
                    f"{domain}/{legacy} shares {canonical} but disposition is "
                    f"{entry['disposition']!r}"
                )


def test_adapter_interface_stable_for_generated_maps() -> None:
    adapter = _adapter()
    assert adapter.interface == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    inventory = adapter.vocabulary_inventory()
    assert "analysis_family" in inventory["domains"]
    # Enum surfaces used by the generated maps remain non-empty closed sets.
    assert _unique_enum_values(LogicFamily)
    assert _unique_enum_values(PropertyKind)
    assert _unique_enum_values(LogicForm)
    assert _unique_enum_values(TranslationClass)
    assert _unique_enum_values(CacheScope)


def test_vocabulary_projection_retains_residual_for_reverse_mapping() -> None:
    projection = VocabularyProjection(
        domain="analysis_family",
        supervisor_id="flogic",
        canonical_id="frame_logic",
        residual={
            "supervisor_enum": "LogicFamily",
            "supervisor_member": "FLOGIC",
        },
    )
    assert projection.residual["supervisor_id"] == "flogic"
    assert projection.residual["domain"] == "analysis_family"
    restored = VocabularyProjection.from_dict(projection.to_dict())
    assert restored.canonical_id == "frame_logic"
    assert restored.residual["supervisor_member"] == "FLOGIC"


def test_note_lookup_returns_full_acceptance_fields() -> None:
    """Acceptance: identity, disposition, residual, deprecation, catalog root."""

    entry = map_supervisor_legacy("analysis_family", "flogic")
    assert entry["canonical_identity"] == "frame_logic"
    assert entry["disposition"] == "map_with_residual"
    assert entry["deprecation"] == "active"
    assert entry["catalog_root"] == CATALOG_ROOT_AUTHORITY
    residual = entry["residual"]
    assert residual["supervisor_id"] == "flogic"
    assert residual["domain"] == "analysis_family"
    assert residual["supervisor_enum"] == "LogicFamily"
    assert residual["supervisor_member"] == "FLOGIC"

    kg = map_supervisor_legacy("analysis_family", LogicFamily.KNOWLEDGE_GRAPH)
    assert kg["canonical_identity"] == "supervisor.kg"
    assert kg["disposition"] == "supervisor_extension"
    assert kg["deprecation"] == "legacy_via_adapter"
    assert kg["residual"]["supervisor_member"] == "KNOWLEDGE_GRAPH"

    coq = map_supervisor_legacy("prover_id", "coq")
    assert coq["canonical_identity"] == "rocq"
    assert coq["disposition"] == "provider_alias"
    assert coq["catalog_root"] == CATALOG_ROOT_AUTHORITY
