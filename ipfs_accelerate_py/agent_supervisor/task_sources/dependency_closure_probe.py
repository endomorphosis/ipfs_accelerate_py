"""Measure a declared dependency frontier without sealing a legal theory.

The dependency owner keeps a missing target as an open edge and refuses an
exhaustive inventory. The lookup owner keeps two competing definitions
ambiguous and invalidates the former unique resolution. Duplicate identities
are refused instead of merged. A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json

from .dependency_block_resolution import EXPECTED_DECLARED_CLOSURE_PROBE

_TERM = "covered filer"
_DOCUMENT = "doc-A"
_METADATA = {"interpretation_status": "unreviewed_caller_declaration"}


def measure_dependency_closure_probe() -> dict[str, object] | None:
    """Return the fixed declared-closure contract, or None when owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_DECLARED_CLOSURE_PROBE:
        return None
    return measured


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wire(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _cid(label: str) -> str:
    from ipfs_datasets_py.utils.cid_utils import cid_for_dag_json

    return cid_for_dag_json({"authored_fixture": label})


def _dependencies():
    from ipfs_datasets_py.logic.autoformal import legal_graph_dependencies

    return legal_graph_dependencies


def _lookup():
    from ipfs_datasets_py.logic.autoformal import scoped_definition_resolution

    return scoped_definition_resolution


def _symbols():
    from ipfs_datasets_py.logic.integration.reasoning import legal_ir_symbols

    return legal_ir_symbols


def _artifact(subject, key: str, **pins: str | None) -> dict:
    declared = {kind: [] for kind in subject.PIN_KINDS}
    declared.update(
        {
            kind: [{"key": f"{kind}:{key}", "sha256": value}]
            for kind, value in pins.items()
        }
    )
    return {
        "artifact_id": key,
        "artifact_sha256": "a" * 64,
        "dependency_pins": declared,
    }


def _observations(subject, manifest: dict) -> dict:
    pins = {kind: {} for kind in subject.PIN_KINDS}
    for row in manifest["artifacts"]:
        for kind, values in row["dependency_pins"].items():
            pins[kind].update({pin["key"]: pin["sha256"] for pin in values})
    return {
        "expected_manifest_sha256": manifest["manifest_sha256"],
        "observed_artifact_sha256s": {
            row["artifact_id"]: row["artifact_sha256"] for row in manifest["artifacts"]
        },
        "observed_pins": pins,
    }


def _table(definitions: tuple[tuple[str, str], ...]) -> dict:
    symbols = _symbols()
    builder = symbols.LegalIRSymbolTableBuilder(
        symbol_table_id="authored-table-" + _DOCUMENT
    )
    scope = builder.add_document_scope(_DOCUMENT)
    for identity, name in definitions:
        builder.add_definition(
            name,
            symbols.LegalIRSymbolKind.DEFINED_TERM,
            symbol_id=identity,
            scope_id=scope.scope_id,
            document_id=_DOCUMENT,
            metadata=dict(_METADATA),
        )
    return builder.to_symbol_table().to_dict()


def _partition(identity: str, definitions: tuple[tuple[str, str], ...]) -> dict:
    raw = _wire(_table(definitions))
    return {
        "partition_id": identity,
        "release_id": "authored-release-2024",
        "edition": 2024,
        "document_id": _DOCUMENT,
        "source_version_cid": _cid("source-" + _DOCUMENT + "-2024"),
        "scope_cid": _cid("scope-" + _DOCUMENT + "-2024"),
        "applicability_cid": _cid("declared-applicability"),
        "table_raw": raw,
    }


def _registry(*parts: dict) -> dict:
    expected = [
        {"partition_id": row["partition_id"], "table_sha256": _sha(row["table_raw"])}
        for row in parts
    ]
    return _lookup().build_definition_lookup_registry(list(parts), expected)


def _request() -> dict:
    symbols = _symbols()
    reference = symbols.LegalIRSymbolReference(
        reference_id="authored-reference-A",
        name=_TERM,
        symbol_kind=symbols.LegalIRSymbolKind.DEFINED_TERM,
        scope_id="document:" + _DOCUMENT,
        document_id=_DOCUMENT,
        explicit_target_document_id="",
        explicit_scope_id="",
        source_node_ids=("authored-use-node",),
        span_ids=("authored-use-span",),
        metadata=dict(_METADATA),
    ).to_dict()
    return {
        "reference": reference,
        "partition_ids": ["local"],
        "source_occurrence_cid": _cid("source-use-occurrence"),
        "interpretation_cid": _cid("unreviewed-interpretation"),
    }


def _resolve(registry: dict, request: dict) -> dict:
    raw = _wire(request)
    return _lookup().resolve_definition_lookup(registry, raw, _sha(raw))


def _refuse_authority(result: dict) -> None:
    if result["legal_dependency_inventory_exhaustive"] is not False:
        raise RuntimeError("dependency inventory claimed to be exhaustive")
    if result["proof_authority"] is not False or result["admitted"] is not False:
        raise RuntimeError("dependency invalidation admitted a result")
    if result["source_semantics_verified"] is not False:
        raise RuntimeError("dependency invalidation verified source semantics")
    if result["global_consistency_verified"] is not False:
        raise RuntimeError("dependency invalidation verified global consistency")


def _frontier(subject) -> dict:
    rows = [
        _artifact(subject, "rule", context=None),
        _artifact(subject, "doctrine"),
    ]
    manifest = subject.build_dependency_manifest(
        rows,
        [
            {"dependent_id": "rule", "dependency_id": "missing-source"},
            {"dependent_id": "doctrine", "dependency_id": "rule"},
        ],
        expected_artifact_ids=["rule", "doctrine"],
    )
    result = subject.invalidate_dependencies(manifest, **_observations(subject, manifest))
    _refuse_authority(result)
    if result["schema"] != "legal-graph-reverse-invalidation/v1":
        raise RuntimeError("invalidation schema changed")
    if manifest["schema"] != "legal-graph-dependency-manifest/v1":
        raise RuntimeError("dependency manifest schema changed")
    if result["stale_artifact_ids"] != ["rule", "doctrine"]:
        raise RuntimeError("missing target did not keep the declared frontier")
    if result["missing_dependencies"] != [
        {"dependent_id": "rule", "dependency_id": "missing-source"}
    ]:
        raise RuntimeError("missing target was dropped or rewritten")
    if result["population_complete"] is not True:
        raise RuntimeError("declared population was not reported complete")
    if result["coverage_scope"] != "exact_declared_artifacts_pins_and_edges_only":
        raise RuntimeError("coverage scope widened past the declared population")
    return result


def _competing(subject) -> dict[str, object]:
    lookup = _lookup()
    first = _partition("local", (("term:first", _TERM),))
    original = _registry(first)
    request = _request()
    resolved = _resolve(original, request)
    changed = _partition(
        "local",
        (("term:first", _TERM), ("term:new", _TERM)),
    )
    ambiguous = _resolve(_registry(changed), request)
    if resolved["status"] != "resolved" or resolved["selected_symbol_id"] != "term:first":
        raise RuntimeError("former unique resolution was not recorded")
    symbol_ids = sorted(ambiguous["owner_resolution"]["symbol_ids"])
    if (
        ambiguous["status"] != "ambiguous"
        or ambiguous["selected_symbol_id"] is not None
        or symbol_ids != ["term:first", "term:new"]
    ):
        raise RuntimeError("competing definitions were unioned or dropped")
    manifest = lookup.build_definition_lookup_dependencies(
        original,
        [resolved],
        [resolved["record_cid"]],
    )
    observed = _observations(subject, manifest)
    observed["observed_pins"]["definition"]["definition-table:local"] = _sha(
        changed["table_raw"]
    )
    invalidation = subject.invalidate_dependencies(manifest, **observed)
    _refuse_authority(invalidation)
    stale = invalidation["stale_artifact_ids"]
    if stale != [
        "definition-partition:local",
        "definition-lookup:" + resolved["record_cid"],
    ]:
        raise RuntimeError("competing definition did not stale the former resolution")
    return {
        "candidate_ids": symbol_ids,
        "former_lookup_stale": True,
        "former_partition_stale": True,
        "former_selected_symbol_id": "term:first",
        "former_status": "resolved",
        "revised_selection_absent": True,
        "revised_status": "ambiguous",
        "stale_count": 2,
    }


def _duplicate_refused() -> None:
    try:
        _registry(
            _partition("local", (("term:first", _TERM),)),
            _partition("other", (("term:first", _TERM),)),
        )
    except Exception as exc:
        if "globally duplicate definition or alias identity" not in str(exc):
            raise RuntimeError("duplicate identity failed for another reason") from exc
        return
    raise RuntimeError("duplicate definition identity was accepted")


def _measure() -> dict[str, object]:
    subject = _dependencies()
    frontier = _frontier(subject)
    competing = _competing(subject)
    _duplicate_refused()
    return {
        "admitted": False,
        "candidate_ids": competing["candidate_ids"],
        "closure_is_exhaustive_legal_theory": False,
        "coverage_scope": frontier["coverage_scope"],
        "duplicate_identity_refused": True,
        "former_lookup_stale": True,
        "former_partition_stale": True,
        "former_selected_symbol_id": competing["former_selected_symbol_id"],
        "former_status": competing["former_status"],
        "frontier_stale_artifact_ids": list(frontier["stale_artifact_ids"]),
        "global_consistency_verified": False,
        "interpretations_unioned": False,
        "invalidation_schema": frontier["schema"],
        "legal_dependency_inventory_exhaustive": False,
        "manifest_schema": "legal-graph-dependency-manifest/v1",
        "missing_dependencies": [
            {
                "dependency_id": "missing-source",
                "dependent_id": "rule",
            }
        ],
        "missing_dependency_retained": True,
        "population_complete": True,
        "proof_authority": False,
        "revised_selection_absent": True,
        "revised_status": competing["revised_status"],
        "source_semantics_verified": False,
        "stale_count": competing["stale_count"],
    }
