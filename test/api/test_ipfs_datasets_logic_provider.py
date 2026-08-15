"""LPC-120 regression: Hammer adapter vocabularies derived from the catalog.

The durable generated artifact is
``data/agent_supervisor/logic_platform_canonicalization/notes/hammer_adapter.md``.
This module parses every ``hammer-vocab`` fence, expands the catalog-root
binding against ``DEFAULT_CANONICAL_CATALOG_SNAPSHOT``, derives the same
vocabulary sets from the live catalog, and verifies the live Hammer adapter
exports only catalog-projected identities (plus documented residual wire
aliases). Semantic separations required by LPC-G120 are enforced.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    LogicFamily,
)
from ipfs_accelerate_py.agent_supervisor.integrations import (
    ipfs_datasets_logic_provider as hammer_adapter,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    IPFS_DATASETS_LOGIC_PROVIDER_ID,
    KNOWN_HAMMER_SOLVERS,
    SUPPORTED_LOGIC_FAMILIES,
    SUPPORTED_TRANSLATION_FAMILIES,
    HammerSupervisorPolicy,
    IpfsDatasetsLogicProvider,
    normalize_registry_logic_family,
    to_canonical_registry_logic_family,
)
from ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter import (
    map_analysis_family_to_canonical,
    map_prover_id_to_canonical_provider,
)
from ipfs_datasets_py.logic.families.canonical_catalog import (
    DEFAULT_CANONICAL_CATALOG_SNAPSHOT,
)
from ipfs_datasets_py.logic.families.providers import (
    ADVISORY_AUTHORITY_CEILINGS,
    ADVISORY_PROVIDER_IDS,
)


# ---------------------------------------------------------------------------
# Errors / constants
# ---------------------------------------------------------------------------


class HammerAdapterVocabError(ValueError):
    """Raised when a Hammer wire value cannot be mapped fail-closed."""


REQUIRED_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "canonical_identity",
        "axis",
        "disposition",
        "residual",
        "catalog_root",
    }
)

REQUIRED_VOCAB_DOMAINS: Final[tuple[str, ...]] = (
    "logic_family",
    "translation_family",
    "translation_alias",
    "solver_provider",
    "solver_alias",
    "target_itp",
    "family_target",
    "authority_ceiling",
    "evidence_kind",
    "semantic_separation",
)

VOCAB_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "map",
        "residual_alias",
        "residual_collapse",
        "ceiling",
    }
)

VOCAB_AXES: Final[frozenset[str]] = frozenset(
    {
        "family",
        "encoding",
        "notation",
        "provider",
        "lane",
        "evidence_kind",
        "evidence_authority",
    }
)

_HAMMER_VOCAB_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```hammer-vocab\n(.*?)\n```",
    re.DOTALL,
)

_HAMMER_VOCAB_META_RE: Final[re.Pattern[str]] = re.compile(
    r"```hammer-vocab-meta\n(.*?)\n```",
    re.DOTALL,
)

_META_KEYS: Final[frozenset[str]] = frozenset(
    {
        "domain",
        "surface",
        "fail_closed",
        "catalog_root",
    }
)

VOCAB_NOTE_RELATIVE: Final[Path] = Path(
    "data/agent_supervisor/logic_platform_canonicalization/notes/"
    "hammer_adapter.md"
)

ADAPTER_MODULE_RELATIVE: Final[Path] = Path(
    "ipfs_accelerate_py/agent_supervisor/integrations/"
    "ipfs_datasets_logic_provider.py"
)

# Residual module-level names that may hold closed wire sets. They are dual-read
# residuals documented in the note, not independent free-form inventories.
DOCUMENTED_RESIDUAL_CONSTANTS: Final[frozenset[str]] = frozenset(
    {
        "SUPPORTED_LOGIC_FAMILIES",
        "SUPPORTED_TRANSLATION_FAMILIES",
        "KNOWN_HAMMER_SOLVERS",
        "_FAMILY_ALIASES",
        "_FAMILY_ITP",
        "_FAMILY_TARGET",
        "_SOLVER_ALIASES",
    }
)


# ---------------------------------------------------------------------------
# Note loading / parsing
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / VOCAB_NOTE_RELATIVE).is_file() and (
            parent / "ipfs_accelerate_py"
        ).is_dir():
            return parent
    return here.parents[2]


def vocab_note_path() -> Path:
    return _repo_root() / VOCAB_NOTE_RELATIVE


def adapter_source_path() -> Path:
    return _repo_root() / ADAPTER_MODULE_RELATIVE


def sealed_catalog_root() -> str:
    root = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
    if not isinstance(root, str) or not root:
        raise HammerAdapterVocabError("sealed catalog root is missing")
    return root


def sealed_catalog_digest() -> str:
    digest = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise HammerAdapterVocabError("sealed catalog digest is missing")
    return digest


def _expand_catalog_root(raw: str, *, live_root: str) -> str:
    token = raw.strip()
    if token in {
        "$catalog_root_binding",
        "DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root",
    }:
        return live_root
    if not token:
        raise HammerAdapterVocabError("catalog_root must be non-empty")
    return token


def _parse_residual(raw: str) -> Mapping[str, str]:
    residual: dict[str, str] = {}
    for part in raw.split("|"):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise HammerAdapterVocabError(
                f"residual must be k=v pairs joined by |; got {raw!r}"
            )
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise HammerAdapterVocabError(f"empty residual pair in {raw!r}")
        residual[key] = value
    if not residual:
        raise HammerAdapterVocabError(f"empty residual: {raw!r}")
    return MappingProxyType(residual)


def _parse_row_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in raw.split(";"):
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise HammerAdapterVocabError(
                f"row field must be key=value; got {piece!r} in {raw!r}"
            )
        key, value = piece.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise HammerAdapterVocabError(f"empty field in row {raw!r}")
        if key in fields:
            raise HammerAdapterVocabError(f"duplicate field {key!r} in row {raw!r}")
        fields[key] = value
    return fields


def parse_hammer_vocab_meta(text: str) -> Mapping[str, str]:
    match = _HAMMER_VOCAB_META_RE.search(text)
    if match is None:
        raise HammerAdapterVocabError("missing hammer-vocab-meta fence")
    meta: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise HammerAdapterVocabError(
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
        "candidate_producing",
        "candidate_authoritative",
        "authority_ceiling",
    }
    missing = required - set(meta)
    if missing:
        raise HammerAdapterVocabError(f"hammer-vocab-meta missing {sorted(missing)}")
    if meta["task"] != "LPC-120":
        raise HammerAdapterVocabError(f"unexpected task id {meta['task']!r}")
    if meta["goal"] != "LPC-G120":
        raise HammerAdapterVocabError(f"unexpected goal id {meta['goal']!r}")
    if meta["adapter_interface"] != "DatasetsLogicProvider@1":
        raise HammerAdapterVocabError(
            f"adapter interface mismatch: {meta['adapter_interface']!r}"
        )
    if meta["catalog_interface"] != "CanonicalLogicCatalogSnapshot@1":
        raise HammerAdapterVocabError(
            f"catalog interface mismatch: {meta['catalog_interface']!r}"
        )
    if meta["fail_closed"].lower() != "true":
        raise HammerAdapterVocabError("vocab must declare fail_closed: true")
    if meta["candidate_producing"].lower() != "true":
        raise HammerAdapterVocabError("Hammer must remain candidate-producing")
    if meta["candidate_authoritative"].lower() != "false":
        raise HammerAdapterVocabError(
            "Hammer candidate_authoritative must be false"
        )
    if meta["authority_ceiling"] != "advisory":
        raise HammerAdapterVocabError(
            f"authority_ceiling must be advisory; got {meta['authority_ceiling']!r}"
        )
    return MappingProxyType(meta)


def parse_hammer_vocab_blocks(
    text: str,
    *,
    live_catalog_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse every ``hammer-vocab`` fence into domain records."""

    live_root = live_catalog_root or sealed_catalog_root()
    domains: dict[str, dict[str, Any]] = {}

    for match in _HAMMER_VOCAB_FENCE_RE.finditer(text):
        body = match.group(1)
        meta: dict[str, str] = {}
        labels: dict[str, dict[str, Any]] = {}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise HammerAdapterVocabError(
                    f"hammer-vocab line must be key: value; got {raw_line!r}"
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
                raise HammerAdapterVocabError(
                    f"wire value {key!r} missing fields {sorted(missing)}"
                )
            disposition = fields["disposition"]
            if disposition not in VOCAB_DISPOSITIONS:
                raise HammerAdapterVocabError(
                    f"unknown disposition {disposition!r} for {key!r}"
                )
            axis = fields["axis"]
            if axis not in VOCAB_AXES:
                raise HammerAdapterVocabError(
                    f"unknown axis {axis!r} for {key!r}"
                )
            catalog_root = _expand_catalog_root(
                fields["catalog_root"], live_root=live_root
            )
            if catalog_root != live_root:
                raise HammerAdapterVocabError(
                    f"catalog_root for {key!r} does not match sealed snapshot "
                    f"({catalog_root!r} != {live_root!r})"
                )
            residual = _parse_residual(fields["residual"])
            labels[key] = {
                "wire_value": key,
                "canonical_identity": fields["canonical_identity"],
                "axis": axis,
                "disposition": disposition,
                "residual": residual,
                "catalog_root": catalog_root,
            }

        domain = meta.get("domain")
        surface = meta.get("surface")
        if not domain or not surface:
            raise HammerAdapterVocabError(
                "hammer-vocab block requires domain and surface"
            )
        fail_closed = meta.get("fail_closed", "true").lower() == "true"
        if not fail_closed:
            raise HammerAdapterVocabError(
                f"surface {surface!r} must declare fail_closed: true"
            )
        block_catalog_root = _expand_catalog_root(
            meta.get("catalog_root", "$catalog_root_binding"),
            live_root=live_root,
        )
        if block_catalog_root != live_root:
            raise HammerAdapterVocabError(
                f"block catalog_root for {surface!r} does not match sealed snapshot"
            )
        if not labels:
            raise HammerAdapterVocabError(
                f"surface {surface!r} requires at least one label"
            )
        if domain in domains:
            raise HammerAdapterVocabError(
                f"duplicate hammer-vocab domain {domain!r}"
            )
        domains[domain] = {
            "domain": domain,
            "surface": surface,
            "fail_closed": True,
            "catalog_root": block_catalog_root,
            "labels": MappingProxyType(labels),
        }
    return domains


def load_hammer_vocabs(
    note_path: Path | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    path = note_path if note_path is not None else vocab_note_path()
    text = path.read_text(encoding="utf-8")
    parse_hammer_vocab_meta(text)
    return MappingProxyType(parse_hammer_vocab_blocks(text))


def map_hammer_wire(domain: str, label: object) -> Mapping[str, Any]:
    """Fail-closed lookup of one wire value against the vocabulary artifact."""

    maps = load_hammer_vocabs()
    if domain not in maps:
        raise HammerAdapterVocabError(f"unknown hammer vocab domain {domain!r}")
    record = maps[domain]
    if not isinstance(label, str) or not label or label != label.strip():
        raise HammerAdapterVocabError(
            f"wire label must be a non-empty trimmed string; got {label!r}"
        )
    labels: Mapping[str, Mapping[str, Any]] = record["labels"]
    if label not in labels:
        allowed = ", ".join(sorted(labels))
        raise HammerAdapterVocabError(
            f"unknown label {label!r} for domain {domain!r}; allowed: {allowed}"
        )
    return labels[label]


# ---------------------------------------------------------------------------
# Catalog-derived projections (authoritative derivation helpers)
# ---------------------------------------------------------------------------


def catalog_family_ids() -> frozenset[str]:
    return frozenset(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.family_ids)


def catalog_encoding_ids() -> frozenset[str]:
    return frozenset(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.encodings)


def catalog_notation_ids() -> frozenset[str]:
    return frozenset(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.notations)


def catalog_provider_ids() -> frozenset[str]:
    return frozenset(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.provider_ids)


def catalog_lane_ids() -> frozenset[str]:
    return frozenset(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.lanes)


def catalog_evidence_ids() -> frozenset[str]:
    return frozenset(DEFAULT_CANONICAL_CATALOG_SNAPSHOT.evidence)


def catalog_provider_aliases() -> Mapping[str, str]:
    """Reviewed aliases from the generated provider projection."""

    generated = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.generated
    aliases = getattr(generated, "reviewed_aliases", None)
    if aliases is None:
        return MappingProxyType({})
    return MappingProxyType(dict(aliases))


def derive_translation_family_canonical(wire: str) -> str:
    """Project a Hammer translation-family wire id onto a catalog identity."""

    return map_hammer_wire("translation_family", wire)["canonical_identity"]


def derive_solver_canonical(wire: str) -> str:
    """Project a Hammer solver wire id onto a catalog provider identity."""

    row = map_hammer_wire("solver_provider", wire)
    return str(row["canonical_identity"])


def _axis_membership(axis: str, identity: str) -> bool:
    """Return True when *identity* is admitted on *axis* by the sealed catalog."""

    if identity.startswith("supervisor."):
        # Supervisor-namespaced extensions are residual and not baseline taxonomy.
        return axis == "family"
    if axis == "family":
        return identity in catalog_family_ids()
    if axis == "encoding":
        return identity in catalog_encoding_ids()
    if axis == "notation":
        return identity in catalog_notation_ids()
    if axis == "provider":
        providers = catalog_provider_ids()
        aliases = catalog_provider_aliases()
        return identity in providers or identity in aliases.values()
    if axis == "lane":
        return identity in catalog_lane_ids()
    if axis == "evidence_kind":
        return identity in catalog_evidence_ids()
    if axis == "evidence_authority":
        return identity in {
            "none",
            "advisory",
            "bounded",
            "independently_checkable",
            "authoritative",
            "unknown",
        }
    return False


def _module_level_string_literals(source: str) -> dict[str, frozenset[str]]:
    """Collect string literals assigned at module level for residual constants."""

    tree = ast.parse(source)
    found: dict[str, set[str]] = {}
    for node in tree.body:
        targets: list[str] = []
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets.append(node.target.id)
            value_node = node.value
        if value_node is None:
            continue
        for name in targets:
            if name not in DOCUMENTED_RESIDUAL_CONSTANTS:
                continue
            literals: set[str] = set()
            for child in ast.walk(value_node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    literals.add(child.value)
            found[name] = literals
    return {name: frozenset(values) for name, values in found.items()}


# ---------------------------------------------------------------------------
# Structural / catalog-root tests
# ---------------------------------------------------------------------------


def test_vocab_note_exists_and_declares_lpc_120() -> None:
    path = vocab_note_path()
    assert path.is_file(), f"missing hammer adapter artifact: {path}"
    text = path.read_text(encoding="utf-8")
    assert "LPC-120" in text
    assert "DatasetsLogicProvider@1" in text
    assert "CanonicalLogicCatalogSnapshot@1" in text
    meta = parse_hammer_vocab_meta(text)
    assert meta["catalog_root_binding"] == (
        "DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root"
    )
    assert meta["provider_id"] == "hammer"
    assert meta["authority_ceiling"] == "advisory"


def test_every_required_domain_is_present() -> None:
    maps = load_hammer_vocabs()
    missing = set(REQUIRED_VOCAB_DOMAINS) - set(maps)
    assert not missing, f"missing hammer-vocab domains: {sorted(missing)}"


def test_every_mapped_row_has_required_fields_and_catalog_root() -> None:
    live_root = sealed_catalog_root()
    maps = load_hammer_vocabs()
    row_count = 0
    for domain, record in maps.items():
        labels = record["labels"]
        assert labels, f"domain {domain} has no labels"
        assert record["catalog_root"] == live_root
        assert record["fail_closed"] is True
        for wire, row in labels.items():
            row_count += 1
            for field in REQUIRED_ROW_FIELDS:
                assert field in row, f"{domain}/{wire} missing {field}"
            assert row["catalog_root"] == live_root
            assert row["disposition"] in VOCAB_DISPOSITIONS
            assert row["axis"] in VOCAB_AXES
            assert row["canonical_identity"]
            assert row["residual"]
    assert row_count >= 30, f"expected exhaustive rows, found {row_count}"


def test_catalog_root_matches_sealed_snapshot_identity() -> None:
    live_root = sealed_catalog_root()
    live_digest = sealed_catalog_digest()
    assert live_root.startswith("b")
    assert live_digest.startswith("sha256:")
    rebuilt = DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_identity()
    assert rebuilt.cid == live_root
    assert rebuilt.digest == live_digest


# ---------------------------------------------------------------------------
# Fail-closed lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", REQUIRED_VOCAB_DOMAINS)
def test_unknown_wire_values_fail_closed(domain: str) -> None:
    with pytest.raises(HammerAdapterVocabError):
        map_hammer_wire(domain, "__not_a_hammer_wire_value__")
    with pytest.raises(HammerAdapterVocabError):
        map_hammer_wire(domain, "")


# ---------------------------------------------------------------------------
# Catalog membership of canonical identities
# ---------------------------------------------------------------------------


def test_canonical_identities_are_catalog_admitted() -> None:
    maps = load_hammer_vocabs()
    for domain, record in maps.items():
        if domain == "semantic_separation":
            # Invariant rows may name multi-axis pairs; still require axis membership
            # of the primary canonical_identity where applicable.
            pass
        for wire, row in record["labels"].items():
            identity = row["canonical_identity"]
            axis = row["axis"]
            if domain == "semantic_separation" and axis == "family":
                # Shared canonical may be frame_logic while the row names a pair.
                assert _axis_membership(axis, identity), (
                    f"{domain}/{wire}: {identity!r} not admitted on axis {axis}"
                )
                continue
            assert _axis_membership(axis, identity), (
                f"{domain}/{wire}: canonical {identity!r} not admitted on "
                f"axis {axis} under sealed catalog"
            )


def test_no_hand_maintained_authority_above_advisory_ceiling() -> None:
    maps = load_hammer_vocabs()
    authority_rows = maps["authority_ceiling"]["labels"]
    for wire, row in authority_rows.items():
        assert row["axis"] == "evidence_authority"
        assert row["disposition"] == "ceiling"
        assert row["canonical_identity"] == "advisory", (
            f"authority row {wire!r} must ceiling at advisory, got "
            f"{row['canonical_identity']!r}"
        )
    # Catalog hard ceiling for the hammer provider id.
    assert "hammer" in ADVISORY_PROVIDER_IDS
    ceiling = ADVISORY_AUTHORITY_CEILINGS["hammer"]
    assert getattr(ceiling, "value", str(ceiling)) == "advisory"


# ---------------------------------------------------------------------------
# Live adapter consistency with catalog-derived vocabularies
# ---------------------------------------------------------------------------


def test_provider_id_is_catalog_hammer() -> None:
    assert IPFS_DATASETS_LOGIC_PROVIDER_ID == "hammer"
    assert "hammer" in catalog_provider_ids() or "hammer" in ADVISORY_PROVIDER_IDS


def test_supported_logic_families_are_enum_derived_not_free_form() -> None:
    maps = load_hammer_vocabs()
    labels = maps["logic_family"]["labels"]
    # Enum-derived (including Enum aliases collapsed by iteration), not a
    # free-form string inventory.
    enum_values = tuple(item.value for item in LogicFamily)
    assert SUPPORTED_LOGIC_FAMILIES == enum_values
    assert set(SUPPORTED_LOGIC_FAMILIES) == set(labels)
    for wire in SUPPORTED_LOGIC_FAMILIES:
        row = labels[wire]
        projected = map_analysis_family_to_canonical(wire)
        assert projected == row["canonical_identity"]
        assert to_canonical_registry_logic_family(wire) == projected


def test_supported_translation_families_match_vocab_and_catalog() -> None:
    maps = load_hammer_vocabs()
    labels = maps["translation_family"]["labels"]
    assert set(SUPPORTED_TRANSLATION_FAMILIES) == set(labels)
    for wire in SUPPORTED_TRANSLATION_FAMILIES:
        identity = derive_translation_family_canonical(wire)
        axis = labels[wire]["axis"]
        assert _axis_membership(axis, identity), (
            f"translation family {wire!r} → {identity!r} not on axis {axis}"
        )


def test_known_hammer_solvers_match_vocab_and_catalog_providers() -> None:
    maps = load_hammer_vocabs()
    labels = maps["solver_provider"]["labels"]
    assert set(KNOWN_HAMMER_SOLVERS) == set(labels)
    providers = catalog_provider_ids()
    aliases = catalog_provider_aliases()
    for wire in KNOWN_HAMMER_SOLVERS:
        identity = derive_solver_canonical(wire)
        assert identity in providers or identity in set(aliases.values()), (
            f"solver {wire!r} projects to unknown provider {identity!r}"
        )
        # LPC-090 provider route projection agrees for non-alias-only wires.
        if wire != "e":
            assert map_prover_id_to_canonical_provider(wire) in {
                identity,
                wire,
            }


def test_solver_alias_eprover_projects_to_e() -> None:
    maps = load_hammer_vocabs()
    alias = maps["solver_alias"]["labels"]["eprover"]
    assert alias["canonical_identity"] == "eprover"
    assert alias["residual"]["alias_of"] == "e"
    # Policy normalizes eprover → e residual wire, which projects to eprover.
    policy = HammerSupervisorPolicy(
        allowed_solvers=("eprover",),
        environment_lock={
            "itp": "lean",
            "itp_version": "4.19.0",
            "kernel_command_template": "lean {source}",
            "solver_versions": {"e": "e-pinned"},
            "executable_paths": {
                "lean": "/opt/pinned/bin/lean",
                "e": "/opt/pinned/bin/e",
            },
            "os_info": "linux-x86_64-pinned",
            "container_digest": "sha256:environment",
        },
        fallback_checks=("pytest:alias",),
    )
    assert policy.allowed_solvers == ("e",)
    assert derive_solver_canonical("e") == "eprover"


def test_policy_rejects_unknown_solver_and_translation_family() -> None:
    lock = {
        "itp": "lean",
        "itp_version": "4.19.0",
        "kernel_command_template": "lean {source}",
        "solver_versions": {"z3": "z3-pinned"},
        "executable_paths": {
            "lean": "/opt/pinned/bin/lean",
            "z3": "/opt/pinned/bin/z3",
        },
        "os_info": "linux-x86_64-pinned",
        "container_digest": "sha256:environment",
    }
    with pytest.raises(ValueError, match="unknown Hammer solver"):
        HammerSupervisorPolicy(
            allowed_solvers=("not_a_solver",),
            environment_lock=lock,
            fallback_checks=("pytest:unknown-solver",),
        )
    with pytest.raises(ValueError, match="unsupported values"):
        HammerSupervisorPolicy(
            allowed_solvers=("z3",),
            translation_families=("not_a_family",),
            environment_lock=lock,
            fallback_checks=("pytest:unknown-family",),
        )


def test_translation_aliases_project_through_primary_wire() -> None:
    maps = load_hammer_vocabs()
    aliases = maps["translation_alias"]["labels"]
    primaries = maps["translation_family"]["labels"]
    for wire, row in aliases.items():
        alias_of = row["residual"]["alias_of"]
        assert alias_of in primaries
        assert row["canonical_identity"] == primaries[alias_of]["canonical_identity"]


# ---------------------------------------------------------------------------
# Semantic separations (LPC-G120 acceptance)
# ---------------------------------------------------------------------------


def test_flogic_and_frame_wire_remain_distinct_with_shared_canonical() -> None:
    maps = load_hammer_vocabs()
    labels = maps["logic_family"]["labels"]
    assert labels["flogic"]["canonical_identity"] == "frame_logic"
    assert labels["frame"]["canonical_identity"] == "frame_logic"
    assert labels["flogic"]["disposition"] == "residual_collapse"
    assert labels["frame"]["disposition"] == "residual_collapse"
    # Wire normalization keeps them distinct.
    assert normalize_registry_logic_family("flogic") is LogicFamily.FLOGIC
    assert normalize_registry_logic_family("frame") is LogicFamily.FRAME
    assert normalize_registry_logic_family("flogic") is not (
        normalize_registry_logic_family("frame")
    )
    # Canonical projection collapses with residual reverse map owned by LPC-090.
    assert to_canonical_registry_logic_family("flogic") == "frame_logic"
    assert to_canonical_registry_logic_family("frame") == "frame_logic"


def test_dcec_and_deontic_remain_distinct_families() -> None:
    maps = load_hammer_vocabs()
    labels = maps["logic_family"]["labels"]
    assert labels["dcec"]["canonical_identity"] == "dcec"
    assert labels["deontic"]["canonical_identity"] == "deontic"
    assert labels["dcec"]["canonical_identity"] != labels["deontic"][
        "canonical_identity"
    ]
    assert normalize_registry_logic_family("dcec") is LogicFamily.DCEC
    assert normalize_registry_logic_family("deontic") is LogicFamily.DEONTIC
    assert to_canonical_registry_logic_family("dcec") == "dcec"
    assert to_canonical_registry_logic_family("deontic") == "deontic"


def test_family_encoding_provider_axes_are_non_interchangeable() -> None:
    maps = load_hammer_vocabs()
    # first_order is a family; smt_lib2 is an encoding — distinct axes.
    family_id = maps["translation_family"]["labels"]["first_order"][
        "canonical_identity"
    ]
    encoding_id = maps["translation_family"]["labels"]["smtlib2"][
        "canonical_identity"
    ]
    assert maps["translation_family"]["labels"]["first_order"]["axis"] == "family"
    assert maps["translation_family"]["labels"]["smtlib2"]["axis"] == "encoding"
    assert family_id in catalog_family_ids()
    assert encoding_id in catalog_encoding_ids()
    assert family_id != encoding_id
    # lean4 encoding ≠ lean provider
    lean_encoding = maps["translation_family"]["labels"]["lean4"][
        "canonical_identity"
    ]
    lean_provider = maps["target_itp"]["labels"]["lean"]["canonical_identity"]
    assert maps["translation_family"]["labels"]["lean4"]["axis"] == "encoding"
    assert maps["target_itp"]["labels"]["lean"]["axis"] == "provider"
    assert lean_encoding in catalog_encoding_ids()
    assert lean_provider in catalog_provider_ids()
    # vampire provider ≠ atp lane
    assert "vampire" in catalog_provider_ids()
    assert "atp" in catalog_lane_ids()
    assert "vampire" != "atp"


def test_atp_candidates_smt_sat_and_kernel_proof_stay_distinct() -> None:
    maps = load_hammer_vocabs()
    separations = maps["semantic_separation"]["labels"]
    assert "atp_candidate_vs_smt_sat" in separations
    assert "lean_source_vs_proof_authority" in separations
    assert "portfolio_vs_kernel" in separations
    authority = maps["authority_ceiling"]["labels"]
    assert authority["atp_candidate"]["canonical_identity"] == "advisory"
    assert authority["smt_candidate"]["canonical_identity"] == "advisory"
    assert authority["portfolio_success"]["canonical_identity"] == "advisory"
    evidence = maps["evidence_kind"]["labels"]
    assert evidence["candidate"]["canonical_identity"] == "candidate"
    assert evidence["kernel_checked_proof"]["canonical_identity"] == (
        "kernel_checked_proof"
    )
    assert evidence["candidate"]["canonical_identity"] != evidence[
        "kernel_checked_proof"
    ]["canonical_identity"]


def test_hammer_capability_remains_candidate_producing() -> None:
    provider = IpfsDatasetsLogicProvider(
        policy=HammerSupervisorPolicy(
            allowed_solvers=("z3",),
            environment_lock={
                "itp": "lean",
                "itp_version": "4.19.0",
                "kernel_command_template": "lean {source}",
                "solver_versions": {"z3": "z3-pinned"},
                "executable_paths": {
                    "lean": "/opt/pinned/bin/lean",
                    "z3": "/opt/pinned/bin/z3",
                },
                "os_info": "linux-x86_64-pinned",
                "container_digest": "sha256:environment",
            },
            fallback_checks=("pytest:capability",),
        )
    )
    capability = provider.capabilities()
    metadata = dict(capability.metadata or {})
    assert metadata.get("candidate_authoritative") is False
    assert metadata.get("kernel_reconstruction_required") is True
    # Provider id stays the catalog hammer advisory lane.
    assert capability.provider_id == "hammer"


# ---------------------------------------------------------------------------
# Residual constants: no free-form inventories beyond the note
# ---------------------------------------------------------------------------


def test_adapter_residual_constants_are_subset_of_documented_vocab() -> None:
    """Every string literal in residual constants must appear in the note.

    Acceptance: no hand-maintained family/provider/encoding/authority list may
    invent identities outside the catalog-bound vocabulary artifact.
    """

    maps = load_hammer_vocabs()
    documented_wire: set[str] = set()
    documented_canonical: set[str] = set()
    for domain, record in maps.items():
        for wire, row in record["labels"].items():
            documented_wire.add(wire)
            documented_canonical.add(row["canonical_identity"])
            for value in row["residual"].values():
                documented_wire.add(value)

    source = adapter_source_path().read_text(encoding="utf-8")
    literals = _module_level_string_literals(source)
    # Required residual surfaces must exist in the adapter module.
    for name in (
        "SUPPORTED_TRANSLATION_FAMILIES",
        "KNOWN_HAMMER_SOLVERS",
        "SUPPORTED_LOGIC_FAMILIES",
    ):
        assert name in literals, f"missing residual constant {name}"

    # No residual constant may introduce an undocumented wire token.
    for name, values in literals.items():
        unknown = sorted(
            value
            for value in values
            if value not in documented_wire
            and value not in documented_canonical
            # Enum member names and structural field names are not wire tokens.
            and value
            not in {
                "LogicFamily",
                "item.value",
            }
        )
        # Filter pure non-vocabulary noise: residual maps only hold vocab tokens.
        if name in {
            "SUPPORTED_TRANSLATION_FAMILIES",
            "KNOWN_HAMMER_SOLVERS",
            "SUPPORTED_LOGIC_FAMILIES",
            "_FAMILY_ALIASES",
            "_FAMILY_ITP",
            "_FAMILY_TARGET",
            "_SOLVER_ALIASES",
        }:
            assert not unknown, (
                f"{name} contains undocumented wire tokens {unknown}; "
                f"catalog-bound note is authoritative"
            )


def test_no_independent_authority_list_in_adapter_source() -> None:
    """Adapter source must not hard-code an authority promotion inventory."""

    source = adapter_source_path().read_text(encoding="utf-8")
    # Candidate-producing floor: capability metadata must declare non-authority.
    assert 'candidate_authoritative": False' in source or (
        "'candidate_authoritative': False" in source
    )
    assert "completion_authority" in source
    # Must not claim authoritative proof from portfolio success alone.
    forbidden_promotions = (
        'evidence_authority": "authoritative"',
        "EvidenceAuthority.AUTHORITATIVE",
        "authority_ceiling = \"authoritative\"",
    )
    for token in forbidden_promotions:
        assert token not in source, f"forbidden authority promotion: {token}"


def test_adapter_module_exports_catalog_projection_helpers() -> None:
    assert hasattr(hammer_adapter, "to_canonical_registry_logic_family")
    assert hasattr(hammer_adapter, "SUPPORTED_LOGIC_FAMILIES")
    assert hasattr(hammer_adapter, "SUPPORTED_TRANSLATION_FAMILIES")
    assert hasattr(hammer_adapter, "KNOWN_HAMMER_SOLVERS")
    assert hammer_adapter.IPFS_DATASETS_LOGIC_PROVIDER_ID == "hammer"


def test_semantic_separation_domain_covers_required_pairs() -> None:
    maps = load_hammer_vocabs()
    labels = maps["semantic_separation"]["labels"]
    required = {
        "flogic_vs_frame_wire",
        "dcec_vs_deontic",
        "family_vs_encoding",
        "encoding_vs_provider",
        "provider_vs_lane",
        "atp_candidate_vs_smt_sat",
        "lean_source_vs_proof_authority",
        "portfolio_vs_kernel",
    }
    missing = required - set(labels)
    assert not missing, f"missing semantic separation rows: {sorted(missing)}"
