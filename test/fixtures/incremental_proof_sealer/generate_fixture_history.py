"""Deterministic IPS fixture repository/history/proof-graph generator (IPS-045).

No wall clock, host environment, or network.  Simulated evidence is labeled
``rejection_only`` and never production success.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

EVIDENCE_SUBSET: Final[str] = "ips/fixture-corpus@1"
MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "fixture-manifest@1"
)
GENESIS_PARENT: Final[str] = "fixture:genesis@1"
FIXTURE_DIR: Final[Path] = Path(__file__).resolve().parent
MANIFEST_NAME: Final[str] = "fixture_manifest.json"

REQUIRED_SCENARIO_KINDS: Final[tuple[str, ...]] = (
    "source_implementation",
    "public_interface",
    "test_selector",
    "test_source",
    "test_add",
    "test_delete",
    "fixture",
    "relevant_configuration",
    "network_policy",
    "verification_policy",
    "dependency_lock",
    "tool_prover_version",
    "circuit_key",
    "proof_schema",
    "canonicalization",
    "checked_specification_document",
    "ordinary_documentation",
    "graph_manifest",
    "branch",
    "merge",
    "rollback",
    "corruption",
    "independent_module",
)

_FULL_FALLBACK_KINDS: Final[frozenset[str]] = frozenset(
    {
        "circuit_key",
        "proof_schema",
        "canonicalization",
        "dependency_lock",
        "verification_policy",
        "tool_prover_version",
        "corruption",
        "merge",
    }
)


def _cid(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _scenario(
    kind: str,
    *,
    parent: str,
    changed_artifact: str,
    direct_units: Sequence[str],
    transitive_units: Sequence[str],
    aggregate_effect: str,
    full_fallback: bool,
    simulated: bool = False,
) -> dict[str, Any]:
    body = {
        "kind": kind,
        "parent": parent,
        "changed_artifact": changed_artifact,
        "changed_artifact_provenance": {
            "path": changed_artifact,
            "mutation": kind,
            "byte_stable": True,
        },
        "expected_direct_unit_closure": list(direct_units),
        "expected_transitive_unit_closure": list(transitive_units),
        "aggregate_effect": aggregate_effect,
        "full_fallback_decision": {
            "required": full_fallback,
            "reason": kind if full_fallback else "incremental_reuse_justified",
        },
        "simulated_proving": "rejection_only" if simulated else "absent",
        "production_success": False if simulated else None,
    }
    body["scenario_cid"] = _cid(body)
    return body


def generate_corpus() -> dict[str, Any]:
    """Return the canonical fixture corpus.  Repeated calls are byte-identical."""

    scenarios: list[dict[str, Any]] = []
    parent = GENESIS_PARENT
    for index, kind in enumerate(REQUIRED_SCENARIO_KINDS):
        unit = f"unit/{kind}"
        transitive = (unit,) if kind != "independent_module" else (unit, "unit/source_implementation")
        simulated = kind == "corruption"
        record = _scenario(
            kind,
            parent=parent,
            changed_artifact=f"artifact/{kind}@{index:02d}",
            direct_units=(unit,),
            transitive_units=transitive,
            aggregate_effect="rebuild_affected_branch" if not kind in _FULL_FALLBACK_KINDS else "full_checkpoint",
            full_fallback=kind in _FULL_FALLBACK_KINDS,
            simulated=simulated,
        )
        scenarios.append(record)
        parent = record["scenario_cid"]

    corpus = {
        "schema": MANIFEST_SCHEMA,
        "evidence_subset": EVIDENCE_SUBSET,
        "genesis_parent": GENESIS_PARENT,
        "scenario_kinds": list(REQUIRED_SCENARIO_KINDS),
        "scenarios": scenarios,
        "notes": [
            "No wall-clock or host-environment inputs.",
            "Simulated proving appears only as rejection_only and never as production success.",
        ],
    }
    corpus["corpus_cid"] = _cid({k: v for k, v in corpus.items() if k != "corpus_cid"})
    return corpus


def render_manifest(corpus: Mapping[str, Any] | None = None) -> str:
    payload = corpus if corpus is not None else generate_corpus()
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def write_manifest(path: Path | None = None) -> Path:
    target = path or (FIXTURE_DIR / MANIFEST_NAME)
    target.write_text(render_manifest(), encoding="utf-8")
    return target


def main() -> int:
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
