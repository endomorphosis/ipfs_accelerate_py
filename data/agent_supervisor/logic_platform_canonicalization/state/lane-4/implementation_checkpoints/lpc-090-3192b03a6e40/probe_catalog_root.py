"""One-shot probe for LPC-090 catalog root + adapter projections."""
from __future__ import annotations

import json

from ipfs_datasets_py.logic.families.canonical_catalog import (
    DEFAULT_CANONICAL_CATALOG_SNAPSHOT,
)
from ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter import (
    SupervisorCanonicalLogicAdapter,
)


def main() -> None:
    snap = DEFAULT_CANONICAL_CATALOG_SNAPSHOT
    adapter = SupervisorCanonicalLogicAdapter()
    inv = adapter.vocabulary_inventory()
    rows = []
    for family in inv["analysis_families"]:
        p = adapter.project_analysis_family(family)
        rows.append(
            {
                "domain": p.domain,
                "legacy": p.supervisor_id,
                "canonical": p.canonical_id,
                "residual": dict(p.residual),
            }
        )
    for kind in inv["property_kinds"]:
        p = adapter.project_property_kind(kind)
        rows.append(
            {
                "domain": p.domain,
                "legacy": p.supervisor_id,
                "canonical": p.canonical_id,
                "residual": dict(p.residual),
            }
        )
    for form in inv["logic_forms"]:
        p = adapter.project_logic_form(form)
        rows.append(
            {
                "domain": p.domain,
                "legacy": p.supervisor_id,
                "canonical": p.canonical_id,
                "residual": dict(p.residual),
            }
        )
    for tc in inv["translation_classes"]:
        p = adapter.project_translation_class(tc)
        rows.append(
            {
                "domain": p.domain,
                "legacy": p.supervisor_id,
                "canonical": p.canonical_id,
                "residual": dict(p.residual),
            }
        )
    for scope in inv["cache_scopes"]:
        p = adapter.project_cache_scope(scope)
        rows.append(
            {
                "domain": p.domain,
                "legacy": p.supervisor_id,
                "canonical": p.canonical_id,
                "residual": dict(p.residual),
            }
        )
    # publication dispositions for family ids
    pub = {}
    publication = snap.publication
    entries = getattr(publication, "entries", ()) or ()
    by_id = getattr(publication, "by_family_id", None)
    if callable(by_id):
        pass
    for entry in entries:
        fid = getattr(entry, "family_id", None)
        disp = getattr(entry, "disposition", None)
        if fid is not None:
            pub[str(fid)] = str(getattr(disp, "value", disp))
    # taxonomy family support levels
    taxonomy_support = {}
    for fid, desc in snap.taxonomy.families.items():
        taxonomy_support[str(fid)] = str(
            getattr(getattr(desc, "support_level", None), "value", getattr(desc, "support_level", ""))
        )
    payload = {
        "catalog_root": snap.content_root,
        "catalog_digest": snap.content_digest,
        "rows": rows,
        "publication_dispositions": pub,
        "taxonomy_support": taxonomy_support,
        "prover_aliases": {"coq": "rocq", "e": "eprover"},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
