"""Measure a shared legal module corpus without running Lake.

Two caller norms share one explicit definition. The existing Lake library
target stays ``legal``. Each declaration keeps its graph, interpretation,
and source identity, and the corpus is several module files.
"""

from __future__ import annotations

import base64
import hashlib
import json

from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

from .dependency_block_resolution import EXPECTED_SHARED_MODULE_PROBE

_TOOLCHAIN = "leanprover/lean4:v4.34.1"


def measure_shared_module_probe() -> dict[str, object] | None:
    """Return the fixed shared-module contract, or None when it drifts."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_SHARED_MODULE_PROBE:
        return None
    return measured


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wire(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode()


def _source() -> tuple[bytes, dict[str, object]]:
    from ipfs_datasets_py.logic.autoformal import legal_statutory_context as context

    raw = (
        '<meta charset="utf-8"><h3 class="section-head">§1. Authored shared café fixture</h3>'
        '<p class="statutory-body">(a) Définitions — In this section:</p>'
        '<p class="statutory-body-1em">(1) The term "covered filer" means registered or licensed.</p>'
        '<p class="statutory-body">(b) Each covered filer shall register.</p>'
        '<p class="statutory-body">(c) Each covered filer may file.</p>'
        '<p class="statutory-body">(d) Each covered filer shall not disclose.</p>'
    ).encode()
    url = (
        "https://www.govinfo.gov/content/pkg/USCODE-2024-title40/html/"
        "USCODE-2024-title40-sec1.htm"
    )
    document = context.extract_document(raw, url=url, edition=2024, legal_id="usc:us:40:1")
    return raw, document


def _candidate(modality: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from ipfs_datasets_py.logic.autoformal import corpus_definition_norm_adapter as adapter
    from ipfs_datasets_py.logic.autoformal import legal_source_definition_norm_bridge as bridge

    raw, document = _source()
    text = str(document["document_text"])
    definition_index = 1
    definition_start = int(document["paragraphs"][definition_index]["char_start"])

    def span(literal: str, start: int = 0) -> dict[str, object]:
        left = text.index(literal, start)
        return {"char_start": left, "char_end": left + len(literal), "text": literal}

    declaration = {
        "declaration_id": "authored-shared-definition",
        "profile": bridge.definitions.PROFILE,
        "interpretation_status": "unreviewed_caller_declaration",
        "paragraph_index": definition_index,
        "scope_span": span("In this section:"),
        "definition_cue": span("means", definition_start),
        "head": span("covered filer", definition_start),
        "body": {
            "op": "any",
            "operands": [
                {"op": "atom", "span": span(literal, definition_start)}
                for literal in ("registered", "licensed")
            ],
        },
    }
    definition = bridge.definitions.prepare_definition(
        {
            "document": document,
            "raw_html_base64": base64.b64encode(raw).decode(),
            "declaration": declaration,
            "family": "fol",
        }
    )
    norm_index = {"O": 2, "P": 3}[modality]
    start = int(document["paragraphs"][norm_index]["char_start"])
    cue = bridge.CUES[modality]
    action = {"O": "register", "P": "file"}[modality]
    request = {
        "definition_request": {
            "document": document,
            "raw_html_base64": base64.b64encode(raw).decode(),
            "declaration": declaration,
            "family": "fol",
        },
        "norm_declaration": {
            "norm_id": "authored-shared-" + modality,
            "profile": bridge.PROFILE,
            "interpretation_status": "unreviewed_caller_declaration",
            **{
                key: document[key]
                for key in ("document_id", "document_text_sha256", "legal_id", "edition")
            },
            "paragraph_index": norm_index,
            "source_span": span(f"Each covered filer {cue} {action}.", start),
            "quantifier_span": span("Each", start),
            "category_span": span("covered filer", start),
            "modality_span": span(cue, start),
            "action_span": span(action, span(cue, start)["char_end"]),
            "modality": modality,
            "variable": "x",
            "definition_head_symbol": definition["native_ast"]["formula"]["left"]["name"],
            "activation_scope": bridge.ACTIVATION,
            "modal_context": None,
        },
        "family": "deontic_fol",
        "source_provenance": {
            "kind": "authored_official_shape_fixture",
            "authenticity_verified": False,
            "independently_reviewed": False,
            "training_admitted": False,
        },
    }
    report = bridge.prepare_definition_norm(request)
    request_raw, report_raw = _wire(request), _wire(report)
    extracted_sha = _sha(text.encode())
    record = adapter.adapt_definition_norm_bytes(
        report_raw,
        expected_report_sha256=_sha(report_raw),
        request_raw=request_raw,
        expected_request_sha256=_sha(request_raw),
        raw_html=raw,
        expected_raw_sha256=_sha(raw),
        expected_extracted_sha256=extracted_sha,
    )
    candidate_raw = canonical_dag_json_bytes(record)
    inputs = {
        "candidate_raw": candidate_raw,
        "report_raw": report_raw,
        "request_raw": request_raw,
        "raw_html": raw,
    }
    pins = {
        "candidate_sha256": _sha(candidate_raw),
        "candidate_cid": record["record_cid"],
        "graph_cid": record["graph"]["record_cid"],
        "request_sha256": _sha(request_raw),
        "report_sha256": _sha(report_raw),
        "raw_sha256": _sha(raw),
        "extracted_sha256": extracted_sha,
    }
    return inputs, pins, record


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_corpus_definition_use_lake_v2 as v2
    from ipfs_datasets_py.logic.autoformal import legal_corpus_shared_lake as shared
    from ipfs_datasets_py.logic.autoformal.corpus_contracts import CorpusContractError

    cases = [_candidate("O"), _candidate("P")]
    preparation = shared.prepare_shared_definition_corpus(
        [case[0] for case in cases],
        expected_population=[case[1] for case in cases],
        toolchain=_TOOLCHAIN,
    )
    manifest = preparation.to_dict()
    files = dict(preparation.files)
    lakefile = files["lakefile.toml"]
    library_line = [
        line for line in lakefile.splitlines() if line.startswith("name = ")
    ]
    if library_line != ['name = "legal_corpus_shared"', 'name = "legal"']:
        raise RuntimeError("shared corpus did not reuse the legal library target")
    if 'roots = ["LegalCorpus"]' not in lakefile:
        raise RuntimeError("shared corpus changed the LegalCorpus root")
    root = files["LegalCorpus.lean"]
    if not root.startswith("import ") or "def " in root:
        raise RuntimeError("shared corpus root is not an import list")
    definitions = manifest["definition_modules"]
    norms = manifest["norm_bindings"]
    if len(definitions) != 1 or len(norms) != 2:
        raise RuntimeError("shared corpus did not keep one definition and two norms")
    definition = definitions[0]
    graphs: set[str] = set()
    interpretations: set[str] = set()
    sources: set[str] = set()
    norm_modules: list[str] = []
    for ordinal, (norm, case) in enumerate(zip(norms, cases, strict=True)):
        record = case[2]
        original = norm["original_joint_binding"]
        contributor = definition["contributors"][ordinal]
        if original["graph_cid"] != record["graph"]["record_cid"]:
            raise RuntimeError("declaration lost its graph")
        if contributor["interpretation_cid"] != record["report_record"]["record_cid"]:
            raise RuntimeError("declaration lost its interpretation")
        occurrence_ids = [row["record_cid"] for row in record["source_occurrences"]]
        if (
            original["source_occurrence_cids"] != occurrence_ids
            or original["source_bindings"] != record["source_bindings"]
            or not occurrence_ids
        ):
            raise RuntimeError("declaration lost its source")
        if norm["qualified_declaration"] != norm["module"] + ".formula_0":
            raise RuntimeError("declaration is not module-local")
        if norm["module_path"] not in files:
            raise RuntimeError("norm module is not its own file")
        graphs.add(str(original["graph_cid"]))
        interpretations.add(str(contributor["interpretation_cid"]))
        sources.add(str(record["source_version"]["record_cid"]))
        norm_modules.append(str(norm["module"]))
    if definition["qualified_declaration"] != definition["module"] + ".formula_0":
        raise RuntimeError("shared definition declaration is not module-local")
    if definition["module_path"] not in files:
        raise RuntimeError("shared definition is not its own file")
    lean_files = [name for name in files if name.endswith(".lean")]
    if len(files) <= 1 or len(lean_files) <= 1:
        raise RuntimeError("shared corpus collapsed to one file")
    if (
        manifest["preparation_only"] is not True
        or manifest["backend_executed"] is not False
        or manifest["lean_compilation_verified"] is not False
        or manifest["admitted"] is not False
        or manifest["formalized"] is not False
    ):
        raise RuntimeError("shared preparation executed a build or claimed admission")
    try:
        v2.prepare_definition_use_corpus_v2(
            b"",
            expected_assessment_file_sha256="0" * 64,
            review_inputs={},
            toolchain="nope",
        )
    except CorpusContractError as exc:
        refused = str(exc)
    else:
        raise RuntimeError("v2 preparation accepted an unpinned toolchain")
    return {
        "admitted": False,
        "backend_executed": False,
        "declaration_graph_roots_retained": True,
        "declaration_interpretation_roots_retained": True,
        "declaration_source_roots_retained": True,
        "definition_module_count": 1,
        "definition_modules": [str(definition["module"])],
        "external_definition_resolution": False,
        "file_count": len(files),
        "formalized": False,
        "graph_count": len(graphs),
        "independently_reviewed": False,
        "interpretation_count": len(interpretations),
        "lake_build_executed": False,
        "lean_compilation_verified": False,
        "lean_file_count": len(lean_files),
        "library_roots": ["LegalCorpus"],
        "library_target": "legal",
        "norm_module_count": 2,
        "norm_modules": sorted(norm_modules),
        "one_file_corpus": False,
        "preparation_only": True,
        "project_name": "legal_corpus_shared",
        "proof_authority": False,
        "qualified_declarations_are_module_local": True,
        "root_file_is_import_list": True,
        "shared_definition_deduplication": True,
        "source_semantics_verified": False,
        "source_version_count": len(sources),
        "training_qualified": False,
        "v2_build_eligible": v2.FALSE_FLAGS["build_eligible"] is False,
        "v2_kernel_proofs_checked": v2.FALSE_FLAGS["kernel_proofs_checked"] is False,
        "v2_max_files": v2.MAX_FILES,
        "v2_toolchain_probed": v2.FALSE_FLAGS["toolchain_probed"] is False,
        "v2_unpinned_toolchain_refused": refused,
    }
