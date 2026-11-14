"""Compare incremental builds with a clean ``lake build legal``.

Unchanged, changed, and newly ambiguous shared-definition partitions are
built with the incremental owner and again with the shared Lake owner.
Expected modules and semantic obligations must agree. Direct ``lean -o``
olean bytes stay distinct from the Lake artifacts, so that cache does not
substitute for ``lake build legal``. Neither build establishes law.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .dependency_block_resolution import EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE

_LAKE = Path("/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lake")
_TOOLCHAIN = "leanprover/lean4:v4.34.1"
_ARGV = [
    "installed-sibling-lean",
    "-o",
    "compiled/<module>.olean",
    "<module>.lean",
]
_LAKE_TAIL = ["--offline", "--no-cache", "build", "legal"]
_ASSUMPTIONS = [
    "The exact shared definition holds at the evaluation time.",
    "The universally guarded norm holds at that same time under the same interpretation.",
    "The declared definition body holds for the supplied entity at that time.",
    "The conclusion is the parameterized modal action; it does not establish actual action truth or legal truth.",
]
_FALSE_KEYS = (
    "admitted",
    "formalized",
    "proof_authority",
    "source_semantics_verified",
    "source_authenticity_verified",
    "independently_reviewed",
    "training_qualified",
)
_cached: dict[str, object] | None = None


def measure_incremental_lake_agreement_probe() -> dict[str, object] | None:
    """Return the executed Lake comparison, or None when a check drifts."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE:
        return None
    _cached = measured
    return dict(measured)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wire(value: object, *, indent: int | None) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    ).encode()


def _source(context, *, edition: int = 2024, duplicate_definition: bool = False):
    second = (
        '<p class="statutory-body-1em">(2) The term "covered filer" means registered or licensed.</p>'
        if duplicate_definition
        else ""
    )
    raw = (
        '<meta charset="utf-8"><h3 class="section-head">§1. Authored shared café fixture</h3>'
        '<p class="statutory-body">(a) Définitions — In this section:</p>'
        '<p class="statutory-body-1em">(1) The term "covered filer" means registered or licensed.</p>'
        + second
        + '<p class="statutory-body">(b) Each covered filer shall register.</p>'
        '<p class="statutory-body">(c) Each covered filer may file.</p>'
        '<p class="statutory-body">(d) Each covered filer shall not disclose.</p>'
    ).encode()
    url = (
        f"https://www.govinfo.gov/content/pkg/USCODE-{edition}-title40/html/"
        f"USCODE-{edition}-title40-sec1.htm"
    )
    document = context.extract_document(
        raw, url=url, edition=edition, legal_id="usc:us:40:1"
    )
    return raw, document


def _candidate(
    bridge,
    adapter,
    context,
    canonical_dag_json_bytes,
    modality: str = "O",
    family: str = "deontic_fol",
    *,
    edition: int = 2024,
    duplicate_definition: bool = False,
    definition_index: int = 1,
    norm_id: str | None = None,
):
    raw, document = _source(
        context, edition=edition, duplicate_definition=duplicate_definition
    )
    text = document["document_text"]
    definition_start = document["paragraphs"][definition_index]["char_start"]

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
    definition_request = {
        "document": document,
        "raw_html_base64": base64.b64encode(raw).decode(),
        "declaration": declaration,
        "family": "fol",
    }
    definition = bridge.definitions.prepare_definition(definition_request)
    norm_index = {"O": 2, "P": 3, "F": 4}[modality] + int(duplicate_definition)
    start = document["paragraphs"][norm_index]["char_start"]
    cue = bridge.CUES[modality]
    action = {"O": "register", "P": "file", "F": "disclose"}[modality]
    norm = {
        "norm_id": norm_id or "authored-shared-" + modality,
        "profile": bridge.PROFILE,
        "interpretation_status": "unreviewed_caller_declaration",
        **{
            key: document[key]
            for key in (
                "document_id",
                "document_text_sha256",
                "legal_id",
                "edition",
            )
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
    }
    request = {
        "definition_request": definition_request,
        "norm_declaration": norm,
        "family": family,
        "source_provenance": {
            "kind": "authored_official_shape_fixture",
            "authenticity_verified": False,
            "independently_reviewed": False,
            "training_admitted": False,
        },
    }
    report = bridge.prepare_definition_norm(request)
    request_raw, report_raw = _wire(request, indent=2), _wire(report, indent=2)
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
    return inputs, pins, record, report


def _changed_definition(bridge, case):
    """Change only the caller interpretation and keep the physical source bytes."""

    request = json.loads(case[0]["request_raw"])
    declaration = request["definition_request"]["declaration"]
    if declaration["body"]["op"] != "any":
        raise ValueError("changed partition started from a different definition body")
    declaration["body"]["op"] = "all"
    definition = bridge.definitions.prepare_definition(request["definition_request"])
    request["norm_declaration"]["definition_head_symbol"] = definition["native_ast"][
        "formula"
    ]["left"]["name"]
    report = bridge.prepare_definition_norm(request)
    request_raw, report_raw = _wire(request, indent=None), _wire(report, indent=None)
    raw_html = case[0]["raw_html"]
    extracted_sha = _sha(request["definition_request"]["document"]["document_text"].encode())
    from ipfs_datasets_py.logic.autoformal import corpus_definition_norm_adapter as adapter
    from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

    record = adapter.adapt_definition_norm_bytes(
        report_raw,
        expected_report_sha256=_sha(report_raw),
        request_raw=request_raw,
        expected_request_sha256=_sha(request_raw),
        raw_html=raw_html,
        expected_raw_sha256=_sha(raw_html),
        expected_extracted_sha256=extracted_sha,
    )
    candidate_raw = canonical_dag_json_bytes(record)
    inputs = {
        "candidate_raw": candidate_raw,
        "report_raw": report_raw,
        "request_raw": request_raw,
        "raw_html": raw_html,
    }
    pins = {
        "candidate_sha256": _sha(candidate_raw),
        "candidate_cid": record["record_cid"],
        "graph_cid": record["graph"]["record_cid"],
        "request_sha256": _sha(request_raw),
        "report_sha256": _sha(report_raw),
        "raw_sha256": _sha(raw_html),
        "extracted_sha256": extracted_sha,
    }
    return inputs, pins, record, report


def _prepare(shared, cases):
    return shared.prepare_shared_definition_corpus(
        [case[0] for case in cases],
        expected_population=[case[1] for case in cases],
        toolchain=_TOOLCHAIN,
    )


def _incremental(incremental, cases, preparation, directory, previous=None):
    if directory.exists():
        shutil.rmtree(directory)
    arguments = {
        "candidate_inputs": [case[0] for case in cases],
        "expected_population": [case[1] for case in cases],
        "toolchain": _TOOLCHAIN,
        "lake_executable": str(_LAKE),
        "output_directory": directory,
        "timeout_seconds": 60,
    }
    if previous is not None:
        old_receipt, old_directory = previous
        raw = (old_directory / "cache.json").read_bytes()
        arguments.update(
            previous_cache_raw=raw,
            expected_previous_cache_sha256=_sha(raw),
            previous_artifacts={
                module: (old_directory / path).read_bytes()
                for module, path in old_receipt["artifact_files"].items()
            },
        )
    receipt = incremental.build_incremental_shared_corpus(preparation, **arguments)
    return receipt, directory


def _lake(shared, cases, preparation, directory):
    if directory.exists():
        shutil.rmtree(directory)
    return shared.build_shared_definition_corpus(
        preparation,
        candidate_inputs=[case[0] for case in cases],
        expected_population=[case[1] for case in cases],
        toolchain=_TOOLCHAIN,
        lake_executable=str(_LAKE),
        output_directory=directory,
        timeout_seconds=60,
    )


def _authority_false(receipt: dict) -> None:
    for key in _FALSE_KEYS:
        if receipt.get(key) is not False:
            raise ValueError("build receipt granted " + key)
    if "legal_acceptance" in receipt or "federal_corpus_acceptance" in receipt:
        raise ValueError("build receipt carries an acceptance key")


def _axioms(receipt: dict, directory: Path) -> dict[str, list]:
    cache = json.loads((directory / "cache.json").read_bytes())
    found: dict[str, list] = {}
    for row in cache["module_facts"]:
        found.update(row["output"]["theorem_axiom_dependencies"])
    recipe = cache["execution_contract"]["argv_recipe"]
    if recipe != _ARGV:
        raise ValueError("incremental argv is not the direct lean -o recipe")
    observations = [
        json.loads(line)
        for line in (directory / "module-executions.jsonl").read_bytes().splitlines()
    ]
    for row in observations:
        command = row["command"]
        if command[0] != str(_LAKE.parent / "lean") or command[1] != "-o":
            raise ValueError("incremental compile was not direct lean -o")
    if [row["module"] for row in observations] != receipt["compiled_modules"]:
        raise ValueError("incremental observations differ from compiled modules")
    return found


def _lake_module(path: str) -> str:
    prefix = ".lake/build/lib/lean/"
    if not path.startswith(prefix) or not path.endswith(".olean"):
        raise ValueError("unexpected lake artifact path")
    return path[len(prefix) : -len(".olean")].replace("/", ".")


def _agree(incremental_receipt, incremental_directory, lake_receipt, manifest) -> None:
    _authority_false(incremental_receipt)
    _authority_false(lake_receipt)
    if not (
        incremental_receipt["build_passed"] is True
        and lake_receipt["build_passed"] is True
        and incremental_receipt["composition_obligations_checked"] is True
        and lake_receipt["composition_obligations_checked"] is True
        and incremental_receipt["lean_compilation_verified"] is True
        and lake_receipt["lean_compilation_verified"] is True
    ):
        raise ValueError("incremental and clean Lake builds did not both pass")
    expected = list(manifest["expected_modules"])
    if (
        list(incremental_receipt["expected_modules"]) != expected
        or list(lake_receipt["expected_modules"]) != expected
    ):
        raise ValueError("expected module population differs from the clean Lake build")
    if (
        list(lake_receipt["theorem_assumptions"]) != _ASSUMPTIONS
        or list(manifest["theorem_assumptions"]) != _ASSUMPTIONS
    ):
        raise ValueError("semantic obligations differ from the clean Lake build")
    axioms = _axioms(incremental_receipt, incremental_directory)
    if axioms != lake_receipt["theorem_axiom_dependencies"] or any(axioms.values()):
        raise ValueError("axiom obligations differ from the clean Lake build")
    if lake_receipt["command"][1:] != _LAKE_TAIL or not str(lake_receipt["command"][0]).endswith("/lake"):
        raise ValueError("clean build command is not lake build legal")
    for name in ("lake", "lean"):
        probe = lake_receipt["version_probes"][name]
        if probe["returncode"] != 0 or "4.34.1" not in probe["stdout"]:
            raise ValueError("clean Lake build did not use Lean 4.34.1")
    produced = set(incremental_receipt["compiled_modules"]) | set(incremental_receipt["reused_modules"])
    if produced != set(expected) or set(lake_receipt["compiled_modules"]) != set(expected):
        raise ValueError("an expected artifact is missing from one build")
    lake_bytes = {
        _lake_module(path): digest
        for path, digest in lake_receipt["artifact_sha256"].items()
    }
    if set(lake_bytes) != set(expected):
        raise ValueError("Lake artifact population differs from the expected modules")
    for module in expected:
        incremental_path = incremental_directory / incremental_receipt["artifact_files"][module]
        raw = incremental_path.read_bytes()
        if not raw or _sha(raw) != incremental_receipt["artifact_sha256"][module]:
            raise ValueError("incremental artifact bytes differ from its receipt")
        if incremental_receipt["artifact_sha256"][module] == lake_bytes[module]:
            raise ValueError("direct lean -o bytes matched the Lake artifact")
    if set(incremental_receipt["compiled_modules"]) & set(incremental_receipt["reused_modules"]):
        raise ValueError("a module was both compiled and reused")


def _measure() -> dict[str, object]:
    if not _LAKE.is_file() or _LAKE.is_symlink() or _LAKE.parent.name != "bin":
        raise ValueError("pinned Lean 4.34.1 lake binary required")
    from ipfs_datasets_py.logic.autoformal import corpus_definition_norm_adapter as adapter
    from ipfs_datasets_py.logic.autoformal import legal_corpus_incremental as incremental
    from ipfs_datasets_py.logic.autoformal import legal_corpus_shared_lake as shared
    from ipfs_datasets_py.logic.autoformal import legal_source_definition_norm_bridge as bridge
    from ipfs_datasets_py.logic.autoformal import legal_statutory_context as context
    from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

    if '"fresh_proof_for_reused_modules": False' not in Path(incremental.__file__).read_text(encoding="utf-8"):
        raise ValueError("incremental owner no longer refuses fresh proof for a cache hit")
    root = Path(tempfile.mkdtemp(prefix="lir1502-"))
    try:
        unchanged_cases = [
            _candidate(bridge, adapter, context, canonical_dag_json_bytes, "O"),
            _candidate(bridge, adapter, context, canonical_dag_json_bytes, "P"),
        ]
        unchanged = _prepare(shared, unchanged_cases)
        unchanged_manifest = unchanged.to_dict()
        cold = _incremental(
            incremental, unchanged_cases, unchanged, root / "unchanged-cold"
        )
        warm = _incremental(
            incremental,
            unchanged_cases,
            unchanged,
            root / "unchanged-warm",
            previous=cold,
        )
        unchanged_lake = _lake(
            shared, unchanged_cases, unchanged, root / "unchanged-lake"
        )
        _agree(warm[0], warm[1], unchanged_lake, unchanged_manifest)
        if warm[0]["compiled_modules"] or warm[0]["backend_executed"] is not False:
            raise ValueError("unchanged partition recompiled")
        if len(cold[0]["compiled_modules"]) != len(unchanged_manifest["expected_modules"]):
            raise ValueError("unchanged cold build did not compile every module")
        if len(unchanged_manifest["definition_modules"]) != 1:
            raise ValueError("unchanged partition did not share one definition")

        first = _candidate(
            bridge, adapter, context, canonical_dag_json_bytes,
            "O", edition=2024, norm_id="independent-A",
        )
        second = _candidate(
            bridge, adapter, context, canonical_dag_json_bytes,
            "P", edition=2025, norm_id="independent-B",
        )
        changed_first = _changed_definition(bridge, first)
        original_cases = [first, second]
        changed_cases = [changed_first, second]
        original = _prepare(shared, original_cases)
        changed = _prepare(shared, changed_cases)
        original_cold = _incremental(
            incremental, original_cases, original, root / "changed-cold"
        )
        changed_partial = _incremental(
            incremental, changed_cases, changed, root / "changed-partial", previous=original_cold
        )
        changed_lake = _lake(shared, changed_cases, changed, root / "changed-lake")
        changed_manifest = changed.to_dict()
        _agree(changed_partial[0], changed_partial[1], changed_lake, changed_manifest)
        if len(changed_manifest["definition_modules"]) != 2:
            raise ValueError("changed partition lost its separate definitions")
        if len(changed_partial[0]["compiled_modules"]) != 3 or len(changed_partial[0]["reused_modules"]) != 3:
            raise ValueError("changed partition did not rebuild only its closure")

        unique_cases = [
            _candidate(
                bridge, adapter, context, canonical_dag_json_bytes,
                "O", norm_id="unique-definition",
            )
        ]
        unique = _prepare(shared, unique_cases)
        ambiguous_cases = [
            _candidate(
                bridge, adapter, context, canonical_dag_json_bytes,
                duplicate_definition=True,
                definition_index=index,
                norm_id=f"definition-occurrence-{index}",
            )
            for index in (1, 2)
        ]
        ambiguous = _prepare(shared, ambiguous_cases)
        unique_cold = _incremental(
            incremental, unique_cases, unique, root / "unique-cold"
        )
        ambiguous_partial = _incremental(
            incremental,
            ambiguous_cases,
            ambiguous,
            root / "ambiguous-partial",
            previous=unique_cold,
        )
        ambiguous_lake = _lake(
            shared, ambiguous_cases, ambiguous, root / "ambiguous-lake"
        )
        ambiguous_manifest = ambiguous.to_dict()
        _agree(
            ambiguous_partial[0],
            ambiguous_partial[1],
            ambiguous_lake,
            ambiguous_manifest,
        )
        unique_modules = {row["module"] for row in unique.to_dict()["definition_modules"]}
        ambiguous_modules = {row["module"] for row in ambiguous_manifest["definition_modules"]}
        if len(unique_modules) != 1 or len(ambiguous_modules) != 2:
            raise ValueError("newly ambiguous partition was merged or never split")
        if unique_modules & set(ambiguous_partial[0]["reused_modules"]):
            raise ValueError("the former unique definition was reused after it became ambiguous")
        if ambiguous_partial[0]["reused_modules"] != ["LegalCorpus.Prelude"]:
            raise ValueError("newly ambiguous partition reused a definition module")
        if len(ambiguous_partial[0]["compiled_modules"]) != 5:
            raise ValueError("newly ambiguous partition did not compile its new modules")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return {
        "admitted": False,
        "ambiguous_compiled_count": 5,
        "ambiguous_definition_module_count": 2,
        "ambiguous_reused_count": 1,
        "axiom_lists_empty": True,
        "changed_compiled_count": 3,
        "changed_definition_module_count": 2,
        "changed_reused_count": 3,
        "composition_obligations_checked": True,
        "direct_lean_o_substitutes_for_lake_build_legal": False,
        "expected_module_population_agrees": True,
        "federal_corpus_acceptance_claimed": False,
        "formalized": False,
        "fresh_proof_for_reused_modules": False,
        "incremental_argv_recipe": list(_ARGV),
        "incremental_compiler_invoked": True,
        "lake_build_executed": True,
        "lake_command_tail": list(_LAKE_TAIL),
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "olean_bytes_identical": False,
        "old_unique_definition_reused": False,
        "proof_authority": False,
        "recorded_legal_truth_assumption": True,
        "semantic_obligations_agree": True,
        "source_semantics_verified": False,
        "theorem_assumptions": list(_ASSUMPTIONS),
        "unchanged_cold_compile_count": 5,
        "unchanged_definition_module_count": 1,
        "unchanged_warm_compile_count": 0,
        "unique_definition_module_count": 1,
    }
