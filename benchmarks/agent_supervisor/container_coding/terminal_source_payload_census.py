"""Bounded content-free byte census of real indexed worker payloads.

This diagnostic does not alter a capsule, admit a projection, or remove required
evidence. Optional program source and support source are counted separately.
Model prompts, provider logs, hidden verifiers and task solutions are not read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = "terminal-source-payload-census@1"
MAX_PAYLOAD_BYTES = 256000
MAX_NODES = 20000
MAX_DEPTH = 64
AUTHORED_PROGRAM = '''def calculate_total(values, limit=10):
    """Return a bounded nonnegative sum."""
    total = 0
    for value in values:
        if value > 0:
            total += min(value, limit)
    return total

def outer(value):
    def nested(item):
        return item + 718291
    return nested(value)

class Calculator:
    scale = 2
    def calculate(self, value):
        """Preserve the authored method body marker."""
        return value * 837461
'''
AUTHORED_INSTRUCTION = "Inspect calculate_total, outer and Calculator.calculate using exact public requirements.\n"


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _walk(value):
    pending = [((), value)]
    seen = 0
    while pending:
        path, current = pending.pop()
        seen += 1
        if seen > MAX_NODES or len(path) > MAX_DEPTH:
            raise ValueError("payload census traversal exceeds bounds")
        yield path, current
        if type(current) is dict:
            if any(type(key) is not str for key in current):
                raise ValueError("JSON object keys must be strings")
            pending.extend((path + (key,), child) for key, child in current.items())
        elif type(current) is list:
            pending.extend((path + (index,), child) for index, child in enumerate(current))
        elif type(current) not in (str, int, float, bool, type(None)):
            raise ValueError("payload must contain JSON values only")


def _overlap(payload):
    """Count disjoint repeated dict/list values; scalar identifiers are excluded."""
    groups = defaultdict(list)
    raw_by_digest = {}
    for path, value in _walk(payload):
        if path and type(value) in (dict, list) and value:
            raw = encoded(value)
            if len(raw) >= 64:
                digest = sha(raw)
                groups[digest].append(path)
                raw_by_digest[digest] = raw
    occupied = []
    selected = []
    for digest in sorted(groups, key=lambda key: (-len(raw_by_digest[key]), key)):
        available = [path for path in groups[digest] if not any(
            path[:len(parent)] == parent or parent[:len(path)] == path for parent in occupied)]
        if len(available) < 2:
            continue
        occupied.extend(available)
        selected.append({"subtree_sha256": digest, "serialized_bytes": len(raw_by_digest[digest]),
            "occurrences": len(available), "repeated_value_bytes": (len(available) - 1) * len(raw_by_digest[digest]),
            "top_level_groups": dict(Counter(str(path[0]) for path in available))})
    return {"schema": "disjoint-identical-json-subtrees@1", "minimum_subtree_bytes": 64,
        "groups": selected, "strict_identical_repeated_subtree_bytes": sum(row["repeated_value_bytes"] for row in selected),
        "scope": "disjoint identical nonempty dict/list values; excludes repeated scalar identifiers and overlapping descendants",
        "safe_to_remove_bytes": None, "omission_authority": False}


def _long_string_overlap(payload):
    groups = defaultdict(list)
    sizes = {}
    for path, value in _walk(payload):
        if type(value) is str:
            raw = encoded(value)
            if len(raw) >= 128:
                digest = sha(raw)
                groups[digest].append(path)
                sizes[digest] = len(raw)
    selected = [{"value_sha256": digest, "serialized_bytes": sizes[digest],
        "occurrences": len(paths), "repeated_value_bytes": (len(paths) - 1) * sizes[digest],
        "top_level_groups": dict(Counter(str(path[0]) for path in paths))}
        for digest, paths in sorted(groups.items()) if len(paths) > 1]
    return {"minimum_string_serialized_bytes": 128, "groups": selected,
        "strict_identical_repeated_string_bytes": sum(row["repeated_value_bytes"] for row in selected),
        "additive_with_structured_subtree_overlap": False,
        "scope": "exact long string values only; report separately because strings may belong to counted structured subtrees",
        "safe_to_remove_bytes": None, "omission_authority": False}


def census_worker_payload(payload: dict) -> dict:
    """Measure supplied canonical payload shape without exporting its contents."""
    scalar_bytes = 0
    for _, value in _walk(payload):
        if type(value) is str:
            if len(value) > MAX_PAYLOAD_BYTES:
                raise ValueError("bounded worker payload required")
            scalar_bytes += len(value.encode())
            if scalar_bytes > MAX_PAYLOAD_BYTES:
                raise ValueError("bounded worker payload required")
    raw = encoded(payload)
    if type(payload) is not dict or len(raw) > MAX_PAYLOAD_BYTES:
        raise ValueError("bounded worker payload required")
    capsules, admissions = payload.get("capsules"), payload.get("admissions")
    manifest, inline, program = payload.get("manifest"), payload.get("raw_sources"), payload.get("program_paths")
    required, projection = payload.get("required_raw_paths"), payload.get("worker_projection")
    if (payload.get("schema") != "supervisor-semantic-worker-context@2"
            or type(capsules) is not list or len(capsules) > 32 or type(admissions) is not list
            or len(admissions) != len(capsules) or any(type(row) is not dict for row in capsules + admissions)
            or type(manifest) is not dict or not 1 <= len(manifest) <= 64 or type(inline) is not dict
            or type(program) is not list or type(required) is not list
            or any(type(name) is not str for name in program + required)
            or not set(required) <= set(inline) <= set(manifest) or not set(program) <= set(manifest)
            or any(type(value) is not str for value in inline.values())
            or type(projection) is not dict or type(payload.get("pack")) is not dict):
        raise ValueError("explicit indexed program/support worker payload required")
    for name, text in inline.items():
        binding = manifest[name]
        if type(binding) is not dict or sha(text.encode()) != binding.get("sha256"):
            raise ValueError("inline raw bytes differ from manifest")
    facet_rows, facet_ast_bytes = [], 0
    selected = projection.get("selected_symbols")
    if type(selected) is not list or len(selected) != len(capsules):
        raise ValueError("selected symbols must match capsule population")
    if any(type(row) is not dict or type(row.get("stable_symbol_id")) is not str for row in selected):
        raise ValueError("selected symbol identities must be strings")
    selections = {row["stable_symbol_id"]: row for row in selected}
    if len(selections) != len(capsules):
        raise ValueError("selected symbol identities must be unique")
    for capsule in capsules:
        identity = capsule.get("stable_symbol_id")
        selection = selections.get(identity)
        if type(identity) is not str or not selection:
            raise ValueError("capsule outside selected symbol population")
        metadata = capsule.get("metadata")
        facets = metadata.get("facets", []) if type(metadata) is dict else None
        if type(facets) is not list or len(facets) > 64 or any(type(facet) is not dict for facet in facets):
            raise ValueError("bounded capsule facet list required")
        values = [facet["version_evidence"] for facet in facets if "version_evidence" in facet]
        ast_bytes = sum(len(encoded(value)) for value in values)
        facet_ast_bytes += ast_bytes
        kinds = [value.get("_type") for value in values if type(value) is dict]
        facet_rows.append({"symbol_identity_sha256": sha(identity.encode()),
            "selected_binding_sha256": sha(encoded(selection)), "capsule_sha256": sha(encoded(capsule)),
            "capsule_bytes": len(encoded(capsule)), "facet_version_evidence_bytes": ast_bytes,
            "facet_count": len(facets), "evidence_slot_count": len(values),
            "evidence_root_kinds": {kind: kinds.count(kind) for kind in sorted({kind for kind in kinds if type(kind) is str})}})
    values = {name: len(encoded(value)) for name, value in payload.items()}
    program_inline = set(program) & set(inline)
    support_inline = set(inline) - set(program)
    program_json = sum(len(encoded(inline[name])) for name in program_inline)
    support_json = sum(len(encoded(inline[name])) for name in support_inline)
    measured = {"capsule_facet_version_evidence": facet_ast_bytes,
        "capsule_other_including_array_framing": values["capsules"] - facet_ast_bytes,
        "admissions": values["admissions"], "pack": values["pack"],
        "inline_program_raw_json_values": program_json, "inline_support_raw_json_values": support_json,
        "raw_source_map_keys_and_framing": values["raw_sources"] - program_json - support_json,
        "worker_projection": values["worker_projection"],
        "other_top_level_values": sum(value for key, value in values.items()
            if key not in {"capsules", "admissions", "pack", "raw_sources", "worker_projection"}),
        "top_level_keys_and_framing": len(raw) - sum(values.values())}
    assert sum(measured.values()) == len(raw)
    pack = payload["pack"]
    pack_fields = {key: len(encoded(value)) for key, value in pack.items()}
    return {"schema": SCHEMA, "payload_sha256": sha(raw), "payload_bytes": len(raw),
        "disjoint_byte_partition": measured, "byte_partition_reconciles": sum(measured.values()) == len(raw),
        "top_level_value_bytes": values, "pack_value_bytes": pack_fields,
        "pack_keys_and_framing_bytes": len(encoded(pack)) - sum(pack_fields.values()),
        "capsules": facet_rows, "source_count_in_manifest": len(manifest),
        "program_source_count": len(program), "required_raw_path_count": len(required),
        "inline_program_count": len(program_inline), "inline_support_count": len(support_inline),
        "inline_program_source_utf8_bytes": sum(len(inline[name].encode()) for name in program_inline),
        "inline_support_source_utf8_bytes": sum(len(inline[name].encode()) for name in support_inline),
        "omitted_optional_program_count": len(set(program) - set(required) - set(inline)),
        "required_raw_preserved_in_payload": True, "selected_capsule_count": len(capsules),
        "strict_subtree_overlap": _overlap(payload),
        "strict_long_string_overlap": _long_string_overlap(payload),
        "external_native_or_model_wrapper_bytes": None, "provider_usage_tokens": None,
        "source_text_exported": False, "capsule_content_exported": False,
        "canonical_payload_modified": False, "projection_admission_authority": False,
        "limitations": ["This byte census is not a token estimate or proof of behavioral equivalence.",
            "Identical JSON values are observations, not permission to remove required evidence.",
            "Optional program raw omission does not imply a whole-repository source copy in capsules or pack.",
            "Facet projections may contain selected implementation AST while eliding separately addressable nested bodies.",
            "External native prompt, provider wrappers and model interaction history are outside this worker-payload census."]}


def _save(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _new_output(path):
    path = Path(path).absolute()
    if path.exists() or path.resolve() != path:
        raise ValueError("new canonical output directory required")
    path.mkdir(parents=True)
    return path


def _authored_shape_checks():
    from ipfs_datasets_py.logic.software_contracts.semantic_index.python_analysis import analyze_python_source
    analysis = analyze_python_source(AUTHORED_PROGRAM, "program.py", "authored-census")
    facts = {fact.symbol.qualified_name: fact.symbol.to_dict()["metadata"]["facets"][0]["version_evidence"] for fact in analysis.symbols}
    outer, nested = facts["program.outer"], facts["program.outer.nested"]
    cls, method = facts["program.Calculator"], facts["program.Calculator.calculate"]
    checks = {"ordinary_function_has_implementation_ast": b'"For"' in encoded(facts["program.calculate_total"]),
        "nested_body_elided_from_parent": b"718291" not in encoded(outer),
        "nested_body_retained_in_own_symbol": b"718291" in encoded(nested),
        "method_body_elided_from_class": b"837461" not in encoded(cls),
        "class_retains_member_interface": b'"member":"calculate"' in encoded(cls),
        "method_body_retained_in_own_symbol": b"837461" in encoded(method)}
    return {"scope": "real authoritative Python analyzer facts from authored source", "checks": checks,
        "all_checks_passed": all(checks.values()), "source_sha256": sha(AUTHORED_PROGRAM.encode())}


def run_authored_census(*, output: Path):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context, load_semantic_worker_context
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    output = _new_output(output)
    repository = output / "repository"
    repository.mkdir()
    sources = {"program.py": AUTHORED_PROGRAM, "required.md": AUTHORED_INSTRUCTION, "support.json": '{"limit":10}\n'}
    for name, text in sources.items():
        (repository / name).write_text(text)
    result = prepare_semantic_context(repository=repository, paths=sorted(sources), required_raw_paths=["required.md"],
        program_paths=["program.py"], objective="Authored capsule payload census", task_id="authored-census",
        output=repository / "semantic", worker_query="calculate_total outer Calculator calculate nested",
        worker_selector="duckdb-query-symbol-name@1", worker_capsule_limit=8, worker_max_bytes=32768,
        context_input_tokens=8192)
    raw = (repository / "semantic/worker-context.json").read_bytes()
    verified = load_semantic_worker_context(repository=repository, artifact="semantic/worker-context.json",
        expected_sha256=sha(raw), task_id="authored-census")
    if verified.encode() != raw:
        raise ValueError("real producer replay changed canonical bytes")
    payload = json.loads(raw)
    census = census_worker_payload(payload)
    model_prompt, advisory = render_model_prompt(prompt=raw.decode(), purpose="coding", workspace=repository,
        semantic_transport=True)
    census.update(case="authored-program-support", real_producer_replay_byte_exact=True,
        producer_capsules=result["capsules"], shape_qualification=_authored_shape_checks(),
        authored_wrapper_probe={"schema": "authored-real-runner-wrapper-census@1",
            "worker_payload_bytes": len(raw), "workspace_advisory_bytes": len(advisory.encode()),
            "complete_probe_bytes": len(model_prompt.encode()),
            "byte_equation_verified": len(raw) + len(advisory.encode()) == len(model_prompt.encode()),
            "actual_provider_input": False, "scope": "real workspace renderer around authored worker JSON only; no native capsule/Doctor/public-instruction composition"},
        original_sources_unchanged=all((repository / name).read_text() == text for name, text in sources.items()),
        models_invoked=0, containers_started=0, hidden_task_material_read=False)
    _save(output / "census.json", census)
    return census


def run_public_census(*, dataset: Path, task_name: str, output: Path):
    from . import terminal_profile_catalog as catalog
    from . import terminal_indexed_preparation as preparation
    selected = catalog.reviewed_profile(dataset=dataset, task_name=task_name)
    dataset = Path(dataset).resolve(strict=True)
    candidate = Path(output).absolute()
    if candidate == dataset or dataset in candidate.parents:
        raise ValueError("public census output must be outside the dataset")
    output = _new_output(output)
    repository = output / "repository"
    repository.mkdir()
    for item in selected["bindings"]["inputs"]:
        raw = catalog._read(Path(dataset).resolve() / task_name / "environment" / item["environment_path"], item["sha256"])
        destination = repository / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    for args in (("init", "-q"), ("add", "--all"), ("-c", "user.name=Public payload census", "-c",
            "user.email=census@example.invalid", "commit", "-qm", "Exact reviewed public inputs")):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True, timeout=30)
    instruction = output / "instruction.md"
    instruction.write_bytes(catalog._read(Path(dataset).resolve() / task_name / "instruction.md",
        selected["bindings"]["public_metadata"]["instruction.md"]))
    preparation.prepare(repository=repository, instruction=instruction, state=output / "state",
        task_profile=selected["profile"], disable_intent_autoencoder=True, semantic_context_profile="indexed-default@1")
    prepared = preparation.initial_context(state=output / "state")
    payload_path = repository / ".runtime/terminal-initial-context/semantic/worker-context.json"
    payload = json.loads(payload_path.read_bytes())
    result = census_worker_payload(payload)
    result.update(case=task_name, dataset_revision_reviewed=selected["dataset_revision_reviewed"],
        public_metadata_sha256=selected["bindings"]["public_metadata"],
        source_boundary=selected["source_boundary"], semantic_context_profile="indexed-default@1",
        producer_capsules=prepared["full_capsules"], models_invoked=0, containers_started=0,
        original_public_inputs_unchanged=all(sha(catalog._read(Path(dataset).resolve() / task_name / "environment" /
            item["environment_path"], item["sha256"])) == item["sha256"] for item in selected["bindings"]["inputs"]),
        hidden_task_material_read=False, official_reward=None)
    _save(output / "census.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args(argv)
    if bool(args.dataset) != bool(args.task):
        parser.error("public census requires both --dataset and --task")
    result = (run_public_census(dataset=args.dataset, task_name=args.task, output=args.output)
        if args.dataset else run_authored_census(output=args.output))
    print(json.dumps({key: result[key] for key in ("schema", "case", "payload_bytes", "selected_capsule_count", "models_invoked")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
