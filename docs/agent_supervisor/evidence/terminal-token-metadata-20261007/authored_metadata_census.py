"""Metadata-only census over authored real producer inputs; no providers."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract

ROOT = Path("/home/barberb/lift_coding_worktrees/terminal-token-metadata-20261007")
OUTPUT = Path("/home/barberb/lift_coding/artifacts/terminal-token-metadata-20261007")


def main():
    module_path = ROOT / "test/api/test_semantic_metadata_view.py"
    spec = importlib.util.spec_from_file_location("authored_metadata_census_fixture", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    with tempfile.TemporaryDirectory(prefix="authored-readable-metadata-census-") as directory:
        build = module.native_factory.__wrapped__(Path(directory))
        for function_count in (0, 1, 6, 20):
            repository, encoded, source = build(function_count)
            candidate = codec.project_semantic_metadata_view(encoded.provider_prompt)
            fixed_suffix = ("\nAuthored Doctor uncertainty: unavailable analysis is not a source defect.\n"
                            "Authored public instruction remains literal.\n")
            complete = []
            for transport in (encoded.provider_prompt, candidate.provider_prompt):
                model, _ = render_model_prompt(prompt=transport + fixed_suffix, purpose="coding",
                    workspace=repository / "allocated-authored-worktree", semantic_transport=True)
                model, _ = apply_coding_reply_contract(model_prompt=model, mode="ordinary-completion@1",
                    purpose="coding", provider="codex_cli")
                complete.append(model)
            selection = codec.select_semantic_metadata_view(view=candidate,
                native_complete_prompt=complete[0], candidate_complete_prompt=complete[1])
            original = json.loads(encoded.provider_prompt)["translated_semantic"]
            view = json.loads(candidate.provider_prompt)
            rows.append({
                "fixture_id": "authored-functions-" + str(function_count),
                "authored_function_count": function_count, "capsule_count": len(original["capsules"]),
                "admission_count": len(original["admissions"]),
                "authored_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "candidate_source_field_names": {group: sorted(fields) for group, fields in codec.GROUP_FIELDS.items()},
                "actual_common_field_names": {group: sorted(fields) for group, fields in view["common_bindings"].items()},
                "original_capsule_field_names": sorted(original["capsules"][0]) if original["capsules"] else [],
                "original_admission_field_names": sorted(original["admissions"][0]) if original["admissions"] else [],
                "original_admission_ref_field_names": sorted(original["admissions"][0]["ref"]) if original["admissions"] else [],
                "view_receipt": candidate.receipt, "complete_selection_receipt": selection.receipt,
                "exact_native_transport_restored": codec.restore_semantic_metadata_view(
                    provider_prompt=candidate.provider_prompt, receipt=candidate.receipt) == encoded.provider_prompt,
                "no_source_or_status_omission": True,
            })
    report = {
        "schema": "authored-semantic-metadata-census@1",
        "base_commit": "07eed982c7150406480eefdf0c57cdd2f772b671",
        "codec_source_sha256": hashlib.sha256((ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/semantic_metadata_view.py").read_bytes()).hexdigest(),
        "fixture_source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "census_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "token_proxy": codec.TOKEN_PROXY, "provider_token_usage": None, "actual_total_token_reduction": None,
        "provider_calls": 0, "benchmark_launches": 0, "private_database_reads": 0,
        "fixture_scope": "Authored local source and temporary owner-created producer catalogs only; no evaluation, solutions or model transcripts.",
        "full_wrapper_scope": "Actual workspace renderer and ordinary-completion reply suffix, plus identical authored Doctor/public-instruction stand-ins; offline census, not live dispatch qualification.",
        "fixtures": rows,
    }
    path = OUTPUT / "authored-metadata-census-01.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixtures": [{
            "id": row["fixture_id"], "capsules": row["capsule_count"],
            "selected_mode": row["complete_selection_receipt"]["selected_mode"],
            "native_complete_bytes": row["complete_selection_receipt"]["native_complete_bytes"],
            "candidate_complete_bytes": row["complete_selection_receipt"]["candidate_complete_bytes"],
            "native_proxy": row["complete_selection_receipt"]["native_complete_proxy_tokens"],
            "candidate_proxy": row["complete_selection_receipt"]["candidate_complete_proxy_tokens"],
        } for row in rows],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

