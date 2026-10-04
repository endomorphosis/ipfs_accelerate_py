"""Execute a complete current successor scan across four fresh processes.

The fixed closed setup contributes two historical training epochs and no scan
pages. All ten default pages and one opt-out comparison page execute here.
No fitting, worker dispatch, CUDA scan, proof authority or deployment is claimed.
Native receiving reopens only the new copy; the closed seed stays read-only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import qualify_codebase_inventory_resume as native
from . import qualify_codebase_inventory_scan as base
from . import source_successor_full_scan_fixture as fixture

SCHEMA = "codebase-full-successor-native-qualification@1"
ROOT = Path("/home/barberb/lift_coding")
ARTIFACTS = ROOT / "artifacts/codebase_ir_terminal_bench"
DEFAULT_SOURCE = Path(fixture.SOURCE_NAMESPACE)
DEFAULT_AUDIT = ARTIFACTS / "source-successor-qualification-20261003-02-readonly-audit-20261003-01.json"
DEFAULT_GUARD = ARTIFACTS / "source-successor-reader-guards-20261003-01/receipt.json"
DEFAULT_HOST = ARTIFACTS / "successor-expansion-resources-20261003-01/configuration.json"


def _pins(output, coordinator):
    names = set(coordinator._implementation()["files"])
    names.update({native.__name__, base.__name__, fixture.__name__,
                  "ipfs_datasets_py.duckdb_control.autoencoder_registry",
                  "ipfs_datasets_py.duckdb_control.codebase_catalog",
                  "ipfs_datasets_py.logic.software_contracts.codebase_resources",
                  "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
                  "benchmarks.agent_supervisor.container_coding.qualify_codebase_source_successor",
                  "benchmarks.agent_supervisor.container_coding.audit_codebase_source_successor",
                  "benchmarks.agent_supervisor.container_coding.qualify_codebase_full_successor"})
    destination = output / "producers"
    destination.mkdir()
    rows = []
    for name in sorted(names):
        path = Path(importlib.import_module(name).__file__).resolve()
        raw = path.read_bytes()
        copy = destination / (name + ".py")
        with copy.open("xb") as stream:
            stream.write(raw)
        copy.chmod(0o444)
        rows.append({"name": name, "path": str(path), "copy": copy.relative_to(output).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    record = {"schema": "codebase-full-successor-selected-producers@1", "files": rows,
              "scope": "listed_local_files_only", "execution_attestation": False}
    base._write(output / "generation-inputs.json", record)
    return record


def qualify(output, *, source=DEFAULT_SOURCE, audit=DEFAULT_AUDIT, guards=DEFAULT_GUARD,
            host_configuration=DEFAULT_HOST, overall_seconds=4200.0, reference_seconds=600.0):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as coordinator
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError

    output = Path(output).absolute()
    fixture.need(type(overall_seconds) in {int, float} and 0 < overall_seconds <= 7200,
                 "bounded explicit overall qualification deadline required")
    fixture.need(type(reference_seconds) in {int, float} and 0 < reference_seconds <= 600,
                 "bounded explicit reference qualification deadline required")
    started = time.monotonic()
    deadline = started + overall_seconds
    report = {"schema": SCHEMA, "qualified": False, "pid": os.getpid(), "phases": [], "controls": [],
        "scope": "complete_300_member_successor_cpu8d_scan_and_first32_opt_out_comparison",
        "new_fitting_epochs": 0, "inherited_setup_epochs": 2, "inherited_scan_pages": 0,
        "new_default_scan_pages": 0, "new_reference_scan_pages": 0, "fresh_processes": [],
        "post_setup_fit_attempt_count": 0, "inference_attempts_outside_pages": 0,
        "owner_open_count": 0, "fresh_process_count": 0, "complete_scan_qualified": False,
        "worker_dispatch_qualified": False, "cuda_qualified": False, "384d_qualified": False,
        "proof_authority": False, "source_execution_attested": False, "scan_execution_attested": False,
        "production_default_activated": False, "repository_code_executed": False,
        "kernel_resource_enforcement_claimed": False,
        "operation_deadline_seconds": {"default_selection": 120, "reference_selection": reference_seconds,
            "inference_page": 600, "completion_receiving": 600, "fresh_chunk": 900,
            "overall": overall_seconds, "reference_override_is_qualification_only": True}}
    index = registry = connection = scheduler = None
    staged = False

    def remaining(cap):
        value = deadline - time.monotonic()
        fixture.need(value > 0, "overall full successor qualification deadline exceeded")
        return min(cap, value)

    def options(cap=600):
        return {"scheduler": scheduler, "timeout_seconds": remaining(cap),
                "admission_timeout_seconds": min(30.0, remaining(cap)), "memory_mb": 1024}

    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - started
        if staged:
            base._progress(output / "progress.json", report)

    def phase(name, action):
        row = {"name": name, "status": "running"}
        report["phases"].append(row)
        progress()
        begin = time.monotonic()
        try:
            result = action()
        except BaseException:
            row.update(status="failed", elapsed_seconds=time.monotonic() - begin)
            progress()
            raise
        row.update(status="completed", elapsed_seconds=time.monotonic() - begin)
        progress()
        return result

    def persist(name, value):
        base._write(output / name, {"artifact_cid": value.artifact_cid, "value": value.to_dict()})

    def forbid_inference(*args, **kwargs):
        report["inference_attempts_outside_pages"] += 1
        raise AssertionError("successor selection or receiving attempted numerical inference")

    def no_inference():
        stack = ExitStack()
        stack.enter_context(base._patch(scan, "_worker", forbid_inference))
        stack.enter_context(base._patch(features, "infer_projection_features", forbid_inference))
        return stack

    def refuse(name, action, exception):
        begin = time.monotonic()
        try:
            action()
        except exception as error:
            report["controls"].append({"name": name, "refused": True,
                "error_type": type(error).__name__, "error": str(error),
                "elapsed_seconds": time.monotonic() - begin})
            fixture.assert_clean(scheduler)
            return
        raise AssertionError("refusal control was accepted: " + name)

    try:
        setup = phase("copy_independently_audited_closed_setup", lambda:
            fixture.stage_closed_successor_setup(source, audit, guards, output))
        staged = True
        report["materialization"] = {key: setup[key] for key in (
            "source_namespace", "source_pins", "source_archive_inventory_cid", "copied_files", "copied_bytes",
            "inherited_setup_epochs", "inherited_scan_pages", "new_fitting_epochs")}
        host_raw = fixture.read_absolute(host_configuration)
        host = fixture.inert.parse(host_raw)
        fixture.need(host["schema"] == "successor-expansion-host-configuration@1"
                     and host["auto_renew_leases"] is True and host["kernel_enforcement_claimed"] is False,
                     "explicit shared host authority differs")
        scheduler = fixture.shared_scheduler(host["state_path"], host["persisted_config"])
        report.update(scheduler_state_path=str(scheduler.state_path),
            scheduler_configuration=scheduler.config.persisted_dict(),
            host_configuration_path=str(Path(host_configuration).absolute()),
            host_configuration_pin=fixture.pin(host_raw), cpu_scan_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        base._write(output / "shared-host-configuration.json", host)
        selected = _pins(output, coordinator)
        repository = output / "repository"
        index, registry, connection = phase("open_new_copied_native_owners", lambda: fixture.open_materialized_successor(output))
        report["owner_open_count"] += 1
        frozen = fixture.checkpoint_states(registry, setup)
        before = fixture.owners(index, registry, connection)
        base._write(output / "owners-before.json", before)
        base._write(output / "checkpoint-states-before.json", frozen)
        original_cas = base._files(index.artifacts.root)
        base._write(output / "source-artifacts-before.json", original_cas)
        parent = setup["selected_models"]["root"]["version_id"]
        child = setup["selected_models"]["child"]["version_id"]
        transition_export = fixture.inert.parse(fixture.read_absolute(output / "seed-evidence/source-delta.json"))
        transition = delta.load_codebase_source_delta(index.artifacts, transition_export["artifact_cid"])
        fixture.need(fixture.inert.same(transition.to_dict()["current_head"], setup["current_head"]),
                     "copied current source-delta binding differs")
        report.update(previous_head=transition.to_dict()["previous_head"], current_head=setup["current_head"],
            parent_version_id=parent, child_version_id=child, checkpoint_states=frozen,
            source_delta_cid=transition.artifact_cid)
        persist("source-delta.json", transition)
        with fixture.no_fit(report):
            limits = scan.CodebaseScanResumeLimits(max_inventory_entries=512, page_entries=32)
            def select(optimized):
                with no_inference():
                    return coordinator.start_current_codebase_successor_scan(index, repository,
                        source_delta=transition, registry=registry, previous_version_id=parent,
                        version_id=child, limits=limits, optimized=optimized,
                        **options(120 if optimized else reference_seconds))
            selection = phase("select_fresh_default_successor_root", lambda: select(True))
            root = scan.load_codebase_scan_resume_root(index.artifacts, selection.to_dict()["root_cid"])
            fixture.need(len(root.to_dict()["members"]) == 300 and root.to_dict()["optimized"] is True,
                         "complete default fixture membership differs")
            persist("successor-selection.json", selection)
            persist("scan-root.json", root)
            report.update(root_cid=root.artifact_cid, successor_selection_cid=selection.artifact_cid)
            first = phase("infer_fresh_default_page_01", lambda: scan.scan_current_codebase_page(
                index, repository, root=root, registry=registry, **options()))
            report["new_default_scan_pages"] += 1
            persist("successor-parent-page-01.json", first)
            with no_inference():
                phase("refuse_incomplete_prefix_completion", lambda: refuse("incomplete_prefix_is_unknown",
                    lambda: scan.complete_current_codebase_scan(index, repository, root=root,
                        registry=registry, tail_page_cid=first.artifact_cid, **options()), scan.CodebaseScanResumeError))
            reference_selection = phase("select_fresh_opt_out_comparison_root", lambda: select(False))
            reference_root = scan.load_codebase_scan_resume_root(index.artifacts, reference_selection.to_dict()["root_cid"])
            persist("reference-successor-selection.json", reference_selection)
            persist("reference-scan-root.json", reference_root)
            reference_page = phase("infer_fresh_opt_out_first32", lambda: scan.scan_current_codebase_page(
                index, repository, root=reference_root, registry=registry, **options()))
            report["new_reference_scan_pages"] += 1
            persist("reference-prefix-page.json", reference_page)
            fixture.need(all(fixture.inert.observation_same(first.to_dict()[key], reference_page.to_dict()[key])
                             for key in ("entries", "coverage", "inference")),
                         "fresh optimized/opt-out numerical prefix differs")
            report["opt_out_equivalence"] = {"entries_coverage_and_inference_exact": True,
                "scope": "first32_ordered_members_only", "reference_page_cid": reference_page.artifact_cid,
                "optimized_page_cid": first.artifact_cid, "throughput_qualified": False}
            second = phase("infer_fresh_default_page_02", lambda: scan.scan_current_codebase_page(
                index, repository, root=root, registry=registry, cursor=first.next_cursor, **options()))
            report["new_default_scan_pages"] += 1
            persist("successor-parent-page-02.json", second)
            fixture.need(second.next_cursor is not None and second.next_cursor.to_dict()["next_offset"] == 64,
                         "parent cursor is not the complete fresh two-page prefix")
            warm = fixture.owners(index, registry, connection)
            fixture.need(fixture.inert.observation_same(before, warm)
                         and fixture.inert.same(fixture.checkpoint_states(registry, setup), frozen),
                         "parent scanning changed source/model owner or weights/Adam")
            base._write(output / "owners-after-parent-pages.json", warm)
            cursor = second.next_cursor.to_dict()
            fixture.close_materialized_successor(registry, connection)
            index = registry = connection = None
            base_generation = fixture.owner_generation(before)
            for number in range(1, 5):
                chunk = phase("fresh_process_resume_" + str(number), lambda number=number:
                    fixture.launch_resume_chunk(output, root_cid=root.artifact_cid, cursor=cursor,
                        version_id=child, expected_head=setup["current_head"], checkpoint_states=frozen,
                        expected_registry_owner_generation=base_generation + number, scheduler=scheduler,
                        run_number=number, timeout_seconds=remaining(900), max_pages=2))
                report["fresh_processes"].append(chunk)
                report["fresh_process_count"] += 1
                report["new_default_scan_pages"] += len(chunk["pages_created"])
                report["owner_open_count"] += 1
                cursor = chunk["next_cursor"]
                fixture.need((cursor is None) == (number == 4) and chunk["complete"] is (number == 4),
                             "fresh process completion occurred at a different chunk")
                progress()
            fixture.need(cursor is None and report["new_default_scan_pages"] == 10,
                         "fresh default scan did not cover ten pages")
            index, registry, connection = phase("cold_reopen_new_copy_after_four_processes",
                lambda: fixture.open_materialized_successor(output))
            report["owner_open_count"] += 1
            selection = coordinator.load_codebase_successor_scan(index.artifacts, selection.artifact_cid)
            root = scan.load_codebase_scan_resume_root(index.artifacts, root.artifact_cid)
            exported = fixture.inert.parse(fixture.read_absolute(output / "successor-scan-completion.json"))
            completion = scan.load_codebase_scan_resume_completion(index.artifacts, exported["artifact_cid"])
            with no_inference():
                phase("cold_receive_complete_successor_selection", lambda:
                    coordinator.validate_current_codebase_successor_scan(selection, index, repository,
                        registry=registry, **options(120)))
                phase("cold_receive_all_ten_pages_completion", lambda:
                    scan.validate_current_codebase_scan_completion(completion, index, repository,
                        root=root, registry=registry, **options()))
                cancelled = threading.Event(); cancelled.set()
                phase("refuse_precancelled_complete_receiving", lambda: refuse("precancelled_complete_receiving",
                    lambda: scan.validate_current_codebase_scan_completion(completion, index, repository,
                        root=root, registry=registry, cancel_event=cancelled, **options()), LeaseCancelledError))
            final_owners = fixture.owners(index, registry, connection)
            fixture.need(fixture.inert.observation_same(final_owners, fixture.expected_after_reopens(before, 5)),
                         "source/model changes exceed four fresh opens plus one cold reopen")
            fixture.need(fixture.inert.same(fixture.checkpoint_states(registry, setup), frozen),
                         "full scan/cold receiving changed checkpoint/Adam")
            coverage = completion.to_dict()["coverage"]
            fixture.need(coverage["inventory_entries"] == 300 and coverage["pages"] == 10
                         and sum(coverage["dispositions"].values()) == 300,
                         "completed scan lost explicit member dispositions")
            base._write(output / "owners-after-cold.json", final_owners)
            base._write(output / "checkpoint-states-after.json", frozen)
            ending = {row["path"]: row for row in base._files(index.artifacts.root)}
            fixture.need(all(ending[row["path"]] == row for row in original_cas), "copied original source CAS changed")
            native._require_pins(output, selected)
            fixture.need(registry.resolve_head(registry.get_version(child)["variant_id"], "main") is None,
                         "private child model was promoted")
            fixture.need(report["post_setup_fit_attempt_count"] == report["inference_attempts_outside_pages"] == 0
                         and report["new_fitting_epochs"] == 0 and report["new_reference_scan_pages"] == 1
                         and len({item["pid"] for item in report["fresh_processes"]}) == 4,
                         "fresh process/page/no-fit accounting differs")
            report.update(qualified=True, complete_scan_qualified=True, completion_cid=completion.artifact_cid,
                coverage=coverage, cold_receiving_verified=True, original_source_artifacts_preserved=True,
                selected_producers_unchanged=True, numerical_page_reuse=False, model_head_promoted=False,
                native_owner_unchanged_except_copy_path_relocation_and_six_opens=True)
    except BaseException as error:
        report.update(qualified=False, error_type=type(error).__name__, error=str(error))
    finally:
        if staged:
            try:
                fixture.close_materialized_successor(registry, connection)
                if scheduler is not None:
                    report["final_resources"] = fixture.assert_clean(scheduler,
                        owner_pids=[os.getpid(), *[item["pid"] for item in report["fresh_processes"]]])
                # This is a frozen read of the closed seed, with no native owner/Git.
                fixture.source_bundle(source, audit, guards)
                report["closed_source_archive_preserved_after_job"] = True
            except BaseException as error:
                report.update(qualified=False, cleanup_error_type=type(error).__name__, cleanup_error=str(error))
            report["recorded_seconds"] = time.monotonic() - started
            report["new_scan_pages_created"] = report["new_default_scan_pages"] + report["new_reference_scan_pages"]
            base._write(output / "result.json", report)
            progress()
        else:
            report["recorded_seconds"] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overall-seconds", type=float, default=4200.0)
    parser.add_argument("--reference-seconds", type=float, default=600.0)
    args = parser.parse_args()
    result = qualify(args.output, overall_seconds=args.overall_seconds, reference_seconds=args.reference_seconds)
    print(json.dumps({key: result.get(key) for key in (
        "qualified", "recorded_seconds", "new_default_scan_pages", "new_reference_scan_pages",
        "fresh_process_count", "error_type", "error", "cleanup_error")}, sort_keys=True), flush=True)
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
