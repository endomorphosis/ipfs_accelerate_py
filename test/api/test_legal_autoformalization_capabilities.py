"""Preparatory capability inventory tests without models, auth, or Docker."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("lac_capabilities", REPO / "scripts/legal_autoformalization_capabilities.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def metadata():
    return {
        "registry": {"LEGAL_VERSIONS": ("legacy_v1", "legacy_v1_optimized", "current_v2"),
                     "LEARNED_FORMULA_VERSION": "source_conditioned_formula_v1"},
        "raw384": {"PINNED_REVISION": "1" * 40, "MAX_TOKENS": 512,
                   "_PINNED_ASSETS": {"tokenizer.json": (17, "2" * 64)}},
        "stages384": {"STAGES": ("raw384", "core384", "trained384"),
                      "RECEIPT_SCHEMA": "legal-native-conditioning-stage/v1"},
        "native768": {"PROFILE_ID": "distinct-native768-profile", "MODEL_REPO": "example/gte",
                      "MODEL_REV": "3" * 40, "CODE_REV": "4" * 40, "PROFILE": {"max_tokens": 8192},
                      "PUBLISHED_ASSETS": {("model", "tokenizer.json"): {"sha256": "5" * 64}}},
        "producer768": {"DIMENSION": 768}, "complete768": {"SCHEMA": "complete/v1"},
        "native4096": {"PROFILE_ID": "distinct-native4096-profile", "PRODUCER_ID": "native4096/v1",
                       "DIMENSION": 4096, "MODEL_SHA256": "6" * 64, "MODEL_CID": "baf-example", "MAX_TOKENS": 512},
        "retained_raw384": {"SCHEMA": "retained/v1"},
    }


def missing(name, _source):
    return {"name": name, "evidence_kind": "actual_local_subprocess_probe", "available": False,
            "version_verified": False, "blocker": name + "_executable_missing"}


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lac-capabilities-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.datasets = self.root / "datasets"
        self.datasets.mkdir()
        self.patch = patch.object(driver, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        for role, relative in driver.METADATA_SOURCES.items():
            path = self.datasets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(key + " = " + repr(value) for key, value in metadata()[role].items()) + "\n")
        subprocess.run(["git", "init", "-q", "--initial-branch=main", str(self.datasets)], check=True)
        subprocess.run(["git", "-C", str(self.datasets), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.datasets), "-c", "user.name=LAC Tests", "-c",
                        "user.email=lac-tests@example.invalid", "commit", "-qm", "metadata fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(self.datasets), "rev-parse", "HEAD"], text=True).strip()
        prior = self.root / "prior.json"
        prior.write_text("{}\n")
        pin = {"path": str(prior), "sha256": hashlib.sha256(prior.read_bytes()).hexdigest()}
        self.source = {"schema": driver.SOURCE_SCHEMA, "campaign_id": "lac-stage2-test",
            "datasets": {"root": str(self.datasets), "commit": commit, "expected_branch": "main"},
            "prior_bootstrap_completion": pin, "original_plan": pin,
            "required_tools": {"lean": True, "lake": True, "solvers": []},
            "docker": {"required": True, "socket": "unix:///var/run/docker.sock", "images": []},
            "observation_ttl_seconds": 900, "immutable_inputs": []}
        self.sources_path = self.root / "sources.json"
        self.write_source()

    def write_source(self):
        self.sources_path.write_text(json.dumps(self.source))

    def collect(self):
        with patch.object(driver, "probe_tool", side_effect=missing), \
                patch.object(driver, "probe_docker", return_value={"evidence_kind": "actual_local_image_inspect_only",
                    "runtime_available": False, "images": [], "blocker": "docker_executable_missing"}):
            return driver.collect(self.sources_path)

    def test_missing_tools_keep_inventory_valid_with_named_blockers(self):
        result = self.collect()
        validated = driver.validate_manifest(result, self.sources_path)
        self.assertTrue(validated["valid"])
        self.assertEqual(validated["capability_gates"]["status"], "blocked")
        self.assertIn("lean_executable_missing", validated["capability_gates"]["blockers"])
        self.assertFalse(result["authority"]["runtime_qualified"])

    def test_tool_missing_and_failed_version_are_distinct(self):
        self.assertEqual(driver.probe_tool("lean", self.source, which=lambda name: None)["blocker"], "lean_executable_missing")
        failure = {"returncode": 3, "state": "completed", "output_sha256": "7" * 64,
                   "output_bytes": 16, "text": "not a version"}
        result = driver.probe_tool("lean", self.source, which=lambda name: "/usr/bin/true",
                                  runner=lambda *args, **kwargs: failure)
        self.assertTrue(result["available"])
        self.assertEqual(result["blocker"], "lean_version_probe_failed")

    def test_configured_absent_executable_is_named_missing_tool(self):
        self.source["tool_paths"] = {"lean": str(self.root / "absent-lean")}
        result = driver.probe_tool("lean", self.source)
        self.assertFalse(result["available"])
        self.assertEqual(result["blocker"], "lean_executable_missing")

    def test_wrong_tool_version_cannot_impersonate_lean(self):
        response = {"returncode": 0, "state": "completed", "output_sha256": "7" * 64,
                    "output_bytes": 14, "text": "Python 3.12.3"}
        result = driver.probe_tool("lean", self.source, which=lambda name: "/usr/bin/true",
                                  runner=lambda *args, **kwargs: response)
        self.assertFalse(result["version_verified"])
        self.assertEqual(result["blocker"], "lean_version_probe_failed")

    def test_pinned_docker_image_missing_is_named_and_inspect_only(self):
        self.source["docker"]["images"] = [{"reference": "sha256:" + "1" * 64,
                                               "expected_id": "sha256:" + "1" * 64}]
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            return {"returncode": 1, "state": "completed", "text": "No such image"}

        result = driver.probe_docker(self.source, which=lambda name: "/usr/bin/docker", runner=runner)
        self.assertEqual(result["images"][0]["blocker"], "pinned_docker_image_missing_or_unobservable")
        self.assertIn("--host=unix:///var/run/docker.sock", calls[0])
        self.assertIn("inspect", calls[0])
        self.assertNotIn("pull", calls[0])
        self.assertNotIn("run", calls[0])

    def test_same_width_profiles_do_not_conflate_axes(self):
        result = self.collect()
        entries = result["immutable"]["runtime_profiles"]
        same_width = [row for row in entries if row["dimension"] == 384]
        self.assertEqual(len(same_width), 3)
        self.assertEqual(len({row["identity_sha256"] for row in same_width}), 3)
        self.assertEqual({row["representation"] for row in same_width}, {
            "native_gte_source_embedding", "source_parser_sparse_core_projection", "saved_learned_residual_projection"})
        self.assertTrue(all(row["width_implies_compatibility"] is False for row in entries))
        self.assertIsNone(entries[0]["dimension"])
        self.assertEqual(entries[0]["dimension_role"], "no_vector_input")

    def test_modified_source_at_same_commit_is_refused(self):
        path = self.datasets / driver.METADATA_SOURCES["registry"]
        path.write_bytes(path.read_bytes() + b"# changed effective source\n")
        with self.assertRaisesRegex(ValueError, "pinned commit"):
            driver.capture_immutable(self.sources_path)

    def test_declared_input_identity_mismatch_is_refused(self):
        path = self.root / "receipt.json"
        path.write_text("{}")
        self.source["immutable_inputs"] = [{"role": "prior_receipt", "path": str(path), "sha256": "0" * 64}]
        self.write_source()
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            driver.capture_immutable(self.sources_path)

    def test_idempotent_materialization_never_reruns_volatile_probes(self):
        with patch.object(driver, "probe_tool", side_effect=missing) as probes, \
                patch.object(driver, "probe_docker", return_value={"evidence_kind": "actual_local_image_inspect_only",
                    "runtime_available": False, "images": [], "blocker": "docker_executable_missing"}):
            first = driver.materialize(self.sources_path)
            before = driver.output_path().read_bytes()
            second = driver.materialize(self.sources_path)
        self.assertTrue(first["materialized"])
        self.assertFalse(second["materialized"])
        self.assertEqual(probes.call_count, 2)
        self.assertEqual(before, driver.output_path().read_bytes())

    def test_expired_probe_is_reported_without_invalidating_immutable_replay(self):
        result = self.collect()
        with patch.object(driver, "probe_tool", side_effect=AssertionError("no reprobe")):
            checked = driver.validate_manifest(result, self.sources_path,
                now_ns=result["observations"]["expires_at_epoch_nanoseconds"] + 1)
        self.assertTrue(checked["valid"])
        self.assertIn("live_capability_observations_expired_or_future", checked["capability_gates"]["blockers"])
        self.assertFalse(checked["live_probes_reexecuted"])

    def test_rehashed_axis_relabel_does_not_pass_immutable_replay(self):
        result = self.collect()
        result["immutable"]["runtime_profiles"][1]["representation"] = "saved_learned_residual_projection"
        result["manifest_sha256"] = driver.digest({key: value for key, value in result.items() if key != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "immutable replay"):
            driver.validate_manifest(result, self.sources_path)

    def test_output_and_input_aliases_are_refused(self):
        with self.assertRaisesRegex(ValueError, "exact LAC-S2-006"):
            driver.output_path(self.root / "somewhere-else.json")
        linked = self.root / "alias.json"
        linked.symlink_to(self.sources_path)
        with self.assertRaisesRegex(ValueError, "aliases"):
            driver.read_file(linked)

    def test_ast_metadata_does_not_execute_imports_or_calls(self):
        parsed = driver.constants(b"import forbidden\nraise RuntimeError('not executed')\nPROFILE_ID = 'safe'\nX = dangerous()\n")
        self.assertEqual(parsed, {"PROFILE_ID": "safe"})

    def reseal(self, result):
        result["manifest_sha256"] = driver.digest({key: value for key, value in result.items() if key != "manifest_sha256"})

    def test_rehashed_missing_tool_labels_cannot_clear_required_gates(self):
        self.source["docker"]["required"] = False
        self.write_source()
        result = self.collect()
        for row in result["observations"]["tools"]:
            row["blocker"] = None
        self.reseal(result)
        checked = driver.validate_manifest(result, self.sources_path)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["capability_gates"]["status"], "blocked")
        self.assertIn("lean_executable_missing", checked["capability_gates"]["blockers"])
        self.assertIn("lake_executable_missing", checked["capability_gates"]["blockers"])

    def test_rehashed_failed_version_labels_cannot_clear_required_gates(self):
        self.source["docker"]["required"] = False
        self.write_source()
        result = self.collect()
        for row in result["observations"]["tools"]:
            row.update(available=True, version_verified=False, state="completed", returncode=3, blocker=None)
        self.reseal(result)
        checked = driver.validate_manifest(result, self.sources_path)
        self.assertEqual(checked["capability_gates"]["status"], "blocked")
        self.assertIn("lean_version_probe_failed", checked["capability_gates"]["blockers"])

    def test_rehashed_missing_docker_label_cannot_clear_runtime_gate(self):
        result = self.collect()
        result["observations"]["docker"]["blocker"] = None
        self.reseal(result)
        checked = driver.validate_manifest(result, self.sources_path)
        self.assertIn("docker_executable_missing", checked["capability_gates"]["blockers"])

    def test_rehashed_missing_or_mismatched_image_labels_cannot_clear_gate(self):
        reference = "sha256:" + "1" * 64
        self.source["docker"]["images"] = [{"reference": reference, "expected_id": reference}]
        self.write_source()
        result = self.collect()
        docker = result["observations"]["docker"]
        docker.update(runtime_available=True, blocker=None, images=[{
            "reference": reference, "present": False, "identity_matches": False,
            "state": "completed", "returncode": 1, "blocker": None}])
        self.reseal(result)
        checked = driver.validate_manifest(result, self.sources_path)
        self.assertIn("pinned_docker_image_missing_or_unobservable:" + reference, checked["capability_gates"]["blockers"])
        docker["images"][0].update(present=True, returncode=0)
        self.reseal(result)
        checked = driver.validate_manifest(result, self.sources_path)
        self.assertIn("pinned_docker_image_identity_mismatch:" + reference, checked["capability_gates"]["blockers"])


if __name__ == "__main__":
    unittest.main()
