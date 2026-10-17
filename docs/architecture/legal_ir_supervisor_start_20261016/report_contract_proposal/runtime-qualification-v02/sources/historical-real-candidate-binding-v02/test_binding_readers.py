"""Actual bounded reader controls, never Git/native/provider qualification.

Run with TMPDIR in a unique genuinely precharged external test directory.
The negative symlink/hardlink fixtures must not live in a native attempt tree.
No native owner, provider output, Git identity or acceptance handle is mocked.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("controlled_binding_v02_readers", Path(__file__).with_name("real_candidate_binding.py"))
binding = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = binding
spec.loader.exec_module(binding)
RAW = b"exact trusted implementation bytes\n"
SHA = hashlib.sha256(RAW).hexdigest()


class ReaderControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="legal-ir-binding-readers-")
        self.base = Path(self.temporary.name).absolute()
        self.path = self.base / "source.py"
        self.path.write_bytes(RAW)
        self.path.chmod(0o644)

    def tearDown(self):
        self.temporary.cleanup()

    def refuse(self, reason, function, *args, **kwargs):
        with self.assertRaisesRegex(binding.BindingRefusal, "^" + reason + "$", msg="Exact controlled refusal required"):
            function(*args, **kwargs)

    def test_trusted_0664_requires_and_accepts_exact_sha(self):
        self.path.chmod(0o664)
        self.assertEqual(binding.read_pinned_source(self.path, SHA), RAW)

    def test_trusted_0644_accepts_exact_sha(self):
        self.assertEqual(binding.read_pinned_source(self.path, SHA), RAW)

    def test_private_candidate_0644_unchanged(self):
        self.assertEqual(binding.read_regular(self.path), RAW)

    def test_private_candidate_0664_remains_forbidden(self):
        self.path.chmod(0o664)
        self.refuse("bounded_owned_regular_input_required", binding.read_regular, self.path)

    def test_trusted_world_write_forbidden_even_exact_sha(self):
        self.path.chmod(0o666)
        self.refuse("bounded_sha_pinned_trusted_source_required", binding.read_pinned_source, self.path, SHA)

    def test_private_world_write_forbidden(self):
        self.path.chmod(0o646)
        self.refuse("bounded_owned_regular_input_required", binding.read_regular, self.path)

    def test_trusted_hash_mismatch_refused(self):
        self.path.chmod(0o664)
        self.refuse("observer_implementation_source_changed", binding.read_pinned_source, self.path, "0" * 64)

    def test_trusted_missing_digest_refused_before_read(self):
        with patch.object(binding.os, "open", side_effect=AssertionError("invalid digest must not open a file")):
            self.refuse("implementation_digest_invalid", binding.read_pinned_source, self.path, None)

    def test_trusted_malformed_digest_refused_before_read(self):
        with patch.object(binding.os, "open", side_effect=AssertionError("invalid digest must not open a file")):
            self.refuse("implementation_digest_invalid", binding.read_pinned_source, self.path, "F" * 64)

    def test_trusted_symlink_refused(self):
        alias = self.base / "alias.py"
        alias.symlink_to(self.path)
        self.refuse("canonical_unaliased_path_required", binding.read_pinned_source, alias, SHA)

    def test_private_symlink_refused(self):
        alias = self.base / "alias.json"
        alias.symlink_to(self.path)
        self.refuse("canonical_unaliased_path_required", binding.read_regular, alias)

    def test_trusted_parent_alias_refused(self):
        alias = self.base / "aliased-parent"
        alias.symlink_to(self.base, target_is_directory=True)
        self.refuse("canonical_unaliased_path_required", binding.read_pinned_source, alias / self.path.name, SHA)

    def test_trusted_hardlink_refused(self):
        alias = self.base / "hardlink.py"
        os.link(self.path, alias)
        self.refuse("bounded_sha_pinned_trusted_source_required", binding.read_pinned_source, alias, SHA)

    def test_private_hardlink_refused(self):
        alias = self.base / "hardlink.json"
        os.link(self.path, alias)
        self.refuse("bounded_owned_regular_input_required", binding.read_regular, alias)

    def test_trusted_empty_file_refused(self):
        self.path.write_bytes(b"")
        self.refuse("bounded_sha_pinned_trusted_source_required", binding.read_pinned_source,
            self.path, hashlib.sha256(b"").hexdigest())

    def test_trusted_oversized_file_refused(self):
        self.refuse("bounded_sha_pinned_trusted_source_required", binding.read_pinned_source,
            self.path, SHA, maximum=len(RAW) - 1)

    def test_private_oversized_file_refused(self):
        self.refuse("bounded_owned_regular_input_required", binding.read_regular,
            self.path, maximum=len(RAW) - 1)

    def test_trusted_permission_race_refused_even_same_content(self):
        real_fstat, calls = binding.os.fstat, []
        def observed(descriptor):
            calls.append(descriptor)
            if len(calls) == 2:
                self.path.chmod(0o664)
            return real_fstat(descriptor)
        with patch.object(binding.os, "fstat", side_effect=observed):
            self.refuse("input_changed_during_read", binding.read_pinned_source, self.path, SHA)

    def test_private_permission_race_refused(self):
        real_fstat, calls = binding.os.fstat, []
        def observed(descriptor):
            calls.append(descriptor)
            if len(calls) == 2:
                self.path.chmod(0o600)
            return real_fstat(descriptor)
        with patch.object(binding.os, "fstat", side_effect=observed):
            self.refuse("input_changed_during_read", binding.read_regular, self.path)

    def test_trusted_foreign_owner_refused_from_normative_stat(self):
        self.refuse("bounded_sha_pinned_trusted_source_required", binding.check_regular_metadata,
            mode=stat.S_IFREG | 0o664, nlink=1, uid=987654, owner_uid=os.getuid(), trusted_source=True)

    def test_trusted_root_owner_permitted_from_normative_stat(self):
        self.assertIsNone(binding.check_regular_metadata(mode=stat.S_IFREG | 0o664, nlink=1,
            uid=0, owner_uid=os.getuid(), trusted_source=True))

    def test_trusted_nonregular_refused_from_normative_stat(self):
        self.refuse("bounded_sha_pinned_trusted_source_required", binding.check_regular_metadata,
            mode=stat.S_IFDIR | 0o664, nlink=1, uid=os.getuid(), owner_uid=os.getuid(), trusted_source=True)

    def test_reader_policy_must_be_actual_bool(self):
        self.refuse("reader_policy_must_be_explicit", binding.check_regular_metadata,
            mode=stat.S_IFREG | 0o664, nlink=1, uid=os.getuid(), owner_uid=os.getuid(), trusted_source=1)


if __name__ == "__main__":
    unittest.main()
