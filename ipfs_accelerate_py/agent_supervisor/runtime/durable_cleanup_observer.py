"""Read-only managed-owner joins for the native Docker cleanup protocol.

This module grants neither provider dispatch nor removal authority.  The owner
retains the cleanup namespace before birth, protects exactly bound watchdogs
during STOP, and observes the producer's completed protocol.  Missing evidence
is uncertainty, including after every process has exited.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..control.lifecycle_orchestrator import (
    LifecycleProfile,
    ProcessIdentity,
    ProcessTreeSnapshot,
)

_DIRECTORY = "provider-cleanup-bindings"
_ENTRY = re.compile(r"[0-9a-f]{64}\.(?:json|authority|complete|lock|remove-dispatched)")
_WATCHDOG = "--internal-docker-cleanup-watchdog"
_LAUNCHER = "--internal-docker-cleanup-watchdog-launcher"
_REMOVAL = "--internal-docker-removal-issuer"
_REMOVAL_LAUNCHER = "--internal-docker-removal-issuer-launcher"
_LIFECYCLE = (
    "run_id", "profile_id", "target_id", "repository_root", "state_root",
    "run_root", "configuration_root",
)
_IDENTITY_FIELDS = {"device", "inode", "mode", "uid"}
_BINDING_FIELDS = {
    "schema", "binding_state", *_LIFECYCLE, "fencing_epoch", "runner_pid",
    "runner_start_ticks", "watchdog_pid", "watchdog_start_ticks", "boot_id",
    "provider", "docker_bin", "docker_device", "docker_inode", "docker_mode",
    "docker_uid", "container_name", "cleanup_root", "cleanup_root_identity",
    "lease_root", "docker_config", "cidfile", "provider_home", "prompt_path",
    "effect_observation", "create_command_id", "create_cwd",
    "create_environment_id", "termination_fence", "path_identities",
    "binding_path", "record_id",
}


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_uid


def _path_identity(metadata: os.stat_result) -> dict[str, int]:
    return dict(zip(("device", "inode", "mode", "uid"), (
        metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode), metadata.st_uid,
    ), strict=True))


def _absent(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise ValueError("durable cleanup path absence is unavailable") from exc
    return False


def _assert_private_directory(path: Path, descriptor: int) -> None:
    opened = os.fstat(descriptor)
    current = os.lstat(path)
    if (
        path.resolve(strict=True) != path.absolute()
        or _directory_identity(opened) != _directory_identity(current)
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        raise ValueError("durable cleanup namespace is unavailable or aliased")


def has_cleanup_custody(run_root: Path) -> bool:
    """Conservatively detect any cleanup evidence without creating a namespace.

    Process-only owners use this as an explicit refusal gate until they retain
    a directory before dispatch.  An unavailable or aliased path is UNKNOWN.
    """
    path = Path(run_root).absolute() / _DIRECTORY
    try:
        if path.resolve(strict=False) != path:
            raise ValueError("durable cleanup namespace is aliased")
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError("durable cleanup namespace is unavailable") from exc
    try:
        _assert_private_directory(path, descriptor)
        with os.scandir(descriptor) as entries:
            present = next(entries, None) is not None
        _assert_private_directory(path, descriptor)
        return present
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class CleanupDirectoryAnchor:
    descriptor: int
    path: Path
    identity: tuple[int, int, int, int]
    _closed: bool = field(default=False, init=False, compare=False, repr=False)

    @classmethod
    def open_before_launch(cls, run_root: Path) -> CleanupDirectoryAnchor:
        root = Path(run_root).absolute()
        if root.resolve(strict=False) != root:
            raise ValueError("managed cleanup run root is aliased")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.resolve(strict=True) != root:
            raise ValueError("managed cleanup run root is aliased")
        path = root / _DIRECTORY
        path.mkdir(mode=0o700, exist_ok=True)
        descriptor = os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            _assert_private_directory(path, descriptor)
            return cls(descriptor, path, _directory_identity(os.fstat(descriptor)))
        except BaseException:
            os.close(descriptor)
            raise

    def validate(self) -> None:
        if self._closed:
            raise ValueError("retained cleanup namespace is closed")
        _assert_private_directory(self.path, self.descriptor)
        if _directory_identity(os.fstat(self.descriptor)) != self.identity:
            raise ValueError("retained cleanup namespace identity changed")

    def close(self) -> None:
        if self._closed:
            return
        # Mark first: even an interrupted close must not later close a reused fd.
        object.__setattr__(self, "_closed", True)
        os.close(self.descriptor)


def process_birth(identity: ProcessIdentity) -> tuple[int, int, str]:
    return identity.pid, identity.start_time_ticks, identity.boot_id


class ManagedCleanupObserver:
    """Track native bindings monotonically, observing completion without writes."""

    def __init__(
        self, profile: LifecycleProfile, anchor: CleanupDirectoryAnchor,
        *, fencing_epoch: int = 0,
    ) -> None:
        if (
            not isinstance(anchor, CleanupDirectoryAnchor)
            or anchor.path != Path(profile.run_root).absolute() / _DIRECTORY
            or type(fencing_epoch) is not int or fencing_epoch < 0
        ):
            raise ValueError("managed cleanup owner binding is invalid")
        anchor.validate()
        self.profile = profile
        self.anchor = anchor
        self.fencing_epoch = fencing_epoch
        self.records: dict[str, dict[str, object]] = {}
        self.identities: dict[str, dict[str, int]] = {}
        self.protected: set[tuple[int, int, str]] = set()
        self.failed = False
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def read(self, path: Path) -> dict[str, object] | None:
        # Structural read-only adapter for the producer's completion validator;
        # it cannot acquire a mutation lock, create a store, or replay cleanup.
        from . import grok_cli_runner as producer

        self.anchor.validate()
        if path.parent != self.anchor.path:
            raise ValueError("cleanup observation escaped its retained namespace")
        value = producer._read_private_control_record(
            path.parent, path.name, directory_fd=self.anchor.descriptor,
        )
        self.anchor.validate()
        return value

    def _entries(self) -> tuple[str, ...]:
        self.anchor.validate()
        before = os.fstat(self.anchor.descriptor)
        collected: list[str] = []
        with os.scandir(self.anchor.descriptor) as entries:
            for entry in entries:
                collected.append(entry.name)
                if len(collected) > 640:
                    raise ValueError("durable cleanup entry capacity exceeded")
        names = tuple(sorted(collected))
        after = os.fstat(self.anchor.descriptor)
        self.anchor.validate()
        if (
            len(names) > 640 or any(_ENTRY.fullmatch(name) is None for name in names)
            or (before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError("durable cleanup record set changed or is invalid")
        for name in names:
            metadata = os.stat(name, dir_fd=self.anchor.descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                # The producer can temporarily retain a two-name hard link at
                # retirement.  That crash window is incomplete, not absence.
                raise ValueError("durable cleanup entry is not an exact private inode")
        return names

    def _validate_record(self, value: dict[str, object], stem: str) -> None:
        from . import grok_cli_runner as producer

        provider = value.get("provider")
        name = str(value.get("container_name") or "")
        path = self.anchor.path / (stem + ".json")
        body = {key: item for key, item in value.items() if key != "record_id"}
        observation = value.get("effect_observation")
        if (
            set(value) != _BINDING_FIELDS
            or value.get("schema") != producer._DOCKER_CLEANUP_BINDING_SCHEMA
            or value.get("record_id") != producer._effect_receipt_identity(body)
            or any(value.get(key) != getattr(self.profile, key) for key in _LIFECYCLE)
            or value.get("fencing_epoch") != self.fencing_epoch
            or value.get("boot_id") != self.boot_id
            or any(type(value.get(key)) is not int or int(value[key]) <= 0 for key in (
                "runner_pid", "runner_start_ticks", "watchdog_pid", "watchdog_start_ticks",
            ))
            or provider not in {"codex", "grok"}
            or producer._DOCKER_CONTAINER_NAME_RE.fullmatch(name) is None
            or not name.startswith(f"ipfs-accelerate-{provider}-")
            or hashlib.sha256(name.encode("ascii")).hexdigest() != stem
            or value.get("binding_path") != str(path)
            or not isinstance(observation, dict)
            or set(observation) not in (set(), {
                "logical_attempt_id", "provider_attempt_store", "provider_attempt_store_identity",
            })
            or any(not isinstance(item, str) or not item for item in observation.values())
            or (observation and provider != "codex")
        ):
            raise ValueError("durable cleanup record owner differs")
        docker = Path(str(value.get("docker_bin") or ""))
        metadata = docker.stat()
        if (
            docker not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
            or docker.resolve(strict=True) != docker
            or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
            or metadata.st_mode & 0o022 or not os.access(docker, os.X_OK)
            or any(value.get("docker_" + field) != actual for field, actual in (
                ("device", metadata.st_dev), ("inode", metadata.st_ino),
                ("mode", metadata.st_mode), ("uid", metadata.st_uid),
            ))
        ):
            raise ValueError("durable cleanup Docker executable identity differs")
        lease, home, prompt = (Path(str(value[key])) for key in (
            "lease_root", "provider_home", "prompt_path",
        ))
        producer._validated_docker_cleanup_root(
            lease_root=lease, provider_home=home, prompt_path=prompt,
            expected_root=Path(str(value["cleanup_root"])),
            expected_identity=value["cleanup_root_identity"],
        )
        identities = value.get("path_identities")
        if (
            not lease.name.startswith(f"asref-{provider}-container-")
            or not home.name.startswith(f"asref-{provider}-home-")
            or not prompt.name.startswith("asref-grok-prompt-")
            or value.get("docker_config") != str(lease / "docker-config")
            or value.get("cidfile") != str(lease / "container.cid")
            or not isinstance(identities, dict)
            or set(identities) != {"docker_config", "lease_root", "provider_home", "prompt_path"}
            or any(
                not isinstance(item, dict) or set(item) != _IDENTITY_FIELDS
                or any(type(part) is not int or part < 0 for part in item.values())
                or item.get("uid") != os.geteuid()
                or not (stat.S_ISREG(item["mode"]) if key == "prompt_path"
                        else stat.S_ISDIR(item["mode"]))
                for key, item in identities.items()
            )
        ):
            raise ValueError("durable cleanup resource bindings differ")
        state = value.get("binding_state")
        fence = value.get("termination_fence")
        if not isinstance(fence, dict) or state not in {"prepared_no_dispatch", "command_bound"}:
            raise ValueError("durable cleanup state is invalid")
        if state == "prepared_no_dispatch":
            if fence or any(value.get(key) != "" for key in (
                "create_command_id", "create_cwd", "create_environment_id",
            )):
                raise ValueError("unprepared cleanup has command authority")
        elif (
            any(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value[key])) is None
                for key in ("create_command_id", "create_environment_id"))
            or not Path(str(value["create_cwd"])).is_absolute()
        ):
            raise ValueError("durable cleanup command binding is incomplete")
        if fence:
            producer._validated_docker_termination_fence(
                fence, provider=str(provider), container_name=name,
            )

    def _scan(self) -> tuple[str, ...]:
        if self.failed:
            raise ValueError("durable cleanup observation previously became unknown")
        try:
            return self._scan_checked()
        except (KeyError, OSError, TypeError, ValueError):
            self.failed = True
            raise

    def _scan_checked(self) -> tuple[str, ...]:
        names = self._entries()
        stems = {name.split(".", 1)[0] for name in names}
        if len(stems) > 128:
            raise ValueError("durable cleanup capacity exceeded")
        observed: set[str] = set()
        for stem in sorted(stems):
            active = stem + ".json"
            retired = stem + ".authority"
            if active in names and retired in names:
                raise ValueError("durable cleanup retirement is incomplete")
            selected = active if active in names else retired if retired in names else ""
            if not selected:
                # Orphan locks, journals and self-hashed completions cannot
                # prove that an effect never existed or has been retired.
                raise ValueError("durable cleanup lacks retained binding authority")
            path = self.anchor.path / selected
            before = os.stat(selected, dir_fd=self.anchor.descriptor, follow_symlinks=False)
            value = self.read(path)
            after = os.stat(selected, dir_fd=self.anchor.descriptor, follow_symlinks=False)
            if value is None or _path_identity(before) != _path_identity(after):
                raise ValueError("durable cleanup binding changed during observation")
            self._validate_record(value, stem)
            prior = self.records.get(stem)
            if prior is not None:
                evolving = {"binding_state", "create_command_id", "create_cwd",
                            "create_environment_id", "termination_fence", "record_id"}
                if (
                    any(value[key] != prior[key] for key in prior.keys() - evolving)
                    or (prior["binding_state"] == "command_bound" and any(
                        value[key] != prior[key] for key in (
                            "binding_state", "create_command_id", "create_cwd", "create_environment_id",
                        )))
                    or (prior["termination_fence"] and value["termination_fence"] != prior["termination_fence"])
                    or (prior == value and self.identities[stem] != _path_identity(after))
                ):
                    raise ValueError("durable cleanup record changed ownership or regressed")
            self.records[stem] = value
            self.identities[stem] = _path_identity(after)
            observed.add(stem)
        if set(self.records) != observed:
            raise ValueError("previously observed durable cleanup authority disappeared")
        if self._entries() != names:
            raise ValueError("durable cleanup namespace changed during observation")
        return names

    def observe(self, tree: ProcessTreeSnapshot) -> frozenset[tuple[int, int, str]]:
        """Admit native watchdog births and conservatively retain their children."""
        try:
            return self._observe(tree)
        except (KeyError, OSError, TypeError, ValueError):
            self.failed = True
            raise

    def _observe(self, tree: ProcessTreeSnapshot) -> frozenset[tuple[int, int, str]]:
        from . import grok_cli_runner as producer

        self._scan()
        if tree.profile_id != self.profile.profile_id or tree.run_id != self.profile.run_id:
            raise ValueError("managed cleanup tree owner differs")
        for member in tree.members:
            argv = member.argv
            if _REMOVAL in argv or _REMOVAL_LAUNCHER in argv:
                # The once-only rm issuer double-forks before dispatch.  It may
                # first appear as PPID 1 rather than as a watchdog descendant.
                if (
                    argv.count(_REMOVAL) != 1 or argv.count(_REMOVAL_LAUNCHER) != 1
                    or argv.index(_REMOVAL_LAUNCHER) >= argv.index(_REMOVAL)
                    or argv.count("--binding-path") != 1
                    or argv.index("--binding-path") + 1 >= len(argv)
                    or member.executable != str(Path(sys.executable).resolve(strict=True))
                    or member.parent_pid != 1
                ):
                    raise ValueError("detached cleanup issuer command is invalid")
                binding_path = Path(argv[argv.index("--binding-path") + 1])
                record = self.records.get(binding_path.stem)
                if (
                    record is None or record["binding_path"] != str(binding_path)
                    or any(getattr(member, key) != record[key] for key in _LIFECYCLE)
                    or member.fencing_epoch != self.fencing_epoch
                ):
                    raise ValueError("detached cleanup issuer lacks retained authority")
                dispatch = producer._validated_docker_removal_dispatch(
                    self.read(binding_path.with_suffix(".remove-dispatched")) or {},
                    binding_path=binding_path, binding_record=record,
                    termination_fence=record["termination_fence"],
                )
                expected_birth = {"pid": member.pid, "start_time_ticks": member.start_time_ticks,
                                  "boot_id": member.boot_id, "parent_pid": member.parent_pid}
                if dispatch["issuer_process_birth"] != expected_birth:
                    raise ValueError("detached cleanup issuer process birth differs")
                self.protected.add(process_birth(member))
                continue
            if _WATCHDOG not in argv and _LAUNCHER not in argv:
                continue
            matches = [record for record in self.records.values() if (
                record["watchdog_pid"] == member.pid
                and record["watchdog_start_ticks"] == member.start_time_ticks
                and record["boot_id"] == member.boot_id
            )]
            if len(matches) != 1:
                raise ValueError("native watchdog lacks an exact durable birth")
            record = matches[0]
            if (
                argv.count(_WATCHDOG) != 1 or argv.count(_LAUNCHER) != 1
                or argv.index(_LAUNCHER) >= argv.index(_WATCHDOG)
                or member.executable != str(Path(sys.executable).resolve(strict=True))
                or any(getattr(member, key) != record[key] for key in _LIFECYCLE)
                or member.fencing_epoch != self.fencing_epoch
            ):
                raise ValueError("native watchdog command identity differs")
            for flag, field in (
                ("--provider", "provider"), ("--docker-bin", "docker_bin"),
                ("--container-name", "container_name"), ("--lease-root", "lease_root"),
                ("--cidfile", "cidfile"), ("--provider-home", "provider_home"),
                ("--prompt-path", "prompt_path"), ("--cleanup-binding-record", "binding_path"),
                ("--runner-pid", "runner_pid"), ("--runner-start-ticks", "runner_start_ticks"),
            ):
                if (argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv)
                        or argv[argv.index(flag) + 1] != str(record[field])):
                    raise ValueError("native watchdog launch binding differs")
            self.protected.add(process_birth(member))
        # Retain any already observed child birth even after its watchdog dies
        # and the child is reparented.  This never grants signal authority.
        while True:
            parents = {item.pid for item in tree.members if process_birth(item) in self.protected}
            additions = {process_birth(item) for item in tree.members if item.parent_pid in parents}
            if additions <= self.protected:
                break
            self.protected.update(additions)
        return frozenset(self.protected)

    def _terminal(self, record: Mapping[str, object], completion: Mapping[str, object]) -> object:
        from . import grok_cli_runner as producer
        from ..control.provider_attempt_store import DurableProviderAttemptCAS

        observation = record["effect_observation"]
        intent = completion["cleanup_intent"]
        authority = producer._cleanup_intent_terminal_authority(intent)
        if not observation:
            if authority is not None:
                raise ValueError("unscoped cleanup has a forged scoped completion")
            return None
        store = DurableProviderAttemptCAS(
            observation["provider_attempt_store"],
            expected_directory_identity=observation["provider_attempt_store_identity"],
            create_if_missing=False,
        )
        terminal = store.observe(observation["logical_attempt_id"])
        if terminal is None:
            raise ValueError("terminal cleanup CAS is absent")
        producer._admit_terminal_cleanup_authority(
            launch_receipt=terminal.effect_launch_receipt,
            terminal_observer=store, terminal_reservation=terminal,
        )
        cleanup = terminal.effect_launch_receipt.get("cleanup_receipt", {})
        fence = record["termination_fence"]
        if (
            terminal.terminal_cleanup_authority != authority
            or cleanup.get("watchdog_pid") != record["watchdog_pid"]
            or cleanup.get("watchdog_start_ticks") != record["watchdog_start_ticks"]
            or cleanup.get("lease_root") != record["lease_root"]
            or cleanup.get("docker_config") != record["docker_config"]
            or any(cleanup.get(field) != record[field] for field in (
                "cidfile", "provider_home", "prompt_path",
            ))
            or terminal.effect_launch_receipt.get("container_name") != record["container_name"]
            or (fence and (
                str(terminal.effect_launch_receipt.get("container_id", "")).removeprefix("sha256:")
                != fence["container_id"]
                or terminal.effect_launch_receipt.get("image_id") != fence["image_id"]
            ))
            or not producer._cleanup_progress_matches(
                terminal.terminal_cleanup_progress, intent=intent,
                completion_id=str(completion["completion_id"]),
            )
        ):
            raise ValueError("terminal cleanup CAS does not join exact completion")
        return terminal

    def _completion(self, stem: str) -> tuple[dict[str, object], object]:
        from . import grok_cli_runner as producer

        record = self.records[stem]
        if record["binding_state"] == "command_bound" and not record["termination_fence"]:
            # Name-only absence cannot close an issued create whose recorded
            # outcome is unknown.  Owner create-journal recovery is not yet an
            # admitted part of this bounded observation path.
            raise ValueError("command-bound cleanup lacks an exact termination fence")
        binding_path = self.anchor.path / (stem + ".json")
        completion = self.read(binding_path.with_suffix(".complete"))
        if completion is None or not isinstance(completion.get("cleanup_intent"), dict):
            raise ValueError("durable cleanup completion is absent")
        terminal = self._terminal(record, completion)
        expected = producer._cleanup_completion_value(
            binding_path=binding_path, binding_identity=self.identities[stem],
            binding_record=record,
            terminal_cleanup_authority=producer._cleanup_intent_terminal_authority(
                completion["cleanup_intent"],
            ),
            binding_lock=self,
        )
        if completion != expected or not _absent(binding_path):
            raise ValueError("durable cleanup completion or authority retirement differs")
        for item in completion["resources"]:
            path = Path(item["path"])
            quarantine, owned, marker, _ = producer._cleanup_path_quarantine(
                path, directory=item["directory"], identity=item["identity"],
            )
            if any(not _absent(candidate) for candidate in (path, quarantine, owned, marker)):
                raise ValueError("durable cleanup resource retirement is incomplete")
        return completion, terminal

    def complete(self, *, deadline: float) -> bool:
        """Require stable persisted completion and fresh Docker/kernel absence."""
        from . import grok_cli_runner as producer

        try:
            names = self._scan()
            completions = {stem: self._completion(stem) for stem in self.records}
            for stem, record in self.records.items():
                if time.monotonic() >= deadline:
                    return False
                # issue_removal=False is essential: STOP owns observation only.
                # The retained private directory has no ambient Docker config;
                # its descriptor remains bound through the inspection commands.
                producer._remove_exact_docker_container(
                    docker_bin=str(record["docker_bin"]),
                    docker_config=Path(f"/proc/self/fd/{self.anchor.descriptor}"),
                    container_name=str(record["container_name"]),
                    settle_for_creation=False, deadline=deadline,
                    termination_fence=record["termination_fence"] or None,
                    issue_removal=False, pass_fds=(self.anchor.descriptor,),
                )
            if time.monotonic() >= deadline or self._scan() != names:
                return False
            return all(self._completion(stem) == observed for stem, observed in completions.items())
        except (KeyError, OSError, TypeError, ValueError):
            return False
