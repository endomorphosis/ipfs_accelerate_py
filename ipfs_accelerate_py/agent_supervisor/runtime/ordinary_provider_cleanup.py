"""Observe ordinary provider cleanup without signaling or removing resources.

Process exit during either /proc stat or thread-child reads is normal. Unknown
permission, identity, daemon or artifact changes still fail closed. This module
supplies observations, not protected-route authority or task acceptance.
"""
from pathlib import Path
import json
import os
import re
import subprocess

class OrdinaryProviderCleanupObserver:
    """Observe exact owned legacy reapers; never signal or remove anything."""

    def __init__(self, temporary_root, attempt):
        self.temporary_root, self.attempt = temporary_root, attempt
        self.owned, self.bindings, self.unresolved = {}, {}, None
        self.daemon_id = self.docker('info', '--format', '{{.ID}}')
        assert re.fullmatch(r'[0-9a-f-]{36}', self.daemon_id)
        save(attempt/'provider-docker-allocation.json', dict(
            schema='legal-ir-ordinary-provider-daemon-allocation@1',
            endpoint='unix:///var/run/docker.sock', daemon_id=self.daemon_id,
            image_acquisition=False, native_named_root_covers_daemon_storage=False,
            hard_daemon_quota=False, allocation_basis='existing image; new container metadata and logs',
            temporary_root=str(temporary_root), observed_container_count=0))

    @staticmethod
    def birth(pid):
        try:
            fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
            return int(fields[1]), int(fields[19]), fields[0]
        except (FileNotFoundError, ProcessLookupError):
            return None

    def docker(self, *arguments):
        result = subprocess.run(['/usr/bin/docker', '--host=unix:///var/run/docker.sock',
            '--config', '/nonexistent', *arguments], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=15,
            env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', LANG='C'))
        assert len(result.stdout) <= 4096 and len(result.stderr) <= 4096
        if result.returncode:
            if arguments[:2] == ('container', 'inspect') and 'No such container:' in result.stderr:
                return None
            raise RuntimeError('owned provider Docker observation unavailable')
        return result.stdout.strip()

    def attach(self, child):
        identity = self.birth(child.pid)
        assert identity is not None and identity[2] != 'Z'
        self.owned[child.pid] = identity[1]

    def observe(self):
        """Read child lists only below verified owned PID/start-time births."""
        queue, seen = list(self.owned), set()
        while queue:
            pid = queue.pop()
            if pid in seen:
                continue
            seen.add(pid)
            current = self.birth(pid)
            if current is None or current[1] != self.owned[pid] or current[2] == 'Z':
                continue
            for children in (Path('/proc')/str(pid)/'task').glob('*/children'):
                try:
                    child_pids = children.read_text().split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                parent_after = self.birth(pid)
                if parent_after is None or parent_after[1] != current[1]:
                    continue
                assert len(child_pids) <= 1024
                for value in child_pids:
                    assert value.isdecimal()
                    descendant = int(value)
                    birth = self.birth(descendant)
                    if birth is not None and birth[0] == pid and birth[2] != 'Z':
                        self.owned[descendant] = birth[1]
                        assert len(self.owned) <= 1024
                        queue.append(descendant)
            # Cmdline access is restricted to a verified owned birth; contents
            # are never persisted except the non-secret cleanup argument slots.
            descriptor = None
            try:
                # A retained proc directory binds these relative reads to the
                # observed process, rather than reopening a possibly reused PID.
                descriptor = os.open(Path('/proc')/str(pid), os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                def proc_read(name, limit):
                    fd = os.open(name, os.O_RDONLY|os.O_NOFOLLOW, dir_fd=descriptor)
                    with os.fdopen(fd, 'rb') as stream:
                        return stream.read(limit)
                bound = proc_read('stat', 16384).decode().rsplit(')', 1)[1].split()
                if int(bound[19]) != current[1] or bound[0] == 'Z':
                    continue
                raw = proc_read('cmdline', 65537)
            except (FileNotFoundError, ProcessLookupError):
                continue
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            flag = '--internal-legacy-docker-cleanup-watchdog'
            if len(raw) > 65536 or flag.encode() not in raw.rstrip(b'\0').split(b'\0'):
                continue
            argv = raw.decode('utf-8', errors='strict').rstrip('\0').split('\0')
            assert argv.count(flag) == 1
            def argument(name):
                assert argv.count(name) == 1
                return argv[argv.index(name)+1]
            provider, name = argument('--provider'), argument('--container-name')
            assert provider in {'grok', 'codex'}
            match = re.fullmatch(r'ipfs-accelerate-'+provider+r'-([0-9]+)-[0-9a-f]{32}', name)
            assert match is not None and int(match.group(1)) in self.owned
            lease_root = Path(argument('--lease-root'))
            assert lease_root.parent == self.temporary_root
            assert lease_root.name.startswith('asref-'+provider+'-container-')
            assert Path(argument('--cidfile')) == lease_root/'container.cid'
            assert argument('--docker-bin') in {'/usr/bin/docker', '/usr/local/bin/docker'}
            binding = dict(provider=provider, container_name=name, lease_root=str(lease_root),
                watchdog_pid=pid, watchdog_start_time_ticks=current[1],
                creator_pid=int(match.group(1)), endpoint='unix:///var/run/docker.sock',
                daemon_id=self.daemon_id)
            assert name not in self.bindings or self.bindings[name] == binding
            self.bindings[name] = binding
            assert len(self.bindings) <= 16
        save(self.attempt/'provider-docker-observations.json', dict(
            schema='legal-ir-owned-ordinary-provider-cleanup-observations@1',
            observation_only=True, foreign_cmdlines_read=False,
            bindings=list(self.bindings.values()), unresolved=self.unresolved))

    def complete(self):
        observations = []
        try:
            self.observe()
            assert self.docker('info', '--format', '{{.ID}}') == self.daemon_id
            assert not self.temporary_root.is_symlink()
            roots = [path for prefix in ('asref-grok-container-', 'asref-codex-container-')
                     for path in self.temporary_root.glob(prefix+'*')]
            for binding in self.bindings.values():
                current = self.birth(binding['watchdog_pid'])
                watchdog_live = (current is not None and
                    current[1] == binding['watchdog_start_time_ticks'] and current[2] != 'Z')
                absent = self.docker('container', 'inspect', '--format', '{{.Id}}', binding['container_name']) is None
                observations.append(dict(container_name=binding['container_name'],
                    container_absent=absent, lease_root_absent=not os.path.lexists(binding['lease_root']),
                    watchdog_not_live=not watchdog_live))
            complete = not self.unresolved and not roots and all(
                item['container_absent'] and item['lease_root_absent'] and item['watchdog_not_live']
                for item in observations)
            save(self.attempt/'provider-docker-cleanup.json', dict(
                schema='legal-ir-owned-ordinary-provider-cleanup@1', complete=bool(complete),
                observation_only=True, observed_binding_count=len(self.bindings),
                remaining_lease_directory_count=len(roots), observations=observations,
                claims_only_observed_containers=True, hard_daemon_quota=False))
            return bool(complete)
        except (OSError, RuntimeError, AssertionError, ValueError, IndexError, subprocess.TimeoutExpired):
            return False


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    assert len(raw) <= 1024*1024
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        os.chmod(temporary, 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

