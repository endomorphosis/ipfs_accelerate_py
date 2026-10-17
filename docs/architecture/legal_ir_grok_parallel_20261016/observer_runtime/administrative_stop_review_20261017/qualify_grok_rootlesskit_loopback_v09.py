"""Use an existing delegated RootlessKit profile for the exact v09 fixture.

No host networking fallback or host policy changes. Exec replaces this wrapper
so RootlessKit, its child, Python and Grok fit the existing four process slots.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pwd
import socket
import shutil
import sys

META = Path(__file__).resolve().parent
FIXTURE = META / 'qualify_grok_planning_toolset_loopback_v09.py'
ROOTLESSKIT = Path('/usr/bin/rootlesskit')
FIXTURE_ROOT = Path('/tmp/pytest-of-barberb/grk17f')
STATE = FIXTURE_ROOT / 'rootlesskit'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inner', action='store_true')
    parser.add_argument('--outer-netns')
    parser.add_argument('--outer-userns')
    args = parser.parse_args()
    args.controller = args.controller.resolve(strict=True)
    args.binary = args.binary.absolute()
    args.output = args.output.absolute()
    spec = importlib.util.spec_from_file_location('frozen_grok_fixture_v09', FIXTURE)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    if args.inner:
        actual_netns = os.readlink('/proc/self/ns/net')
        actual_userns = os.readlink('/proc/self/ns/user')
        if actual_netns == args.outer_netns or actual_userns == args.outer_userns:
            raise ValueError('fresh network and user namespaces are mandatory')
        interfaces = sorted(name for _, name in socket.if_nameindex())
        if interfaces != ['lo']:
            raise ValueError('network-none fixture must contain only loopback')
        ipv4_routes = Path('/proc/net/route').read_text().splitlines()
        if len(ipv4_routes) != 1:
            raise ValueError('network-none fixture unexpectedly has IPv4 routes')
        fixture.write_new(args.output / 'rootlesskit-isolation.json', {
            'schema': 'grok-rootlesskit-isolation-observation/v1',
            'network_namespace': actual_netns, 'outer_network_namespace': args.outer_netns,
            'user_namespace': actual_userns, 'outer_user_namespace': args.outer_userns,
            'interfaces': interfaces, 'ipv4_routes': [],
            'apparmor_label': Path('/proc/self/attr/current').read_text().strip(),
            'uid_map': Path('/proc/self/uid_map').read_text(),
            'gid_map': Path('/proc/self/gid_map').read_text(),
            'home': os.environ['HOME'], 'state_directory': str(STATE),
            'host_policy_changed': False, 'fallback_to_host_network': False,
            'real_model_calls': False, 'production_configuration_qualified': False})
        return fixture.inner(args)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    original_binary = args.binary
    if fixture.pin(original_binary)['sha256'] != args.binary_sha256:
        raise ValueError('original native Grok binary differs from pin')
    native_directory = args.output / 'native'
    native_directory.mkdir(mode=0o700)
    native_binary = native_directory / 'grok'
    descriptor = os.open(native_binary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o500)
    with os.fdopen(descriptor, 'wb') as target, original_binary.open('rb') as source:
        shutil.copyfileobj(source, target, 1048576)
        target.flush()
        os.fsync(target.fileno())
    if fixture.pin(native_binary)['sha256'] != args.binary_sha256:
        raise ValueError('owned native Grok copy differs from pin')
    args.binary = native_binary
    FIXTURE_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
    (FIXTURE_ROOT / 'cases').mkdir(mode=0o700)
    temporary = FIXTURE_ROOT / 'namespace-tmp'
    temporary.mkdir(mode=0o700)
    STATE.mkdir(mode=0o700, parents=False, exist_ok=False)
    environment = fixture.safe_import_environment(temporary, args.output / 'fixture-allocation.duckdb')
    account = pwd.getpwuid(os.getuid())
    environment.update(USER=account.pw_name, LOGNAME=account.pw_name)
    command = [str(ROOTLESSKIT), '--net=none', '--copy-up=/etc', '--state-dir', str(STATE),
        '/usr/bin/python3', '-B', str(Path(__file__).resolve()), '--inner',
        '--controller', str(args.controller), '--expected-head', args.expected_head,
        '--binary', str(args.binary), '--binary-sha256', args.binary_sha256,
        '--output', str(args.output), '--outer-netns', os.readlink('/proc/self/ns/net'),
        '--outer-userns', os.readlink('/proc/self/ns/user')]
    fixture.write_new(args.output / 'rootlesskit-launch-plan.json', {
        'schema': 'grok-rootlesskit-loopback-launch-plan/v1', 'argv': command,
        'home_unchanged': environment['HOME'] == os.environ['HOME'],
        'credentials_forwarded': False, 'host_policy_changed': False,
        'host_network_fallback': False, 'exec_replaces_wrapper': True})
    os.execve(ROOTLESSKIT, command, environment)
    raise AssertionError('exec unexpectedly returned')


if __name__ == '__main__':
    raise SystemExit(main())
