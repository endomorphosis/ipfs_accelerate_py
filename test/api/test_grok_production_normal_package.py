"""Normal-import and opt-in API checks for the reviewed owner integration.

Install this source at test/api in the genuine fresh main integration view.
These tests import actual project modules and inspect their origins and API;
they do not construct a daemon, owner, claim, lease, store or native actor.
Passing them grants no native, legal or Lean admission.
"""
import inspect
from pathlib import Path
import sys
import types

import ipfs_accelerate_py
import ipfs_accelerate_py.cli_runtime.grok_production_owner as core
import ipfs_accelerate_py.cli_runtime.grok_existing_phase_owner as factory
import ipfs_accelerate_py.cli_runtime.grok_owned_unix_socket as consumer
import ipfs_accelerate_py.cli_runtime.grok_production_owner_server as server
import ipfs_accelerate_py.cli_runtime.grok_production_handoff as handoff
import ipfs_accelerate_py.cli_runtime.grok_same_pid_owner_profile as profile
import ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon as daemon
import ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge as bridge


# Exact source hashes belong to the external QA request and AST seals. Keeping
# them out of this reusable API test lets unrelated daemon edits evolve normally.
REVIEWED_MODULES = (
    (core, "ipfs_accelerate_py/cli_runtime/grok_production_owner.py"),
    (factory, "ipfs_accelerate_py/cli_runtime/grok_existing_phase_owner.py"),
    (consumer, "ipfs_accelerate_py/cli_runtime/grok_owned_unix_socket.py"),
    (server, "ipfs_accelerate_py/cli_runtime/grok_production_owner_server.py"),
    (handoff, "ipfs_accelerate_py/cli_runtime/grok_production_handoff.py"),
    (profile, "ipfs_accelerate_py/cli_runtime/grok_same_pid_owner_profile.py"),
    (daemon, "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"),
    (bridge, "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py"),
)


def test_all_owner_modules_and_patched_callsites_have_genuine_normal_origins():
    root = Path(__file__).resolve(strict=True).parents[2]
    package = root/"ipfs_accelerate_py"
    root_type = type(ipfs_accelerate_py)
    if root_type is not types.ModuleType:
        assert root_type is vars(ipfs_accelerate_py).get("_IPFSAccelerateModule")
        assert issubclass(root_type, types.ModuleType)
        assert root_type.__module__ == "ipfs_accelerate_py"
        assert root_type.__name__ == "_IPFSAccelerateModule"
    assert sys.modules["ipfs_accelerate_py"] is ipfs_accelerate_py
    assert ipfs_accelerate_py.__file__ == str(package/"__init__.py")
    assert ipfs_accelerate_py.__spec__.origin == ipfs_accelerate_py.__file__
    assert Path(ipfs_accelerate_py.__file__).resolve(strict=True) == package/"__init__.py"
    for module, relative in REVIEWED_MODULES:
        path = root/relative
        assert type(module) is types.ModuleType
        assert sys.modules[module.__name__] is module
        assert module.__file__ == module.__spec__.origin == str(path)
        assert path.resolve(strict=True) == path
        assert module.__spec__.loader is not None


def test_existing_capability_factories_export_the_reviewed_api_without_invocation():
    classes = (
        (core, core.ProductionOwnerGuardClient),
        (factory, factory.ExistingProductionOwner),
        (consumer, consumer.OwnedUnixListener),
        (server, server.ProductionOwnerServer),
        (handoff, handoff.ProductionProviderHandoff),
        (profile, profile.ExistingSamePidOwnerProfile),
        (daemon, daemon.PortalImplementationDaemon),
        (daemon, daemon.DatabaseImplementationDaemon),
        (bridge, bridge.DatabasePortalExecutionBridge),
    )
    for module, value in classes:
        assert inspect.isclass(value) and value.__module__ == module.__name__
    functions = (
        (core, core.owner_guard_for),
        (factory, factory.bind_existing_production_owner),
        (consumer, consumer.open_owned_unix_listener),
        (profile, profile.bind_existing_database_callbacks),
    )
    for module, value in functions:
        assert inspect.isfunction(value) and value.__module__ == module.__name__
    parameters = inspect.signature(factory.bind_existing_production_owner).parameters
    assert set(parameters) == {
        "core", "database_owner", "attempt", "root", "phase", "disk", "source_roots",
        "source_pins", "source_locks", "launch_binding", "native_deadline_monotonic",
        "source_profile_pin",
    }
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY and
               parameter.default is inspect.Parameter.empty
               for parameter in parameters.values())
    assert inspect.signature(profile.bind_existing_database_callbacks).parameters.keys() == {
        "profile", "bridge",
    }


def test_both_real_constructor_hooks_are_keyword_only_and_disabled_by_default():
    hooks = (
        (daemon.PortalImplementationDaemon.__init__, "production_owner_handoff_factory"),
        (bridge.DatabasePortalExecutionBridge.__init__, "provider_handoff_binder"),
    )
    for constructor, name in hooks:
        parameter = inspect.signature(constructor).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None
    assert handoff.POLL_SECONDS == 0.05
    assert handoff.OWNER_FLAGS == (
        "--owner-context", "--owner-request-sha256", "--owner-allocation-session-id",
    )
