import os

import pytest


def _fake_result(*, success=True, exit_code=0, stdout="ok", stderr="", execution_time=0.01, error_message=None):
    class R:
        pass

    r = R()
    r.success = success
    r.exit_code = exit_code
    r.stdout = stdout
    r.stderr = stderr
    r.execution_time = execution_time
    r.error_message = error_message
    return r


@pytest.mark.parametrize(
    "task_type",
    [
        "docker.execute",
        "docker.execute_docker_container",
        "docker.hub",
        "docker.run",
    ],
)
def test_worker_docker_hub_task(monkeypatch, tmp_path, task_type):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    called = {}

    def fake_execute_docker_hub_container(*, image, command=None, entrypoint=None, environment=None, volumes=None, **kwargs):
        called["image"] = image
        called["command"] = command
        called["entrypoint"] = entrypoint
        called["environment"] = environment
        called["volumes"] = volumes
        called["kwargs"] = kwargs
        return _fake_result(stdout="hello")

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER", "1")

    import ipfs_accelerate_py.docker_executor as docker_executor

    monkeypatch.setattr(docker_executor, "execute_docker_hub_container", fake_execute_docker_hub_container)

    queue_path = str(tmp_path / "q.duckdb")
    queue = TaskQueue(queue_path)

    tid = queue.submit(
        task_type=task_type,
        model_name="docker",
        payload={
            "image": "python:3.11-slim",
            "command": "python -c print(123)",
            "entrypoint": "bash -lc",
            "environment": {"A": "B"},
            "volumes": {"/host": "/container"},
            "memory_limit": "256m",
            "cpu_limit": 0.5,
            "timeout": 12,
            "network_mode": "none",
            "working_dir": "/work",
            "read_only": True,
            "no_new_privileges": True,
            "user": "1000:1000",
        },
    )

    rc = run_worker(
        queue_path=queue_path,
        worker_id="w1",
        once=True,
        supported_task_types=[task_type],
    )
    assert rc == 0

    task = queue.get(tid)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["success"] is True
    assert task["result"]["stdout"] == "hello"

    assert called["image"] == "python:3.11-slim"
    assert called["command"] == ["python", "-c", "print(123)"]
    assert called["entrypoint"] == ["bash", "-lc"]
    assert called["environment"] == {"A": "B"}
    assert called["volumes"] == {"/host": "/container"}
    assert called["kwargs"]["memory_limit"] == "256m"
    assert called["kwargs"]["cpu_limit"] == 0.5
    assert called["kwargs"]["timeout"] == 12
    assert called["kwargs"]["network_mode"] == "none"
    assert called["kwargs"]["working_dir"] == "/work"
    assert called["kwargs"]["read_only"] is True
    assert called["kwargs"]["no_new_privileges"] is True
    assert called["kwargs"]["user"] == "1000:1000"


@pytest.mark.parametrize(
    "task_type",
    [
        "docker.github",
        "docker.github_repo",
        "docker.build_and_execute_github_repo",
    ],
)
def test_worker_docker_github_task(monkeypatch, tmp_path, task_type):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    called = {}

    def fake_build_and_execute_from_github(
        *,
        repo_url,
        branch="main",
        dockerfile_path="Dockerfile",
        context_path=".",
        command=None,
        entrypoint=None,
        environment=None,
        build_args=None,
        **kwargs,
    ):
        called["repo_url"] = repo_url
        called["branch"] = branch
        called["dockerfile_path"] = dockerfile_path
        called["context_path"] = context_path
        called["command"] = command
        called["entrypoint"] = entrypoint
        called["environment"] = environment
        called["build_args"] = build_args
        called["kwargs"] = kwargs
        return _fake_result(stdout="ran")

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER", "1")

    import ipfs_accelerate_py.docker_executor as docker_executor

    monkeypatch.setattr(docker_executor, "build_and_execute_from_github", fake_build_and_execute_from_github)

    queue_path = str(tmp_path / "q.duckdb")
    queue = TaskQueue(queue_path)

    tid = queue.submit(
        task_type=task_type,
        model_name="docker",
        payload={
            "repo_url": "https://github.com/user/repo",
            "branch": "main",
            "dockerfile_path": "Dockerfile",
            "context_path": ".",
            "command": ["python", "app.py"],
            "entrypoint": "bash -lc",
            "environment": {"X": "Y"},
            "build_args": {"ARG1": "V"},
            "timeout": 99,
            "memory_limit": "1g",
        },
    )

    rc = run_worker(
        queue_path=queue_path,
        worker_id="w1",
        once=True,
        supported_task_types=[task_type],
    )
    assert rc == 0

    task = queue.get(tid)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["stdout"] == "ran"

    assert called["repo_url"] == "https://github.com/user/repo"
    assert called["branch"] == "main"
    assert called["dockerfile_path"] == "Dockerfile"
    assert called["context_path"] == "."
    assert called["command"] == ["python", "app.py"]
    assert called["entrypoint"] == ["bash", "-lc"]
    assert called["environment"] == {"X": "Y"}
    assert called["build_args"] == {"ARG1": "V"}
    assert called["kwargs"]["timeout"] == 99
    assert called["kwargs"]["memory_limit"] == "1g"


def test_worker_docker_disabled_by_env(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    # Explicitly disable even if Docker daemon is available.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER", "0")

    queue_path = str(tmp_path / "q.duckdb")
    queue = TaskQueue(queue_path)

    tid = queue.submit(
        task_type="docker.execute",
        model_name="docker",
        payload={"image": "python:3.11-slim", "command": "echo hi"},
    )

    rc = run_worker(
        queue_path=queue_path,
        worker_id="w1",
        once=True,
        supported_task_types=["docker.execute"],
    )
    assert rc == 0

    task = queue.get(tid)
    assert task is not None
    assert task["status"] == "failed"
    assert "ENABLE_DOCKER" in (task["error"] or "")


def test_worker_docker_auto_enabled_when_daemon_available(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    # Auto-enable should kick in when daemon is reachable.
    monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER", raising=False)

    import ipfs_accelerate_py.p2p_tasks.worker as worker

    monkeypatch.setattr(worker, "_docker_daemon_available", lambda: True)

    called = {}

    def fake_execute_docker_hub_container(*, image, command=None, entrypoint=None, environment=None, volumes=None, **kwargs):
        called["image"] = image
        return _fake_result(stdout="hello", stderr="")

    import ipfs_accelerate_py.docker_executor as docker_executor

    monkeypatch.setattr(docker_executor, "execute_docker_hub_container", fake_execute_docker_hub_container)

    queue_path = str(tmp_path / "q.duckdb")
    queue = TaskQueue(queue_path)

    tid = queue.submit(
        task_type="docker.execute",
        model_name="docker",
        payload={"image": "python:3.11-slim", "command": "echo hi"},
    )

    rc = run_worker(
        queue_path=queue_path,
        worker_id="w1",
        once=True,
        supported_task_types=["docker.execute"],
    )
    assert rc == 0

    task = queue.get(tid)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["stdout"] == "hello"


def test_worker_shell_runs_in_docker(monkeypatch, tmp_path):
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    called = {}

    def fake_execute_docker_hub_container(*, image, command=None, **kwargs):
        called["image"] = image
        called["command"] = command
        called["kwargs"] = kwargs
        return _fake_result(stdout="hi\n")

    # If shell ever tries to execute on-host, fail the test.
    import ipfs_accelerate_py.p2p_tasks.worker as worker_mod

    def _no_subprocess_run(*args, **kwargs):
        raise AssertionError("shell must not call subprocess.run on host")

    monkeypatch.setattr(worker_mod.subprocess, "run", _no_subprocess_run)

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_SHELL", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_SHELL_IMAGE", "ubuntu:22.04")

    import ipfs_accelerate_py.docker_executor as docker_executor

    monkeypatch.setattr(docker_executor, "execute_docker_hub_container", fake_execute_docker_hub_container)

    queue_path = str(tmp_path / "q.duckdb")
    queue = TaskQueue(queue_path)
    tid = queue.submit(task_type="shell", model_name="shell", payload={"argv": ["echo", "hi"]})

    rc = run_worker(queue_path=queue_path, worker_id="w1", once=True, supported_task_types=["shell"])
    assert rc == 0

    task = queue.get(tid)
    assert task is not None
    assert task["status"] == "completed"
    assert task["result"]["stdout"].strip() == "hi"

    assert called["image"] == "ubuntu:22.04"
    assert called["command"] == ["/bin/sh", "-lc", "echo hi"]
    assert called["kwargs"]["network_mode"] == "none"
    assert called["kwargs"]["no_new_privileges"] is True
