import anyio


def test_submit_docker_hub_task_builds_payload(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_docker_hub_task

    captured = {}

    async def fake_submit_task(*, remote, task_type, model_name, payload):
        captured["remote"] = remote
        captured["task_type"] = task_type
        captured["model_name"] = model_name
        captured["payload"] = payload
        return "tid123"

    import ipfs_accelerate_py.p2p_tasks.client as client

    monkeypatch.setattr(client, "submit_task", fake_submit_task)

    rq = RemoteQueue(peer_id="p", multiaddr="/ip4/127.0.0.1/tcp/1/p2p/QmX")

    async def _do():
        tid = await submit_docker_hub_task(
            remote=rq,
            image="python:3.11-slim",
            command=["python", "-c", "print(1)"] ,
            entrypoint="bash -lc",
            environment={"A": 1},
            volumes={"/h": "/c"},
            timeout=12,
            memory_limit="256m",
        )
        assert tid == "tid123"

    anyio.run(_do, backend="trio")

    assert captured["task_type"] == "docker.execute"
    assert captured["model_name"] == "docker"
    assert captured["payload"]["image"] == "python:3.11-slim"
    assert captured["payload"]["command"] == ["python", "-c", "print(1)"]
    assert captured["payload"]["entrypoint"] == "bash -lc"
    assert captured["payload"]["environment"] == {"A": "1"}
    assert captured["payload"]["volumes"] == {"/h": "/c"}
    assert captured["payload"]["timeout"] == 12
    assert captured["payload"]["memory_limit"] == "256m"


def test_submit_docker_github_task_builds_payload(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_docker_github_task

    captured = {}

    async def fake_submit_task(*, remote, task_type, model_name, payload):
        captured["task_type"] = task_type
        captured["model_name"] = model_name
        captured["payload"] = payload
        return "tid999"

    import ipfs_accelerate_py.p2p_tasks.client as client

    monkeypatch.setattr(client, "submit_task", fake_submit_task)

    rq = RemoteQueue(peer_id="p", multiaddr="/ip4/127.0.0.1/tcp/1/p2p/QmX")

    async def _do():
        tid = await submit_docker_github_task(
            remote=rq,
            repo_url="https://github.com/user/repo",
            branch="main",
            dockerfile_path="Dockerfile",
            context_path=".",
            command="python app.py",
            build_args={"ARG": 2},
            environment={"X": True},
            timeout=99,
        )
        assert tid == "tid999"

    anyio.run(_do, backend="trio")

    assert captured["task_type"] == "docker.github"
    assert captured["model_name"] == "docker"
    assert captured["payload"]["repo_url"] == "https://github.com/user/repo"
    assert captured["payload"]["branch"] == "main"
    assert captured["payload"]["dockerfile_path"] == "Dockerfile"
    assert captured["payload"]["context_path"] == "."
    assert captured["payload"]["command"] == "python app.py"
    assert captured["payload"]["build_args"] == {"ARG": "2"}
    assert captured["payload"]["environment"] == {"X": "True"}
    assert captured["payload"]["timeout"] == 99
