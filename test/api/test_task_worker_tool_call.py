import time


def test_task_worker_tool_call_completes(tmp_path):
	from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
	from ipfs_accelerate_py.p2p_tasks.worker import run_worker

	class StubAccelerate:
		def call_tool(self, name, args):
			assert name == "echo"
			return {"echo": args}

	queue_path = str(tmp_path / "task_queue.duckdb")
	queue = TaskQueue(queue_path)
	task_id = queue.submit(task_type="tool.call", model_name="", payload={"tool": "echo", "args": {"x": 1}})

	# Process exactly one task.
	rc = run_worker(
		queue_path=queue_path,
		worker_id="w1",
		poll_interval_s=0.05,
		once=True,
		p2p_service=False,
		accelerate_instance=StubAccelerate(),
		supported_task_types=["tool.call"],
	)
	assert rc == 0

	# Task is completed with result.
	out = queue.get(task_id)
	assert out is not None
	assert out.get("status") == "completed"
	assert isinstance(out.get("result"), dict)
	assert out["result"].get("tool") == "echo"
	assert out["result"].get("result") == {"echo": {"x": 1}}

	# Ensure timestamps exist.
	assert float(out.get("updated_at") or 0) > 0
