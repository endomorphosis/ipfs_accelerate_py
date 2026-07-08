"""Source-compatible MCP++ engine adapters for the tools.mcplusplus package."""

from __future__ import annotations

from typing import Any, Dict, List

from . import native_mcplusplus_tools as native_tools


class TaskQueueEngine:
    """Compatibility adapter that preserves the source TaskQueueEngine surface."""

    async def submit(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: float = 1.0,
        tags: List[str] | None = None,
        timeout: int | None = None,
        retry_policy: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_submit(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            tags=tags,
            timeout=timeout,
            retry_policy=retry_policy,
            metadata=metadata,
        )

    async def get_status(
        self,
        task_id: str,
        include_logs: bool = False,
        include_metrics: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_get_status(
            task_id=task_id,
            include_logs=include_logs,
            include_metrics=include_metrics,
        )

    async def cancel(
        self,
        task_id: str,
        reason: str | None = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_cancel(
            task_id=task_id,
            reason=reason,
            force=force,
        )

    async def list_tasks(
        self,
        status_filter: str | None = None,
        worker_filter: str | None = None,
        tag_filter: List[str] | None = None,
        priority_min: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_list(
            status_filter=status_filter,
            worker_filter=worker_filter,
            tag_filter=tag_filter,
            priority_min=priority_min,
            limit=limit,
            offset=offset,
        )

    async def set_priority(
        self,
        task_id: str,
        new_priority: float,
        requeue: bool = True,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_set_priority(
            task_id=task_id,
            new_priority=new_priority,
            requeue=requeue,
        )

    async def get_result(
        self,
        task_id: str,
        include_output: bool = True,
        include_logs: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_result(
            task_id=task_id,
            include_output=include_output,
            include_logs=include_logs,
        )

    async def get_stats(
        self,
        include_worker_stats: bool = False,
        include_historical: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_stats(
            include_worker_stats=include_worker_stats,
            include_historical=include_historical,
        )

    async def retry(
        self,
        task_id: str,
        retry_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_retry(
            task_id=task_id,
            retry_config=retry_config,
        )

    async def pause(
        self,
        reason: str | None = None,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_pause(reason=reason)

    async def resume(
        self,
        reorder_by_priority: bool = True,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_resume(
            reorder_by_priority=reorder_by_priority,
        )

    async def clear(
        self,
        status_filter: str | None = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_taskqueue_clear(
            status_filter=status_filter,
            confirm=confirm,
        )

    async def register_worker(
        self,
        worker_id: str,
        capabilities: List[str],
        max_concurrent_tasks: int = 5,
        resource_limits: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_worker_register(
            worker_id=worker_id,
            capabilities=capabilities,
            max_concurrent_tasks=max_concurrent_tasks,
            resource_limits=resource_limits,
            metadata=metadata,
        )

    async def unregister_worker(
        self,
        worker_id: str,
        graceful: bool = True,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_worker_unregister(
            worker_id=worker_id,
            graceful=graceful,
            timeout=timeout,
        )

    async def get_worker_status(
        self,
        worker_id: str,
        include_tasks: bool = False,
        include_metrics: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_worker_status(
            worker_id=worker_id,
            include_tasks=include_tasks,
            include_metrics=include_metrics,
        )


class WorkflowEngine:
    """Compatibility adapter that preserves the source WorkflowEngine surface."""

    async def submit(
        self,
        workflow_id: str,
        name: str,
        steps: List[Dict[str, Any]],
        priority: float = 1.0,
        tags: List[str] | None = None,
        dependencies: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_workflow_submit(
            workflow_id=workflow_id,
            name=name,
            steps=steps,
            priority=priority,
            tags=tags,
            dependencies=dependencies,
            metadata=metadata,
        )

    async def get_status(
        self,
        workflow_id: str,
        include_steps: bool = True,
        include_metrics: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_workflow_get_status(
            workflow_id=workflow_id,
            include_steps=include_steps,
            include_metrics=include_metrics,
        )

    async def cancel(
        self,
        workflow_id: str,
        reason: str | None = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_workflow_cancel(
            workflow_id=workflow_id,
            reason=reason,
            force=force,
        )

    async def list_workflows(
        self,
        status_filter: str | None = None,
        peer_filter: str | None = None,
        tag_filter: List[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_workflow_list(
            status_filter=status_filter,
            peer_filter=peer_filter,
            tag_filter=tag_filter,
            limit=limit,
            offset=offset,
        )

    async def get_dependencies(
        self,
        workflow_id: str,
        fmt: str = "json",
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_workflow_dependencies(
            workflow_id=workflow_id,
            fmt=fmt,
        )

    async def get_result(
        self,
        workflow_id: str,
        include_outputs: bool = True,
        include_logs: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_workflow_result(
            workflow_id=workflow_id,
            include_outputs=include_outputs,
            include_logs=include_logs,
        )


class PeerEngine:
    """Compatibility adapter that preserves the source PeerEngine surface."""

    async def discover(
        self,
        capability_filter: List[str] | None = None,
        max_peers: int = 10,
        timeout: int = 30,
        include_metrics: bool = False,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_peer_discover(
            capability_filter=capability_filter,
            max_peers=max_peers,
            timeout=timeout,
            include_metrics=include_metrics,
        )

    async def connect(
        self,
        peer_id: str,
        multiaddr: str,
        timeout: int = 30,
        retry_count: int = 3,
        persist: bool = True,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_peer_connect(
            peer_id=peer_id,
            multiaddr=multiaddr,
            timeout=timeout,
            retry_count=retry_count,
            persist=persist,
        )

    async def disconnect(
        self,
        peer_id: str,
        reason: str | None = None,
        graceful: bool = True,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_peer_disconnect(
            peer_id=peer_id,
            reason=reason,
            graceful=graceful,
        )

    async def list_peers(
        self,
        status_filter: str | None = None,
        capability_filter: List[str] | None = None,
        sort_by: str = "last_seen",
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_peer_list(
            status_filter=status_filter or "",
            capability_filter=capability_filter,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )

    async def get_metrics(
        self,
        peer_id: str,
        include_history: bool = False,
        history_hours: int = 24,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_peer_metrics(
            peer_id=peer_id,
            include_history=include_history,
            history_hours=history_hours,
        )

    async def bootstrap(
        self,
        bootstrap_nodes: List[str] | None = None,
        timeout: int = 60,
        min_connections: int = 3,
        max_connections: int = 10,
    ) -> Dict[str, Any]:
        return await native_tools.mcplusplus_peer_bootstrap_network(
            bootstrap_nodes=bootstrap_nodes,
            timeout=timeout,
            min_connections=min_connections,
            max_connections=max_connections,
        )


__all__ = ["TaskQueueEngine", "PeerEngine", "WorkflowEngine"]