"""
Unified MCP Tools

This module provides unified MCP tools that wrap the ipfs_accelerate_py/kit modules.
All tools are registered with the MCP server and exposed to the JavaScript SDK.

Architecture:
    kit modules (core functionality)
        ↓
    unified_tools (this file - MCP tool wrappers)
        ↓
    MCP server
        ↓
    JavaScript SDK → Dashboard
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.unified_tools")


def register_unified_tools(mcp: Any) -> None:
    """
    Register all unified tools with the MCP server.
    
    This function wraps kit modules as MCP tools with proper schemas.
    
    Args:
        mcp: MCP server instance
    """
    logger.info("Registering unified MCP tools")
    
    # Register GitHub tools
    try:
        register_github_tools(mcp)
        logger.debug("Registered GitHub unified tools")
    except Exception as e:
        logger.warning(f"Failed to register GitHub tools: {e}")
    
    # Register Docker tools
    try:
        register_docker_tools(mcp)
        logger.debug("Registered Docker unified tools")
    except Exception as e:
        logger.warning(f"Failed to register Docker tools: {e}")
    
    # Register Hardware tools
    try:
        register_hardware_tools(mcp)
        logger.debug("Registered Hardware unified tools")
    except Exception as e:
        logger.warning(f"Failed to register Hardware tools: {e}")
    
    # Register Runner tools
    try:
        register_runner_tools(mcp)
        logger.debug("Registered Runner unified tools")
    except Exception as e:
        logger.warning(f"Failed to register Runner tools: {e}")
    
    logger.info("All unified tools registered")


# GitHub Tools

def register_github_tools(mcp: Any) -> None:
    """Register GitHub kit tools with MCP server."""
    from ipfs_accelerate_py.kit.github_kit import get_github_kit
    
    def github_list_repos(owner: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
        """
        List GitHub repositories.
        
        Args:
            owner: Repository owner (uses authenticated user if None)
            limit: Maximum number of repos to return
            
        Returns:
            Dictionary with repository list
        """
        kit = get_github_kit()
        result = kit.list_repos(owner=owner, limit=limit)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def github_get_repo(repo: str) -> Dict[str, Any]:
        """
        Get GitHub repository details.
        
        Args:
            repo: Repository (owner/name)
            
        Returns:
            Dictionary with repository details
        """
        kit = get_github_kit()
        result = kit.get_repo(repo)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def github_list_prs(repo: str, state: str = "open", limit: int = 30) -> Dict[str, Any]:
        """
        List GitHub pull requests.
        
        Args:
            repo: Repository (owner/name)
            state: PR state (open, closed, merged, all)
            limit: Maximum number of PRs to return
            
        Returns:
            Dictionary with PR list
        """
        kit = get_github_kit()
        result = kit.list_prs(repo=repo, state=state, limit=limit)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def github_get_pr(repo: str, number: int) -> Dict[str, Any]:
        """
        Get GitHub pull request details.
        
        Args:
            repo: Repository (owner/name)
            number: PR number
            
        Returns:
            Dictionary with PR details
        """
        kit = get_github_kit()
        result = kit.get_pr(repo=repo, number=number)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def github_list_issues(repo: str, state: str = "open", limit: int = 30) -> Dict[str, Any]:
        """
        List GitHub issues.
        
        Args:
            repo: Repository (owner/name)
            state: Issue state (open, closed, all)
            limit: Maximum number of issues to return
            
        Returns:
            Dictionary with issue list
        """
        kit = get_github_kit()
        result = kit.list_issues(repo=repo, state=state, limit=limit)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def github_get_issue(repo: str, number: int) -> Dict[str, Any]:
        """
        Get GitHub issue details.
        
        Args:
            repo: Repository (owner/name)
            number: Issue number
            
        Returns:
            Dictionary with issue details
        """
        kit = get_github_kit()
        result = kit.get_issue(repo=repo, number=number)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    # Register tools with MCP server
    mcp.register_tool(
        name="github_list_repos",
        function=github_list_repos,
        description="List GitHub repositories",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "limit": {"type": "integer", "description": "Maximum number of repos", "default": 30}
            }
        }
    )
    
    mcp.register_tool(
        name="github_get_repo",
        function=github_get_repo,
        description="Get GitHub repository details",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository (owner/name)"}
            },
            "required": ["repo"]
        }
    )
    
    mcp.register_tool(
        name="github_list_prs",
        function=github_list_prs,
        description="List GitHub pull requests",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository (owner/name)"},
                "state": {"type": "string", "enum": ["open", "closed", "merged", "all"], "default": "open"},
                "limit": {"type": "integer", "description": "Maximum number of PRs", "default": 30}
            },
            "required": ["repo"]
        }
    )
    
    mcp.register_tool(
        name="github_get_pr",
        function=github_get_pr,
        description="Get GitHub pull request details",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository (owner/name)"},
                "number": {"type": "integer", "description": "PR number"}
            },
            "required": ["repo", "number"]
        }
    )
    
    mcp.register_tool(
        name="github_list_issues",
        function=github_list_issues,
        description="List GitHub issues",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository (owner/name)"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                "limit": {"type": "integer", "description": "Maximum number of issues", "default": 30}
            },
            "required": ["repo"]
        }
    )
    
    mcp.register_tool(
        name="github_get_issue",
        function=github_get_issue,
        description="Get GitHub issue details",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository (owner/name)"},
                "number": {"type": "integer", "description": "Issue number"}
            },
            "required": ["repo", "number"]
        }
    )


# Docker Tools

def register_docker_tools(mcp: Any) -> None:
    """Register Docker kit tools with MCP server."""
    from ipfs_accelerate_py.kit.docker_kit import get_docker_kit
    
    def docker_run_container(
        image: str,
        command: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        memory: Optional[str] = None,
        cpus: Optional[float] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run a Docker container.
        
        Args:
            image: Docker image name
            command: Command to run
            environment: Environment variables
            memory: Memory limit (e.g., "512m")
            cpus: CPU limit (e.g., 1.5)
            timeout: Execution timeout
            
        Returns:
            Dictionary with execution result
        """
        kit = get_docker_kit()
        result = kit.run_container(
            image=image,
            command=command.split() if command else None,
            environment=environment,
            memory=memory,
            cpus=cpus,
            timeout=timeout
        )
        return {
            "success": result.success,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "execution_time": result.execution_time
        }
    
    def docker_list_containers(all_containers: bool = False) -> Dict[str, Any]:
        """
        List Docker containers.
        
        Args:
            all_containers: Include stopped containers
            
        Returns:
            Dictionary with container list
        """
        kit = get_docker_kit()
        result = kit.list_containers(all_containers=all_containers)
        return {
            "success": result.success,
            "containers": result.data,
            "error": result.error
        }
    
    def docker_stop_container(container: str) -> Dict[str, Any]:
        """
        Stop a Docker container.
        
        Args:
            container: Container ID or name
            
        Returns:
            Dictionary with stop status
        """
        kit = get_docker_kit()
        result = kit.stop_container(container)
        return {
            "success": result.success,
            "error": result.error
        }
    
    def docker_pull_image(image: str) -> Dict[str, Any]:
        """
        Pull a Docker image.
        
        Args:
            image: Image name
            
        Returns:
            Dictionary with pull status
        """
        kit = get_docker_kit()
        result = kit.pull_image(image)
        return {
            "success": result.success,
            "error": result.error
        }
    
    # Register tools with MCP server
    mcp.register_tool(
        name="docker_run_container",
        function=docker_run_container,
        description="Run a Docker container",
        input_schema={
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Docker image name"},
                "command": {"type": "string", "description": "Command to run"},
                "environment": {"type": "object", "description": "Environment variables"},
                "memory": {"type": "string", "description": "Memory limit (e.g., '512m')"},
                "cpus": {"type": "number", "description": "CPU limit (e.g., 1.5)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 300}
            },
            "required": ["image"]
        }
    )
    
    mcp.register_tool(
        name="docker_list_containers",
        function=docker_list_containers,
        description="List Docker containers",
        input_schema={
            "type": "object",
            "properties": {
                "all_containers": {"type": "boolean", "description": "Include stopped containers", "default": False}
            }
        }
    )
    
    mcp.register_tool(
        name="docker_stop_container",
        function=docker_stop_container,
        description="Stop a Docker container",
        input_schema={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container ID or name"}
            },
            "required": ["container"]
        }
    )
    
    mcp.register_tool(
        name="docker_pull_image",
        function=docker_pull_image,
        description="Pull a Docker image",
        input_schema={
            "type": "object",
            "properties": {
                "image": {"type": "string", "description": "Image name"}
            },
            "required": ["image"]
        }
    )


# Hardware Tools

def register_hardware_tools(mcp: Any) -> None:
    """Register Hardware kit tools with MCP server."""
    from ipfs_accelerate_py.kit.hardware_kit import get_hardware_kit
    
    def hardware_get_info(include_detailed: bool = False) -> Dict[str, Any]:
        """
        Get hardware information.
        
        Args:
            include_detailed: Include detailed information
            
        Returns:
            Dictionary with hardware info
        """
        kit = get_hardware_kit()
        result = kit.get_hardware_info(include_detailed=include_detailed)
        return {
            "cpu": result.cpu,
            "gpu": result.gpu,
            "memory": result.memory,
            "accelerators": result.accelerators,
            "platform": result.platform_info
        }
    
    def hardware_test(accelerator: str = "all", test_level: str = "basic") -> Dict[str, Any]:
        """
        Test hardware accelerators.
        
        Args:
            accelerator: Accelerator to test (cuda, cpu, all)
            test_level: Level of testing (basic, comprehensive)
            
        Returns:
            Dictionary with test results
        """
        kit = get_hardware_kit()
        result = kit.test_hardware(accelerator=accelerator, test_level=test_level)
        return result
    
    def hardware_recommend(
        model_name: str,
        task: str = "inference",
        consider_available_only: bool = True
    ) -> Dict[str, Any]:
        """
        Get hardware recommendations for a model.
        
        Args:
            model_name: Model name
            task: Task type (inference, training, fine-tuning)
            consider_available_only: Only consider available hardware
            
        Returns:
            Dictionary with recommendations
        """
        kit = get_hardware_kit()
        result = kit.recommend_hardware(
            model_name=model_name,
            task=task,
            consider_available_only=consider_available_only
        )
        return result
    
    # Register tools with MCP server
    mcp.register_tool(
        name="hardware_get_info",
        function=hardware_get_info,
        description="Get hardware information",
        input_schema={
            "type": "object",
            "properties": {
                "include_detailed": {"type": "boolean", "description": "Include detailed information", "default": False}
            }
        }
    )
    
    mcp.register_tool(
        name="hardware_test",
        function=hardware_test,
        description="Test hardware accelerators",
        input_schema={
            "type": "object",
            "properties": {
                "accelerator": {"type": "string", "enum": ["cuda", "cpu", "all"], "default": "all"},
                "test_level": {"type": "string", "enum": ["basic", "comprehensive"], "default": "basic"}
            }
        }
    )
    
    mcp.register_tool(
        name="hardware_recommend",
        function=hardware_recommend,
        description="Get hardware recommendations for a model",
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Model name"},
                "task": {"type": "string", "enum": ["inference", "training", "fine-tuning"], "default": "inference"},
                "consider_available_only": {"type": "boolean", "description": "Only consider available hardware", "default": True}
            },
            "required": ["model_name"]
        }
    )


# Runner Tools (GitHub Actions Autoscaler)

def register_runner_tools(mcp: Any) -> None:
    """Register Runner autoscaler tools with MCP server."""
    from ipfs_accelerate_py.kit.runner_kit import get_runner_kit, RunnerConfig
    
    # Shared runner kit instance for stateful operations
    _runner_kit_instance = None
    
    def _get_runner_kit():
        """Get or create runner kit instance."""
        nonlocal _runner_kit_instance
        if _runner_kit_instance is None:
            _runner_kit_instance = get_runner_kit()
        return _runner_kit_instance
    
    def runner_start_autoscaler(
        owner: Optional[str] = None,
        poll_interval: int = 120,
        max_runners: int = 10,
        runner_image: str = "myoung34/github-runner:latest",
        background: bool = True
    ) -> Dict[str, Any]:
        """
        Start GitHub Actions runner autoscaler.
        
        Args:
            owner: GitHub owner (user or org) to monitor
            poll_interval: Poll interval in seconds
            max_runners: Maximum concurrent runners
            runner_image: Docker image for runners
            background: Run in background
            
        Returns:
            Dictionary with start status
        """
        config = RunnerConfig(
            owner=owner,
            poll_interval=poll_interval,
            max_runners=max_runners,
            runner_image=runner_image
        )
        kit = get_runner_kit(config)
        success = kit.start_autoscaler(background=background)
        
        return {
            "success": success,
            "running": kit.running,
            "message": "Autoscaler started" if success else "Failed to start autoscaler"
        }
    
    def runner_stop_autoscaler() -> Dict[str, Any]:
        """
        Stop GitHub Actions runner autoscaler.
        
        Returns:
            Dictionary with stop status
        """
        kit = _get_runner_kit()
        success = kit.stop_autoscaler()
        
        return {
            "success": success,
            "running": kit.running,
            "message": "Autoscaler stopped" if success else "Autoscaler not running"
        }
    
    def runner_get_status() -> Dict[str, Any]:
        """
        Get GitHub Actions runner autoscaler status.
        
        Returns:
            Dictionary with autoscaler status
        """
        kit = _get_runner_kit()
        status = kit.get_status()
        
        return {
            "success": True,
            "data": {
                "running": status.running,
                "start_time": status.start_time.isoformat() if status.start_time else None,
                "iterations": status.iterations,
                "active_runners": status.active_runners,
                "queued_workflows": status.queued_workflows,
                "repositories_monitored": status.repositories_monitored,
                "last_check": status.last_check.isoformat() if status.last_check else None
            }
        }
    
    def runner_list_workflows() -> Dict[str, Any]:
        """
        List GitHub workflow queues.
        
        Returns:
            Dictionary with workflow queues
        """
        kit = _get_runner_kit()
        queues = kit.get_workflow_queues()
        
        data = []
        for queue in queues:
            data.append({
                "repo": queue.repo,
                "total_workflows": queue.total,
                "running": queue.running,
                "failed": queue.failed,
                "pending": queue.pending
            })
        
        return {
            "success": True,
            "data": data,
            "count": len(data)
        }
    
    def runner_provision_for_workflow(repo: str) -> Dict[str, Any]:
        """
        Provision a runner for a specific repository.
        
        Args:
            repo: Repository in format 'owner/repo'
            
        Returns:
            Dictionary with provisioning result
        """
        kit = _get_runner_kit()
        
        # Generate token
        token = kit.generate_runner_token(repo)
        if not token:
            return {
                "success": False,
                "error": "Failed to generate runner token"
            }
        
        # Launch container
        container_id = kit.launch_runner_container(repo, token)
        
        return {
            "success": bool(container_id),
            "data": {
                "repo": repo,
                "container_id": container_id,
                "token_generated": bool(token)
            },
            "message": f"Provisioned runner for {repo}" if container_id else "Failed to launch container"
        }
    
    def runner_list_containers() -> Dict[str, Any]:
        """
        List active GitHub Actions runner containers.
        
        Returns:
            Dictionary with runner container list
        """
        kit = _get_runner_kit()
        runners = kit.list_runner_containers()
        
        data = []
        for runner in runners:
            data.append({
                "container_id": runner.container_id,
                "repo": runner.repo,
                "status": runner.status,
                "created_at": runner.created_at.isoformat()
            })
        
        return {
            "success": True,
            "data": data,
            "count": len(data)
        }
    
    def runner_stop_container(container: str) -> Dict[str, Any]:
        """
        Stop a GitHub Actions runner container.
        
        Args:
            container: Container ID or name
            
        Returns:
            Dictionary with stop status
        """
        kit = _get_runner_kit()
        success = kit.stop_runner_container(container)
        
        return {
            "success": success,
            "message": f"Stopped container {container}" if success else "Failed to stop container"
        }
    
    # Register tools with MCP
    mcp.register_tool(
        name="runner_start_autoscaler",
        function=runner_start_autoscaler,
        description="Start GitHub Actions runner autoscaler",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "GitHub owner (user or org) to monitor"},
                "poll_interval": {"type": "number", "default": 120, "description": "Poll interval in seconds"},
                "max_runners": {"type": "number", "default": 10, "description": "Maximum concurrent runners"},
                "runner_image": {"type": "string", "default": "myoung34/github-runner:latest"},
                "background": {"type": "boolean", "default": True, "description": "Run in background"}
            }
        }
    )
    
    mcp.register_tool(
        name="runner_stop_autoscaler",
        function=runner_stop_autoscaler,
        description="Stop GitHub Actions runner autoscaler",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
    
    mcp.register_tool(
        name="runner_get_status",
        function=runner_get_status,
        description="Get GitHub Actions runner autoscaler status",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
    
    mcp.register_tool(
        name="runner_list_workflows",
        function=runner_list_workflows,
        description="List GitHub workflow queues",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
    
    mcp.register_tool(
        name="runner_provision_for_workflow",
        function=runner_provision_for_workflow,
        description="Provision a runner for a specific repository",
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repository in format 'owner/repo'"}
            },
            "required": ["repo"]
        }
    )
    
    mcp.register_tool(
        name="runner_list_containers",
        function=runner_list_containers,
        description="List active GitHub Actions runner containers",
        input_schema={
            "type": "object",
            "properties": {}
        }
    )
    
    mcp.register_tool(
        name="runner_stop_container",
        function=runner_stop_container,
        description="Stop a GitHub Actions runner container",
        input_schema={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container ID or name"}
            },
            "required": ["container"]
        }
    )


__all__ = [
    'register_unified_tools',
    'register_github_tools',
    'register_docker_tools',
    'register_hardware_tools',
    'register_runner_tools',
]
