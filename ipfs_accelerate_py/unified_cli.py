#!/usr/bin/env python3
"""
IPFS Accelerate Unified CLI

This is the unified CLI that wraps all ipfs_accelerate_py/kit modules and
exposes them through a single command-line interface.

Architecture:
    kit modules (core functionality)
        ↓
    unified_cli (this file - CLI interface)
        ↓
    User commands

Usage:
    ipfs-accelerate github list-repos
    ipfs-accelerate github create-pr owner/repo "Fix bug" "This fixes the bug"
    ipfs-accelerate docker run python:3.9 "python --version"
    ipfs-accelerate hardware info
    ipfs-accelerate --help
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_cli")


def import_kit_module(module_name: str):
    """
    Import a kit module dynamically.
    
    Args:
        module_name: Name of the module (e.g., 'github', 'docker', 'runner')
        
    Returns:
        Module or None if not available
    """
    try:
        # Try absolute import first
        if module_name == 'github':
            from ipfs_accelerate_py.kit import github_kit
            return github_kit
        elif module_name == 'docker':
            from ipfs_accelerate_py.kit import docker_kit
            return docker_kit
        elif module_name == 'hardware':
            from ipfs_accelerate_py.kit import hardware_kit
            return hardware_kit
        elif module_name == 'runner':
            from ipfs_accelerate_py.kit import runner_kit
            return runner_kit
        elif module_name == 'ipfs_files':
            from ipfs_accelerate_py.kit import ipfs_files_kit
            return ipfs_files_kit
        elif module_name == 'network':
            from ipfs_accelerate_py.kit import network_kit
            return network_kit
        else:
            logger.error(f"Unknown module: {module_name}")
            return None
    except ImportError:
        # Try relative import
        try:
            import os
            import sys
            # Add parent directory to path
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if module_name == 'github':
                from kit import github_kit
                return github_kit
            elif module_name == 'docker':
                from kit import docker_kit
                return docker_kit
            elif module_name == 'hardware':
                from kit import hardware_kit
                return hardware_kit
            elif module_name == 'runner':
                from kit import runner_kit
                return runner_kit
            elif module_name == 'ipfs_files':
                from kit import ipfs_files_kit
                return ipfs_files_kit
            elif module_name == 'network':
                from kit import network_kit
                return network_kit
        except ImportError as e:
            logger.error(f"Failed to import {module_name} module: {e}")
            return None


def print_result(result: Any, format: str = "json"):
    """
    Print a result in the specified format.
    
    Args:
        result: Result to print
        format: Output format (json, text)
    """
    if format == "json":
        # Handle dataclass results
        if hasattr(result, '__dict__'):
            print(json.dumps(result.__dict__, indent=2, default=str))
        else:
            print(json.dumps(result, indent=2, default=str))
    else:
        print(result)


# GitHub Commands

def github_command(args):
    """Handle GitHub commands."""
    github_kit = import_kit_module('github')
    if not github_kit:
        print("GitHub kit module not available", file=sys.stderr)
        sys.exit(1)
    
    kit = github_kit.get_github_kit()
    
    if args.github_command == 'list-repos':
        result = kit.list_repos(owner=args.owner, limit=args.limit)
        print_result(result, args.format)
    
    elif args.github_command == 'get-repo':
        if not args.repo:
            print("Error: --repo required", file=sys.stderr)
            sys.exit(1)
        result = kit.get_repo(args.repo)
        print_result(result, args.format)
    
    elif args.github_command == 'clone-repo':
        if not args.repo:
            print("Error: --repo required", file=sys.stderr)
            sys.exit(1)
        result = kit.clone_repo(args.repo, args.path)
        print_result(result, args.format)
    
    elif args.github_command == 'list-prs':
        if not args.repo:
            print("Error: --repo required", file=sys.stderr)
            sys.exit(1)
        result = kit.list_prs(args.repo, state=args.state, limit=args.limit)
        print_result(result, args.format)
    
    elif args.github_command == 'get-pr':
        if not args.repo or not args.number:
            print("Error: --repo and --number required", file=sys.stderr)
            sys.exit(1)
        result = kit.get_pr(args.repo, args.number)
        print_result(result, args.format)
    
    elif args.github_command == 'list-issues':
        if not args.repo:
            print("Error: --repo required", file=sys.stderr)
            sys.exit(1)
        result = kit.list_issues(args.repo, state=args.state, limit=args.limit)
        print_result(result, args.format)
    
    elif args.github_command == 'get-issue':
        if not args.repo or not args.number:
            print("Error: --repo and --number required", file=sys.stderr)
            sys.exit(1)
        result = kit.get_issue(args.repo, args.number)
        print_result(result, args.format)
    
    else:
        print(f"Unknown GitHub command: {args.github_command}", file=sys.stderr)
        sys.exit(1)


# Docker Commands

def docker_command(args):
    """Handle Docker commands."""
    docker_kit = import_kit_module('docker')
    if not docker_kit:
        print("Docker kit module not available", file=sys.stderr)
        sys.exit(1)
    
    kit = docker_kit.get_docker_kit()
    
    if args.docker_command == 'run':
        if not args.image:
            print("Error: --image required", file=sys.stderr)
            sys.exit(1)
        
        # Parse environment variables
        env = {}
        if args.env:
            for e in args.env:
                if '=' in e:
                    key, value = e.split('=', 1)
                    env[key] = value
        
        result = kit.run_container(
            image=args.image,
            command=args.command.split() if args.command else None,
            environment=env if env else None,
            memory=args.memory,
            cpus=args.cpus,
            timeout=args.timeout
        )
        print_result(result, args.format)
    
    elif args.docker_command == 'list':
        result = kit.list_containers(all_containers=args.all)
        print_result(result, args.format)
    
    elif args.docker_command == 'stop':
        if not args.container:
            print("Error: --container required", file=sys.stderr)
            sys.exit(1)
        result = kit.stop_container(args.container)
        print_result(result, args.format)
    
    elif args.docker_command == 'pull':
        if not args.image:
            print("Error: --image required", file=sys.stderr)
            sys.exit(1)
        result = kit.pull_image(args.image)
        print_result(result, args.format)
    
    elif args.docker_command == 'images':
        result = kit.list_images()
        print_result(result, args.format)
    
    else:
        print(f"Unknown Docker command: {args.docker_command}", file=sys.stderr)
        sys.exit(1)


# Hardware Commands

def hardware_command(args):
    """Handle Hardware commands."""
    hardware_kit = import_kit_module('hardware')
    if not hardware_kit:
        print("Hardware kit module not available", file=sys.stderr)
        sys.exit(1)
    
    kit = hardware_kit.get_hardware_kit()
    
    if args.hardware_command == 'info':
        result = kit.get_hardware_info(include_detailed=args.detailed)
        print_result(result, args.format)
    
    elif args.hardware_command == 'test':
        result = kit.test_hardware(
            accelerator=args.accelerator,
            test_level=args.level
        )
        print_result(result, args.format)
    
    elif args.hardware_command == 'recommend':
        if not args.model:
            print("Error: --model required", file=sys.stderr)
            sys.exit(1)
        result = kit.recommend_hardware(
            model_name=args.model,
            task=args.task,
            consider_available_only=args.available_only
        )
        print_result(result, args.format)
    
    else:
        print(f"Unknown Hardware command: {args.hardware_command}", file=sys.stderr)
        sys.exit(1)


def runner_command(args):
    """Handle Runner autoscaler commands."""
    runner_kit = import_kit_module('runner')
    if not runner_kit:
        print("Runner kit module not available", file=sys.stderr)
        sys.exit(1)
    
    # Create runner config from args
    config = runner_kit.RunnerConfig(
        owner=getattr(args, 'owner', None),
        poll_interval=getattr(args, 'interval', 120),
        max_runners=getattr(args, 'max_runners', 10),
        runner_image=getattr(args, 'image', 'myoung34/github-runner:latest')
    )
    
    kit = runner_kit.get_runner_kit(config)
    
    if args.runner_command == 'start':
        # Start autoscaler
        result = kit.start_autoscaler(background=args.background)
        if result:
            print("✓ Autoscaler started")
            if args.background:
                print("  Running in background")
            else:
                print("  Press Ctrl+C to stop")
                # Keep running until interrupted
                try:
                    import time
                    while kit.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    kit.stop_autoscaler()
        else:
            print("✗ Failed to start autoscaler", file=sys.stderr)
            sys.exit(1)
    
    elif args.runner_command == 'stop':
        # Stop autoscaler
        result = kit.stop_autoscaler()
        if result:
            print("✓ Autoscaler stopped")
        else:
            print("✗ Autoscaler not running", file=sys.stderr)
    
    elif args.runner_command == 'status':
        # Get status
        status = kit.get_status()
        output = {
            'running': status.running,
            'start_time': status.start_time.isoformat() if status.start_time else None,
            'iterations': status.iterations,
            'active_runners': status.active_runners,
            'queued_workflows': status.queued_workflows,
            'repositories_monitored': status.repositories_monitored,
            'last_check': status.last_check.isoformat() if status.last_check else None
        }
        print_result(output, args.format)
    
    elif args.runner_command == 'list-workflows':
        # List workflow queues
        queues = kit.get_workflow_queues()
        output = []
        for queue in queues:
            output.append({
                'repo': queue.repo,
                'total_workflows': queue.total,
                'running': queue.running,
                'failed': queue.failed,
                'pending': queue.pending
            })
        print_result(output, args.format)
    
    elif args.runner_command == 'list-containers':
        # List runner containers
        runners = kit.list_runner_containers()
        output = []
        for runner in runners:
            output.append({
                'container_id': runner.container_id,
                'repo': runner.repo,
                'status': runner.status,
                'created_at': runner.created_at.isoformat()
            })
        print_result(output, args.format)
    
    elif args.runner_command == 'provision':
        # Manually provision runners
        if args.repo:
            # Provision for specific repo
            token = kit.generate_runner_token(args.repo)
            if token:
                container_id = kit.launch_runner_container(args.repo, token)
                if container_id:
                    print(f"✓ Provisioned runner for {args.repo}")
                    print(f"  Container ID: {container_id}")
                else:
                    print(f"✗ Failed to launch container", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"✗ Failed to generate token", file=sys.stderr)
                sys.exit(1)
        else:
            # Provision for all queues
            queues = kit.get_workflow_queues()
            results = kit.provision_runners_for_queues(queues)
            print_result(results, args.format)
    
    elif args.runner_command == 'stop-container':
        # Stop a specific container
        if not args.container:
            print("Error: --container required", file=sys.stderr)
            sys.exit(1)
        result = kit.stop_runner_container(args.container)
        if result:
            print(f"✓ Stopped container {args.container}")
        else:
            print(f"✗ Failed to stop container", file=sys.stderr)
            sys.exit(1)
    
    else:
        print(f"Unknown runner command: {args.runner_command}", file=sys.stderr)
        sys.exit(1)


# IPFS Files Commands

def ipfs_files_command(args):
    """Handle IPFS files commands."""
    ipfs_files_kit = import_kit_module('ipfs_files')
    if not ipfs_files_kit:
        print("IPFS files kit module not available", file=sys.stderr)
        sys.exit(1)
    
    kit = ipfs_files_kit.get_ipfs_files_kit()
    
    if args.ipfs_files_command == 'add':
        if not args.path:
            print("Error: --path required", file=sys.stderr)
            sys.exit(1)
        result = kit.add_file(path=args.path, pin=args.pin)
        print_result(result, args.format)
    
    elif args.ipfs_files_command == 'get':
        if not args.cid or not args.output:
            print("Error: --cid and --output required", file=sys.stderr)
            sys.exit(1)
        result = kit.get_file(cid=args.cid, output_path=args.output)
        print_result(result, args.format)
    
    elif args.ipfs_files_command == 'cat':
        if not args.cid:
            print("Error: --cid required", file=sys.stderr)
            sys.exit(1)
        result = kit.cat_file(cid=args.cid)
        print_result(result, args.format)
    
    elif args.ipfs_files_command == 'pin':
        if not args.cid:
            print("Error: --cid required", file=sys.stderr)
            sys.exit(1)
        result = kit.pin_file(cid=args.cid)
        print_result(result, args.format)
    
    elif args.ipfs_files_command == 'unpin':
        if not args.cid:
            print("Error: --cid required", file=sys.stderr)
            sys.exit(1)
        result = kit.unpin_file(cid=args.cid)
        print_result(result, args.format)
    
    elif args.ipfs_files_command == 'list':
        result = kit.list_files(path=args.path if hasattr(args, 'path') else '/')
        print_result(result, args.format)
    
    elif args.ipfs_files_command == 'validate-cid':
        if not args.cid:
            print("Error: --cid required", file=sys.stderr)
            sys.exit(1)
        result = kit.validate_cid(cid=args.cid)
        print_result(result, args.format)
    
    else:
        print(f"Unknown ipfs-files command: {args.ipfs_files_command}", file=sys.stderr)
        sys.exit(1)


# Network Commands

def network_command(args):
    """Handle network commands."""
    network_kit = import_kit_module('network')
    if not network_kit:
        print("Network kit module not available", file=sys.stderr)
        sys.exit(1)
    
    kit = network_kit.get_network_kit()
    
    if args.network_command == 'list-peers':
        result = kit.list_peers()
        print_result(result, args.format)
    
    elif args.network_command == 'connect':
        if not args.peer:
            print("Error: --peer required", file=sys.stderr)
            sys.exit(1)
        result = kit.connect_peer(address=args.peer)
        print_result(result, args.format)
    
    elif args.network_command == 'disconnect':
        if not args.peer:
            print("Error: --peer required", file=sys.stderr)
            sys.exit(1)
        result = kit.disconnect_peer(peer_id=args.peer)
        print_result(result, args.format)
    
    elif args.network_command == 'dht-put':
        if not args.key or not args.value:
            print("Error: --key and --value required", file=sys.stderr)
            sys.exit(1)
        result = kit.dht_put(key=args.key, value=args.value)
        print_result(result, args.format)
    
    elif args.network_command == 'dht-get':
        if not args.key:
            print("Error: --key required", file=sys.stderr)
            sys.exit(1)
        result = kit.dht_get(key=args.key)
        print_result(result, args.format)
    
    elif args.network_command == 'swarm-info':
        result = kit.get_swarm_info()
        print_result(result, args.format)
    
    elif args.network_command == 'bandwidth':
        result = kit.get_bandwidth_stats()
        print_result(result, args.format)
    
    elif args.network_command == 'ping':
        if not args.peer:
            print("Error: --peer required", file=sys.stderr)
            sys.exit(1)
        result = kit.ping_peer(peer_id=args.peer, count=args.count if hasattr(args, 'count') else 5)
        print_result(result, args.format)
    
    else:
        print(f"Unknown network command: {args.network_command}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IPFS Accelerate Unified CLI - Unified interface for all kit modules",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='module', help='Module to use')
    
    # GitHub subcommands
    github_parser = subparsers.add_parser('github', help='GitHub operations')
    github_subparsers = github_parser.add_subparsers(dest='github_command', help='GitHub command')
    
    # GitHub list-repos
    list_repos = github_subparsers.add_parser('list-repos', help='List repositories')
    list_repos.add_argument('--owner', help='Repository owner')
    list_repos.add_argument('--limit', type=int, default=30, help='Maximum number of repos')
    
    # GitHub get-repo
    get_repo = github_subparsers.add_parser('get-repo', help='Get repository details')
    get_repo.add_argument('--repo', required=True, help='Repository (owner/name)')
    
    # GitHub clone-repo
    clone_repo = github_subparsers.add_parser('clone-repo', help='Clone repository')
    clone_repo.add_argument('--repo', required=True, help='Repository (owner/name or URL)')
    clone_repo.add_argument('--path', help='Target path')
    
    # GitHub list-prs
    list_prs = github_subparsers.add_parser('list-prs', help='List pull requests')
    list_prs.add_argument('--repo', required=True, help='Repository (owner/name)')
    list_prs.add_argument('--state', default='open', choices=['open', 'closed', 'merged', 'all'])
    list_prs.add_argument('--limit', type=int, default=30)
    
    # GitHub get-pr
    get_pr = github_subparsers.add_parser('get-pr', help='Get pull request')
    get_pr.add_argument('--repo', required=True, help='Repository (owner/name)')
    get_pr.add_argument('--number', type=int, required=True, help='PR number')
    
    # GitHub list-issues
    list_issues = github_subparsers.add_parser('list-issues', help='List issues')
    list_issues.add_argument('--repo', required=True, help='Repository (owner/name)')
    list_issues.add_argument('--state', default='open', choices=['open', 'closed', 'all'])
    list_issues.add_argument('--limit', type=int, default=30)
    
    # GitHub get-issue
    get_issue = github_subparsers.add_parser('get-issue', help='Get issue')
    get_issue.add_argument('--repo', required=True, help='Repository (owner/name)')
    get_issue.add_argument('--number', type=int, required=True, help='Issue number')
    
    # Docker subcommands
    docker_parser = subparsers.add_parser('docker', help='Docker operations')
    docker_subparsers = docker_parser.add_subparsers(dest='docker_command', help='Docker command')
    
    # Docker run
    run_container = docker_subparsers.add_parser('run', help='Run container')
    run_container.add_argument('--image', required=True, help='Docker image')
    run_container.add_argument('--command', help='Command to run')
    run_container.add_argument('--env', action='append', help='Environment variable (KEY=VALUE)')
    run_container.add_argument('--memory', help='Memory limit (e.g., 512m)')
    run_container.add_argument('--cpus', type=float, help='CPU limit (e.g., 1.5)')
    run_container.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    
    # Docker list
    list_containers = docker_subparsers.add_parser('list', help='List containers')
    list_containers.add_argument('--all', '-a', action='store_true', help='Include stopped containers')
    
    # Docker stop
    stop_container = docker_subparsers.add_parser('stop', help='Stop container')
    stop_container.add_argument('--container', required=True, help='Container ID or name')
    
    # Docker pull
    pull_image = docker_subparsers.add_parser('pull', help='Pull image')
    pull_image.add_argument('--image', required=True, help='Image name')
    
    # Docker images
    list_images = docker_subparsers.add_parser('images', help='List images')
    
    # Hardware subcommands
    hardware_parser = subparsers.add_parser('hardware', help='Hardware operations')
    hardware_subparsers = hardware_parser.add_subparsers(dest='hardware_command', help='Hardware command')
    
    # Hardware info
    hw_info = hardware_subparsers.add_parser('info', help='Get hardware information')
    hw_info.add_argument('--detailed', action='store_true', help='Include detailed information')
    
    # Hardware test
    hw_test = hardware_subparsers.add_parser('test', help='Test hardware')
    hw_test.add_argument('--accelerator', default='all', choices=['cuda', 'cpu', 'all'])
    hw_test.add_argument('--level', default='basic', choices=['basic', 'comprehensive'])
    
    # Hardware recommend
    hw_recommend = hardware_subparsers.add_parser('recommend', help='Get hardware recommendations')
    hw_recommend.add_argument('--model', required=True, help='Model name')
    hw_recommend.add_argument('--task', default='inference', choices=['inference', 'training', 'fine-tuning'])
    hw_recommend.add_argument('--available-only', action='store_true', default=True)
    
    # Runner subcommands (GitHub Actions autoscaler)
    runner_parser = subparsers.add_parser('runner', help='GitHub Actions runner autoscaling')
    runner_subparsers = runner_parser.add_subparsers(dest='runner_command', help='Runner command')
    
    # Runner start
    runner_start = runner_subparsers.add_parser('start', help='Start runner autoscaler')
    runner_start.add_argument('--owner', help='GitHub owner (user or org) to monitor')
    runner_start.add_argument('--interval', type=int, default=120, help='Poll interval in seconds')
    runner_start.add_argument('--max-runners', type=int, default=10, help='Maximum concurrent runners')
    runner_start.add_argument('--image', default='myoung34/github-runner:latest', help='Runner Docker image')
    runner_start.add_argument('--background', action='store_true', help='Run in background')
    
    # Runner stop
    runner_stop = runner_subparsers.add_parser('stop', help='Stop runner autoscaler')
    
    # Runner status
    runner_status = runner_subparsers.add_parser('status', help='Get autoscaler status')
    
    # Runner list-workflows
    runner_list_workflows = runner_subparsers.add_parser('list-workflows', help='List workflow queues')
    
    # Runner list-containers
    runner_list_containers = runner_subparsers.add_parser('list-containers', help='List active runner containers')
    
    # Runner provision
    runner_provision = runner_subparsers.add_parser('provision', help='Manually provision runners')
    runner_provision.add_argument('--repo', help='Repository (owner/name) - if omitted, provisions for all queues')
    
    # Runner stop-container
    runner_stop_container = runner_subparsers.add_parser('stop-container', help='Stop a runner container')
    runner_stop_container.add_argument('--container', required=True, help='Container ID or name')
    
    # IPFS Files subcommands
    ipfs_files_parser = subparsers.add_parser('ipfs-files', help='IPFS file operations')
    ipfs_files_subparsers = ipfs_files_parser.add_subparsers(dest='ipfs_files_command', help='IPFS files command')
    
    # IPFS Files add
    ipfs_files_add = ipfs_files_subparsers.add_parser('add', help='Add file to IPFS')
    ipfs_files_add.add_argument('--path', required=True, help='Path to file')
    ipfs_files_add.add_argument('--pin', action='store_true', help='Pin the file after adding')
    
    # IPFS Files get
    ipfs_files_get = ipfs_files_subparsers.add_parser('get', help='Get file from IPFS by CID')
    ipfs_files_get.add_argument('--cid', required=True, help='Content Identifier (CID)')
    ipfs_files_get.add_argument('--output', required=True, help='Output path')
    
    # IPFS Files cat
    ipfs_files_cat = ipfs_files_subparsers.add_parser('cat', help='Read file content from IPFS')
    ipfs_files_cat.add_argument('--cid', required=True, help='Content Identifier (CID)')
    
    # IPFS Files pin
    ipfs_files_pin = ipfs_files_subparsers.add_parser('pin', help='Pin content in IPFS')
    ipfs_files_pin.add_argument('--cid', required=True, help='Content Identifier (CID)')
    
    # IPFS Files unpin
    ipfs_files_unpin = ipfs_files_subparsers.add_parser('unpin', help='Unpin content from IPFS')
    ipfs_files_unpin.add_argument('--cid', required=True, help='Content Identifier (CID)')
    
    # IPFS Files list
    ipfs_files_list = ipfs_files_subparsers.add_parser('list', help='List IPFS files')
    ipfs_files_list.add_argument('--path', default='/', help='IPFS path to list')
    
    # IPFS Files validate-cid
    ipfs_files_validate = ipfs_files_subparsers.add_parser('validate-cid', help='Validate CID format')
    ipfs_files_validate.add_argument('--cid', required=True, help='Content Identifier (CID)')
    
    # Network subcommands
    network_parser = subparsers.add_parser('network', help='Network and peer operations')
    network_subparsers = network_parser.add_subparsers(dest='network_command', help='Network command')
    
    # Network list-peers
    network_list_peers = network_subparsers.add_parser('list-peers', help='List connected peers')
    
    # Network connect
    network_connect = network_subparsers.add_parser('connect', help='Connect to peer')
    network_connect.add_argument('--peer', required=True, help='Peer multiaddr or ID')
    
    # Network disconnect
    network_disconnect = network_subparsers.add_parser('disconnect', help='Disconnect from peer')
    network_disconnect.add_argument('--peer', required=True, help='Peer ID')
    
    # Network dht-put
    network_dht_put = network_subparsers.add_parser('dht-put', help='Store key-value in DHT')
    network_dht_put.add_argument('--key', required=True, help='Key')
    network_dht_put.add_argument('--value', required=True, help='Value')
    
    # Network dht-get
    network_dht_get = network_subparsers.add_parser('dht-get', help='Retrieve value from DHT')
    network_dht_get.add_argument('--key', required=True, help='Key')
    
    # Network swarm-info
    network_swarm_info = network_subparsers.add_parser('swarm-info', help='Get swarm statistics')
    
    # Network bandwidth
    network_bandwidth = network_subparsers.add_parser('bandwidth', help='Get bandwidth statistics')
    
    # Network ping
    network_ping = network_subparsers.add_parser('ping', help='Ping a peer')
    network_ping.add_argument('--peer', required=True, help='Peer ID')
    network_ping.add_argument('--count', type=int, default=5, help='Number of pings')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Route to appropriate handler
    if not args.module:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.module == 'github':
            github_command(args)
        elif args.module == 'docker':
            docker_command(args)
        elif args.module == 'hardware':
            hardware_command(args)
        elif args.module == 'runner':
            runner_command(args)
        elif args.module == 'ipfs-files':
            ipfs_files_command(args)
        elif args.module == 'network':
            network_command(args)
        else:
            print(f"Unknown module: {args.module}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
