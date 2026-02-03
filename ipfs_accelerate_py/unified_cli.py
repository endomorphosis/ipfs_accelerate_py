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
    ipfs-kit github list-repos
    ipfs-kit github create-pr owner/repo "Fix bug" "This fixes the bug"
    ipfs-kit docker run python:3.9 "python --version"
    ipfs-kit hardware info
    ipfs-kit --help
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
logger = logging.getLogger("ipfs_kit_cli")


def import_kit_module(module_name: str):
    """
    Import a kit module dynamically.
    
    Args:
        module_name: Name of the module (e.g., 'github', 'docker')
        
    Returns:
        Module or None if not available
    """
    try:
        if module_name == 'github':
            from ipfs_accelerate_py.kit import github_kit
            return github_kit
        elif module_name == 'docker':
            from ipfs_accelerate_py.kit import docker_kit
            return docker_kit
        elif module_name == 'hardware':
            from ipfs_accelerate_py.kit import hardware_kit
            return hardware_kit
        else:
            logger.error(f"Unknown module: {module_name}")
            return None
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
