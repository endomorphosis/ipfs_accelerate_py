#!/usr/bin/env python3
"""
Example: Automated GitHub Workflow Queue Management

This example demonstrates how to use the GitHub CLI integration to:
1. Detect repositories with recent activity
2. Create workflow queues for running/failed workflows
3. Automatically provision self-hosted runners based on system capacity
"""

import sys
import os
import json
from datetime import datetime

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager


def main():
    """Main example function"""
    print("=" * 80)
    print("GitHub Workflow Queue Management Example")
    print("=" * 80)
    print()
    
    # Step 1: Initialize GitHub CLI
    print("Step 1: Initializing GitHub CLI...")
    try:
        gh = GitHubCLI()
        print("✓ GitHub CLI initialized")
    except Exception as e:
        print(f"✗ Failed to initialize GitHub CLI: {e}")
        print("\nPlease ensure:")
        print("  1. GitHub CLI is installed (gh)")
        print("  2. You are authenticated (gh auth login)")
        return 1
    
    # Check authentication
    print("\nStep 2: Checking authentication...")
    auth_status = gh.get_auth_status()
    if auth_status["authenticated"]:
        print("✓ Authenticated with GitHub")
    else:
        print("✗ Not authenticated with GitHub")
        print("  Run: gh auth login")
        return 1
    
    # Step 3: List recent repositories (optional)
    print("\nStep 3: Listing repositories...")
    repos = gh.list_repos(limit=5)
    print(f"✓ Found {len(repos)} repositories (showing first 5):")
    for repo in repos[:5]:
        owner = repo["owner"]["login"]
        name = repo["name"]
        updated = repo["updatedAt"]
        print(f"  - {owner}/{name} (updated: {updated})")
    
    # Step 4: Create workflow queues
    print("\nStep 4: Creating workflow queues for repos updated in last day...")
    queue_mgr = WorkflowQueue(gh)
    
    # Get repos with recent activity
    recent_repos = queue_mgr.get_repos_with_recent_activity(since_days=1)
    print(f"✓ Found {len(recent_repos)} repositories with recent activity")
    
    if not recent_repos:
        print("\nNo repositories with recent activity found.")
        print("This example requires repositories that were updated in the last 24 hours.")
        return 0
    
    # Create queues for these repos
    queues = queue_mgr.create_workflow_queues(since_days=1)
    print(f"✓ Created workflow queues for {len(queues)} repositories")
    
    # Display queue statistics
    print("\nWorkflow Queue Statistics:")
    print("-" * 80)
    total_workflows = 0
    total_running = 0
    total_failed = 0
    
    for repo, workflows in queues.items():
        running = sum(1 for w in workflows if w.get("status") == "in_progress")
        failed = sum(1 for w in workflows if w.get("conclusion") in ["failure", "timed_out", "cancelled"])
        
        total_workflows += len(workflows)
        total_running += running
        total_failed += failed
        
        print(f"\n{repo}:")
        print(f"  Total workflows: {len(workflows)}")
        print(f"  Running: {running}")
        print(f"  Failed: {failed}")
        
        # Show some workflow details
        if workflows:
            print("  Recent workflows:")
            for workflow in workflows[:3]:
                status = workflow.get("status", "unknown")
                conclusion = workflow.get("conclusion", "pending")
                name = workflow.get("workflowName", workflow.get("name", "Unknown"))
                print(f"    - {name}: {status}/{conclusion}")
    
    print("\n" + "-" * 80)
    print(f"Total: {total_workflows} workflows ({total_running} running, {total_failed} failed)")
    
    # Step 5: Provision runners
    print("\nStep 5: Provisioning self-hosted runners...")
    runner_mgr = RunnerManager(gh)
    
    # Get system capacity
    cores = runner_mgr.get_system_cores()
    print(f"✓ System has {cores} CPU cores")
    
    # Provision runners (limit to available cores)
    print(f"\nProvisioning up to {cores} runners based on workflow load...")
    provisioning = runner_mgr.provision_runners_for_queue(queues, max_runners=cores)
    
    print(f"✓ Provisioned {len(provisioning)} runner registration tokens")
    
    # Display provisioning results
    print("\nProvisioning Results:")
    print("-" * 80)
    for repo, status in provisioning.items():
        if status.get("status") == "token_generated":
            print(f"\n✓ {repo}:")
            print(f"  Token: {status['token'][:20]}...")
            print(f"  Running workflows: {status['running_workflows']}")
            print(f"  Failed workflows: {status['failed_workflows']}")
            print(f"  Total workflows: {status['total_workflows']}")
        else:
            print(f"\n✗ {repo}:")
            print(f"  Error: {status.get('error', 'Unknown error')}")
    
    # Step 6: Save results
    print("\nStep 6: Saving results to JSON files...")
    
    # Save workflow queues
    with open("/tmp/workflow_queues.json", "w") as f:
        json.dump(queues, f, indent=2, default=str)
    print("✓ Saved workflow queues to /tmp/workflow_queues.json")
    
    # Save provisioning results
    with open("/tmp/runner_tokens.json", "w") as f:
        # Don't save full tokens for security
        sanitized = {
            repo: {
                **status,
                "token": status.get("token", "")[:20] + "..." if status.get("token") else None
            }
            for repo, status in provisioning.items()
        }
        json.dump(sanitized, f, indent=2)
    print("✓ Saved provisioning results to /tmp/runner_tokens.json")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Repositories processed: {len(queues)}")
    print(f"Total workflows: {total_workflows}")
    print(f"Runners provisioned: {len(provisioning)}")
    print(f"System capacity: {cores} cores")
    print("\nNext steps:")
    print("1. Use the tokens to configure self-hosted runners")
    print("2. Monitor workflow execution in the dashboard:")
    print("   ipfs-accelerate mcp start --dashboard --open-browser")
    print("3. View results in:")
    print("   - /tmp/workflow_queues.json")
    print("   - /tmp/runner_tokens.json")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
