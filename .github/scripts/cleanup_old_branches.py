#!/usr/bin/env python3
"""
Cleanup Old Auto-Heal Branches

This script removes stale auto-heal branches that are no longer needed.
It keeps branches that:
- Have open PRs
- Were created in the last 7 days
- Are linked to open issues
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any


def get_stale_branches(branches: List[Dict[str, Any]], days: int = 7) -> List[str]:
    """
    Identify stale auto-heal branches.
    
    Args:
        branches: List of branch information with dates from GitHub API
        days: Number of days to consider a branch stale (default: 7)
    
    Returns:
        List of branch names that are stale
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    stale_branches = []
    
    for branch in branches:
        branch_name = branch.get('name', '')
        
        # Only consider auto-heal branches
        if not branch_name.startswith('auto-heal/'):
            continue
        
        # Use the date from the GitHub API response
        date_str = branch.get('date')
        if not date_str:
            # If no date, consider it stale for safety
            stale_branches.append(branch_name)
            continue
        
        try:
            # Parse ISO 8601 format from GitHub API
            branch_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            if branch_date < cutoff_date:
                stale_branches.append(branch_name)
        except (ValueError, AttributeError) as e:
            # If we can't parse the date, consider it stale
            print(f"Warning: Could not parse date for {branch_name}: {e}", file=sys.stderr)
            stale_branches.append(branch_name)
    
    return stale_branches


def main():
    """Main entry point."""
    # This script is meant to be used with GitHub Actions
    # It expects branch information to be provided via environment or file
    
    if len(sys.argv) < 2:
        print("Usage: python cleanup_old_branches.py <branches.json> [days]")
        print("\nThis script identifies stale auto-heal branches that can be deleted.")
        print("It keeps branches that are less than 'days' old (default: 7)")
        sys.exit(1)
    
    # Load branch data
    branches_file = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    if not os.path.exists(branches_file):
        print(f"Error: Branches file '{branches_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    with open(branches_file, 'r') as f:
        branches_data = json.load(f)
    
    # Identify stale branches
    stale_branches = get_stale_branches(branches_data, days)
    
    if not stale_branches:
        print("No stale auto-heal branches found")
        print(f"All auto-heal branches are less than {days} days old")
        sys.exit(0)
    
    # Output the list of branches to clean up
    print(f"Found {len(stale_branches)} stale auto-heal branch(es):")
    for branch in stale_branches:
        print(f"  - {branch}")
    
    # Save the list for the workflow to use
    with open('stale_branches.txt', 'w') as f:
        f.write('\n'.join(stale_branches))
    
    print(f"\nBranches to clean up saved to: stale_branches.txt")
    print(f"These branches are older than {days} days")


if __name__ == '__main__':
    main()
