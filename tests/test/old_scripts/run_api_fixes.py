#!/usr/bin/env python
"""
Run API implementation fixes and then verify the implementation status.
This script serves as a convenient wrapper for:
    1. Running fix_api_implementations.py
    2. Checking the implementation status
    3. Displaying a summary of the changes
    """

    import os
    import sys
    import subprocess
    import argparse
    import json
    from pathlib import Path
    import datetime

def run_command(cmd):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {}}}}' '.join(cmd)}: {}}}}e}")
        print(f"Output: {}}}}e.stdout}")
        print(f"Error: {}}}}e.stderr}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Run API fixes and check implementation status")
    parser.add_argument("--no-fix", action="store_true", help="Skip running fixes, only check status")
    parser.add_argument("--apis", nargs="+", help="Which APIs to fix (default: all)", 
    choices=["gemini", "hf_tei", "hf_tgi", "llvm", "s3_kit", "opea", "all"],
    default=["all"]),
    parser.add_argument("--update-status", action="store_true", help="Update status files")
    parser.add_argument("--detailed", action="store_true", help="Show detailed API information")
    parser.add_argument("--output", help="Output file for report (default: console only)")
    
    args = parser.parse_args()
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Store results for report
    report = {}}
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "apis_fixed": [],
    "initial_status": {}}},
    "final_status": {}}},
    "changes": {}}}
    }
    
    # Check initial status
    print("\n=== Initial API Implementation Status ===")
    initial_status_output = run_command([sys.executable, str(script_dir / "check_api_implementation.py")]),,
    if not initial_status_output:
        print("Failed to check initial API implementation status")
    return 1
    
    # Store initial status for report
    try:
        for line in initial_status_output.strip().split('\n'):
            if ':' in line and not line.startswith('{}}'):
                api, status = line.split(':', 1)
                api = api.strip()
                status = status.strip()
                report["initial_status"][api], = status,
    except Exception as e:
        print(f"Error parsing initial status: {}}}}e}")
    
    # Skip fixes if requested:::
    if not args.no_fix:
        # Build command for fix_api_implementations.py
        cmd = [sys.executable, str(script_dir / "fix_api_implementations.py")]
        ,
        # Add APIs if specified:
        if args.apis != ["all"]:,
        cmd.extend(["--apis"] + args.apis)
        ,
        # Add update-status flag if requested:::
        if args.update_status:
            cmd.append("--update-status")
        
        # Run the fixes
            print("\n=== Running API Implementation Fixes ===")
            fix_output = run_command(cmd)
        if not fix_output:
            print("Failed to run API implementation fixes")
            return 1
        
        # Parse fixed APIs from output
        for line in fix_output.strip().split('\n'):
            if ': FIXED' in line:
                api = line.split(':', 1)[0].strip(),
                report["apis_fixed"].append(api)
                ,
    # Check final status
                print("\n=== Final API Implementation Status ===")
                final_status_output = run_command([sys.executable, str(script_dir / "check_api_implementation.py")]),,
    if not final_status_output:
        print("Failed to check final API implementation status")
                return 1
    
    # Store final status for report
    try:
        for line in final_status_output.strip().split('\n'):
            if ':' in line and not line.startswith('{}}'):
                api, status = line.split(':', 1)
                api = api.strip()
                status = status.strip()
                report["final_status"][api],,, = status,
    except Exception as e:
        print(f"Error parsing final status: {}}}}e}")
    
    # Calculate changes
        for api in report["initial_status"]:,
        if api in report["final_status"]:,
        initial = report["initial_status"][api],
        final = report["final_status"][api],,,
            if initial != final:
                report["changes"],,[api] = {}},
                "from": initial,
                "to": final
                }
    
    # Count complete APIs
                initial_complete = sum(1 for status in report["initial_status"].values() if status == "COMPLETE"),
                final_complete = sum(1 for status in report["final_status"].values() if status == "COMPLETE"),
                total_apis = len(report["final_status"])
                ,
    # Generate report
    print("\n=== API Implementation Report ==="):
        print(f"Timestamp: {}}}}report['timestamp']}"),
        print(f"APIs Fixed: {}}}}', '.join(report['apis_fixed']) if report['apis_fixed'] else 'None'}"):,
        print(f"Initial Complete: {}}}}initial_complete}/{}}}}total_apis} ({}}}}initial_complete/total_apis*100:.1f}%)")
        print(f"Final Complete: {}}}}final_complete}/{}}}}total_apis} ({}}}}final_complete/total_apis*100:.1f}%)")
    
        if report["changes"],,:,
        print("\nChanges:")
        for api, change in report["changes"],,.items():,
        print(f"  {}}}}api}: {}}}}change['from']} -> {}}}}change['to']}")
        ,
    if args.detailed:
        print("\nDetailed API Status:")
        for api in sorted(report["final_status"].keys()):,,
        status = report["final_status"][api],,,
        changed = api in report["changes"],,
        fixed = api in report["apis_fixed"]
        ,
        status_icon = "✅" if status == "COMPLETE" else "⚠️"
        change_info = " (FIXED)" if fixed else " (CHANGED)" if changed else ""
            :
                print(f"  {}}}}status_icon} {}}}}api}: {}}}}status}{}}}}change_info}")
    
    # Write report to file if requested:::
    if args.output:
        try:
            # Format timestamp as filename-safe string if needed:
            if "{}}timestamp}" in args.output:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = args.output.format(timestamp=timestamp_str)
            else:
                output_file = args.output
            
            # Ensure directory exists
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write report
            with open(output_file, 'w') as f:
                f.write(f"# API Implementation Fix Report\n\n")
                f.write(f"Generated: {}}}}report['timestamp']}\n\n")
                ,
                f.write(f"## Summary\n\n")
                f.write(f"- APIs Fixed: {}}}}', '.join(report['apis_fixed']) if report['apis_fixed'] else 'None'}\n"):,
                f.write(f"- Initial Complete: {}}}}initial_complete}/{}}}}total_apis} ({}}}}initial_complete/total_apis*100:.1f}%)\n")
                f.write(f"- Final Complete: {}}}}final_complete}/{}}}}total_apis} ({}}}}final_complete/total_apis*100:.1f}%)\n")
                
                if report["changes"],,:,
                f.write(f"\n## Changes\n\n")
                for api, change in report["changes"],,.items():,
                f.write(f"- {}}}}api}: {}}}}change['from']} -> {}}}}change['to']}\n")
                ,
                f.write(f"\n## Complete API Status\n\n")
                for api in sorted(report["final_status"].keys()):,,
                status = report["final_status"][api],,,
                changed = api in report["changes"],,
                fixed = api in report["apis_fixed"]
                ,
                status_icon = "✅" if status == "COMPLETE" else "⚠️"
                change_info = " (FIXED)" if fixed else " (CHANGED)" if changed else ""
                    :
                        f.write(f"- {}}}}status_icon} {}}}}api}: {}}}}status}{}}}}change_info}\n")
                
                        print(f"Report written to {}}}}output_file}")
        except Exception as e:
            print(f"Error writing report to {}}}}args.output}: {}}}}e}")
    
                        return 0

if __name__ == "__main__":
    sys.exit(main())