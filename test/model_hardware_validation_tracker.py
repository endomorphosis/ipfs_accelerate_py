"""
Model Hardware Validation Tracker

This script tracks the validation status of key HuggingFace models across different
hardware platforms, providing a centralized database of test results, compatibility
information, and performance metrics.

The tracker maintains:
1. Test result history for each model-hardware combination
2. Current implementation status (real, mock, or incompatible)
3. Performance benchmarks and hardware requirements
4. Known issues and workarounds

Usage:
    python model_hardware_validation_tracker.py --update-status
    python model_hardware_validation_tracker.py --add-result [result_file]
    python model_hardware_validation_tracker.py --generate-report
    python model_hardware_validation_tracker.py --visualize
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Import model and hardware definitions
try:
    from test_comprehensive_hardware_coverage import KEY_MODELS, HARDWARE_PLATFORMS
except ImportError:
    print("Error: Could not import from test_comprehensive_hardware_coverage.py")
    print("Make sure it exists in the same directory")
    sys.exit(1)

# Configuration
CONFIG = {
    "database_file": "model_hardware_validation_db.json",
    "database_backup_dir": "validation_db_backups",
    "report_output_dir": "validation_reports",
    "visualization_output_dir": "validation_visualizations",
    "backup_frequency": 10,  # Create backup every N updates
}

class ModelHardwareValidationTracker:
    """
    Tracks validation status for model-hardware combinations
    """
    
    def __init__(self, database_file: str = CONFIG["database_file"]):
        """
        Initialize the tracker with the specified database file.
        
        Args:
            database_file: Path to the JSON database file
        """
        self.database_file = database_file
        self.database = self._load_database()
        self.update_count = 0
    
    def _load_database(self) -> Dict:
        """
        Load the validation database from disk.
        
        Returns:
            Dict: The loaded database or an initialized empty database
        """
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Database file {self.database_file} is corrupted")
                return self._initialize_database()
        else:
            return self._initialize_database()
    
    def _initialize_database(self) -> Dict:
        """
        Initialize an empty validation database.
        
        Returns:
            Dict: An initialized empty database
        """
        database = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
                "update_count": 0
            },
            "models": {},
            "hardware": {},
            "validation_results": {}
        }
        
        # Initialize model entries
        for model_key, model_info in KEY_MODELS.items():
            database["models"][model_key] = {
                "name": model_info["name"],
                "models": model_info["models"],
                "category": model_info["category"],
                "last_updated": datetime.now().isoformat()
            }
        
        # Initialize hardware entries
        for hw_key, hw_info in HARDWARE_PLATFORMS.items():
            database["hardware"][hw_key] = {
                "name": hw_info["name"],
                "flag": hw_info["flag"],
                "last_updated": datetime.now().isoformat()
            }
        
        # Initialize validation result entries
        for model_key in KEY_MODELS:
            database["validation_results"][model_key] = {}
            for hw_key in HARDWARE_PLATFORMS:
                is_compatible = model_key in HARDWARE_PLATFORMS[hw_key]["compatibility"]
                status = "untested"
                if not is_compatible:
                    status = "incompatible"
                
                database["validation_results"][model_key][hw_key] = {
                    "status": status,
                    "implementation_type": "none",
                    "last_test_date": None,
                    "test_history": [],
                    "known_issues": [],
                    "performance": {},
                    "requirements": {},
                    "notes": ""
                }
        
        return database
    
    def _save_database(self):
        """
        Save the validation database to disk.
        """
        # Update metadata
        self.database["metadata"]["last_updated"] = datetime.now().isoformat()
        self.database["metadata"]["update_count"] += 1
        self.update_count += 1
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.database_file)), exist_ok=True)
        
        # Save the database
        with open(self.database_file, 'w') as f:
            json.dump(self.database, f, indent=2)
        
        # Create backup if needed
        if self.update_count % CONFIG["backup_frequency"] == 0:
            self._create_backup()
    
    def _create_backup(self):
        """
        Create a backup of the validation database.
        """
        backup_dir = CONFIG["database_backup_dir"]
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"validation_db_backup_{timestamp}.json")
        
        with open(backup_file, 'w') as f:
            json.dump(self.database, f, indent=2)
        
        print(f"Created database backup: {backup_file}")
    
    def update_status(self, model_key: str, hardware_key: str, status: str,
                      implementation_type: str = None, notes: str = None):
        """
        Update the validation status for a model-hardware combination.
        
        Args:
            model_key: The model key
            hardware_key: The hardware platform key
            status: The validation status (pass, fail, untested, incompatible)
            implementation_type: The implementation type (real, mock, none)
            notes: Additional notes
        """
        if model_key not in self.database["models"]:
            print(f"Error: Unknown model key: {model_key}")
            return
        
        if hardware_key not in self.database["hardware"]:
            print(f"Error: Unknown hardware key: {hardware_key}")
            return
        
        # Validate status
        valid_statuses = ["pass", "fail", "untested", "incompatible"]
        if status not in valid_statuses:
            print(f"Error: Invalid status: {status}. Must be one of {valid_statuses}")
            return
        
        # Validate implementation type
        if implementation_type is not None:
            valid_types = ["real", "mock", "none"]
            if implementation_type not in valid_types:
                print(f"Error: Invalid implementation type: {implementation_type}. Must be one of {valid_types}")
                return
        
        # Get current entry
        validation_entry = self.database["validation_results"][model_key][hardware_key]
        
        # Update test history
        test_date = datetime.now().isoformat()
        
        # Don't add history for incompatible combinations
        if status != "incompatible":
            history_entry = {
                "date": test_date,
                "status": status
            }
            if implementation_type:
                history_entry["implementation_type"] = implementation_type
            if notes:
                history_entry["notes"] = notes
            
            validation_entry["test_history"].append(history_entry)
        
        # Update current status
        validation_entry["status"] = status
        validation_entry["last_test_date"] = test_date
        
        if implementation_type:
            validation_entry["implementation_type"] = implementation_type
        
        if notes:
            validation_entry["notes"] = notes
        
        # Update last_updated
        self.database["models"][model_key]["last_updated"] = test_date
        self.database["hardware"][hardware_key]["last_updated"] = test_date
        
        # Save changes
        self._save_database()
        
        print(f"Updated status for {model_key} on {hardware_key}: {status}")
    
    def add_test_result(self, result_file: str):
        """
        Add a test result from a JSON result file.
        
        Args:
            result_file: Path to the JSON result file
        """
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading result file: {e}")
            return
        
        # Extract required fields
        if "model_key" not in result or "hardware_key" not in result or "status" not in result:
            print(f"Error: Result file missing required fields: model_key, hardware_key, status")
            return
        
        model_key = result["model_key"]
        hardware_key = result["hardware_key"]
        status = result["status"]
        
        # Optional fields
        implementation_type = result.get("implementation_type")
        notes = result.get("notes")
        
        # Update status
        self.update_status(model_key, hardware_key, status, implementation_type, notes)
        
        # Add performance metrics if available
        if "performance" in result:
            self.add_performance_metrics(model_key, hardware_key, result["performance"])
        
        # Add hardware requirements if available
        if "requirements" in result:
            self.add_hardware_requirements(model_key, hardware_key, result["requirements"])
        
        # Add known issues if available
        if "known_issues" in result:
            for issue in result["known_issues"]:
                self.add_known_issue(model_key, hardware_key, issue)
    
    def add_performance_metrics(self, model_key: str, hardware_key: str, metrics: Dict):
        """
        Add performance metrics for a model-hardware combination.
        
        Args:
            model_key: The model key
            hardware_key: The hardware platform key
            metrics: Performance metrics
        """
        if model_key not in self.database["models"]:
            print(f"Error: Unknown model key: {model_key}")
            return
        
        if hardware_key not in self.database["hardware"]:
            print(f"Error: Unknown hardware key: {hardware_key}")
            return
        
        # Get current entry
        validation_entry = self.database["validation_results"][model_key][hardware_key]
        
        # Update performance metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_entry["performance"][timestamp] = metrics
        
        # Save changes
        self._save_database()
        
        print(f"Added performance metrics for {model_key} on {hardware_key}")
    
    def add_hardware_requirements(self, model_key: str, hardware_key: str, requirements: Dict):
        """
        Add hardware requirements for a model-hardware combination.
        
        Args:
            model_key: The model key
            hardware_key: The hardware platform key
            requirements: Hardware requirements
        """
        if model_key not in self.database["models"]:
            print(f"Error: Unknown model key: {model_key}")
            return
        
        if hardware_key not in self.database["hardware"]:
            print(f"Error: Unknown hardware key: {hardware_key}")
            return
        
        # Get current entry
        validation_entry = self.database["validation_results"][model_key][hardware_key]
        
        # Update hardware requirements
        validation_entry["requirements"] = requirements
        
        # Save changes
        self._save_database()
        
        print(f"Added hardware requirements for {model_key} on {hardware_key}")
    
    def add_known_issue(self, model_key: str, hardware_key: str, issue: Dict):
        """
        Add a known issue for a model-hardware combination.
        
        Args:
            model_key: The model key
            hardware_key: The hardware platform key
            issue: Issue description
        """
        if model_key not in self.database["models"]:
            print(f"Error: Unknown model key: {model_key}")
            return
        
        if hardware_key not in self.database["hardware"]:
            print(f"Error: Unknown hardware key: {hardware_key}")
            return
        
        # Get current entry
        validation_entry = self.database["validation_results"][model_key][hardware_key]
        
        # Add timestamp if not provided
        if "date" not in issue:
            issue["date"] = datetime.now().isoformat()
        
        # Add the issue
        validation_entry["known_issues"].append(issue)
        
        # Save changes
        self._save_database()
        
        print(f"Added known issue for {model_key} on {hardware_key}: {issue.get('description', '')}")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a validation report.
        
        Args:
            output_file: Path to save the report, if None, return as string
            
        Returns:
            str: The generated report if output_file is None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = [
            f"# Model Hardware Validation Report ({timestamp})",
            "",
            "## Overview",
            ""
        ]
        
        # Count statistics
        total_combinations = 0
        passed_combinations = 0
        failed_combinations = 0
        untested_combinations = 0
        incompatible_combinations = 0
        
        for model_key in self.database["models"]:
            for hw_key in self.database["hardware"]:
                validation = self.database["validation_results"][model_key][hw_key]
                status = validation["status"]
                
                total_combinations += 1
                if status == "pass":
                    passed_combinations += 1
                elif status == "fail":
                    failed_combinations += 1
                elif status == "untested":
                    untested_combinations += 1
                elif status == "incompatible":
                    incompatible_combinations += 1
        
        report.extend([
            f"- Total model-hardware combinations: {total_combinations}",
            f"- Passed: {passed_combinations} ({passed_combinations/total_combinations*100:.1f}%)",
            f"- Failed: {failed_combinations} ({failed_combinations/total_combinations*100:.1f}%)",
            f"- Untested: {untested_combinations} ({untested_combinations/total_combinations*100:.1f}%)",
            f"- Incompatible: {incompatible_combinations} ({incompatible_combinations/total_combinations*100:.1f}%)",
            "",
            "## Validation Status by Model Category",
            ""
        ])
        
        # Group by category
        categories = {}
        for model_key, model_info in self.database["models"].items():
            category = model_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(model_key)
        
        # Generate tables by category
        for category, model_keys in categories.items():
            report.append(f"### {category.replace('_', ' ').title()}")
            report.append("")
            
            for model_key in model_keys:
                model_name = self.database["models"][model_key]["name"]
                report.append(f"#### {model_name}")
                report.append("")
                report.append("| Hardware | Status | Implementation | Last Tested | Notes |")
                report.append("|----------|--------|----------------|-------------|-------|")
                
                for hw_key in self.database["hardware"]:
                    hw_name = self.database["hardware"][hw_key]["name"]
                    validation = self.database["validation_results"][model_key][hw_key]
                    
                    status = validation["status"]
                    implementation = validation["implementation_type"]
                    last_tested = "Never"
                    if validation["last_test_date"]:
                        last_tested = validation["last_test_date"].split("T")[0]  # Just the date part
                    
                    notes = validation["notes"] or ""
                    
                    # Format status with emoji
                    status_display = {
                        "pass": "✅ Pass",
                        "fail": "❌ Fail",
                        "untested": "⚠️ Untested",
                        "incompatible": "🚫 Incompatible"
                    }.get(status, status)
                    
                    # Format implementation type
                    implementation_display = {
                        "real": "Real",
                        "mock": "Mock",
                        "none": "None"
                    }.get(implementation, implementation)
                    
                    report.append(f"| {hw_name} | {status_display} | {implementation_display} | {last_tested} | {notes} |")
                
                report.append("")
            
            report.append("")
        
        # Add known issues section
        report.extend([
            "## Known Issues",
            ""
        ])
        
        issues_found = False
        for model_key in self.database["models"]:
            model_name = self.database["models"][model_key]["name"]
            
            for hw_key in self.database["hardware"]:
                hw_name = self.database["hardware"][hw_key]["name"]
                validation = self.database["validation_results"][model_key][hw_key]
                
                if validation["known_issues"]:
                    issues_found = True
                    report.append(f"### {model_name} on {hw_name}")
                    report.append("")
                    
                    for issue in validation["known_issues"]:
                        date = issue.get("date", "").split("T")[0]
                        description = issue.get("description", "No description")
                        workaround = issue.get("workaround", "None")
                        
                        report.append(f"- **{date}**: {description}")
                        if workaround != "None":
                            report.append(f"  - Workaround: {workaround}")
                    
                    report.append("")
        
        if not issues_found:
            report.append("No known issues found.")
            report.append("")
        
        # Generate the report
        report_text = "\n".join(report)
        
        # Save to file if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
            return None
        
        return report_text
    
    def create_visualization(self, output_dir: Optional[str] = None):
        """
        Create visualizations of the validation status.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = CONFIG["visualization_output_dir"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataframe for easier plotting
        data = []
        for model_key, model_info in self.database["models"].items():
            for hw_key, hw_info in self.database["hardware"].items():
                validation = self.database["validation_results"][model_key][hw_key]
                
                data.append({
                    "model_key": model_key,
                    "model_name": model_info["name"],
                    "model_category": model_info["category"],
                    "hardware_key": hw_key,
                    "hardware_name": hw_info["name"],
                    "status": validation["status"],
                    "implementation": validation["implementation_type"]
                })
        
        df = pd.DataFrame(data)
        
        # Create heatmap of validation status
        self._create_status_heatmap(df, output_dir)
        
        # Create category-based bar chart
        self._create_category_barchart(df, output_dir)
        
        # Create implementation type pie chart
        self._create_implementation_piechart(df, output_dir)
        
        print(f"Visualizations saved to {output_dir}")
    
    def _create_status_heatmap(self, df: pd.DataFrame, output_dir: str):
        """
        Create a heatmap of validation status.
        
        Args:
            df: DataFrame with validation data
            output_dir: Directory to save visualizations
        """
        # Create pivot table for heatmap
        status_map = {
            "pass": 3,
            "fail": 1,
            "untested": 0,
            "incompatible": -1
        }
        
        df["status_value"] = df["status"].map(status_map)
        
        pivot = df.pivot_table(
            index="model_name", 
            columns="hardware_name", 
            values="status_value",
            aggfunc="first"
        )
        
        # Sort models by category
        model_order = []
        categories = df[["model_category", "model_name"]].drop_duplicates()
        categories = categories.sort_values(by="model_category")
        
        for _, row in categories.iterrows():
            model_order.append(row["model_name"])
        
        pivot = pivot.reindex(model_order)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Custom colormap for status values
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['lightgray', 'red', 'white', 'green'])
        norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 3.5], cmap.N)
        
        ax = plt.gca()
        im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                status_value = pivot.iloc[i, j]
                text = {
                    3: "Pass",
                    1: "Fail",
                    0: "Untested",
                    -1: "N/A"
                }.get(status_value, "")
                
                ax.text(j, i, text, ha="center", va="center", 
                       color="black" if status_value in [0, 3] else "white")
        
        # Set labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Pass'),
            Patch(facecolor='red', label='Fail'),
            Patch(facecolor='white', label='Untested'),
            Patch(facecolor='lightgray', label='Incompatible')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title("Model-Hardware Validation Status")
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"validation_status_heatmap_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Status heatmap saved to {filename}")
    
    def _create_category_barchart(self, df: pd.DataFrame, output_dir: str):
        """
        Create a bar chart of validation status by category.
        
        Args:
            df: DataFrame with validation data
            output_dir: Directory to save visualizations
        """
        # Calculate pass rates by category
        category_stats = df.groupby(["model_category", "status"]).size().unstack(fill_value=0)
        
        if "pass" not in category_stats.columns:
            category_stats["pass"] = 0
        
        if "fail" not in category_stats.columns:
            category_stats["fail"] = 0
        
        if "untested" not in category_stats.columns:
            category_stats["untested"] = 0
        
        if "incompatible" not in category_stats.columns:
            category_stats["incompatible"] = 0
        
        # Calculate total (excluding incompatible)
        category_stats["total_testable"] = category_stats["pass"] + category_stats["fail"] + category_stats["untested"]
        category_stats["pass_rate"] = category_stats["pass"] / category_stats["total_testable"] * 100
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        categories = category_stats.index.tolist()
        categories = [c.replace("_", " ").title() for c in categories]
        
        x = range(len(categories))
        width = 0.2
        
        plt.bar([i - width*1.5 for i in x], category_stats["pass"], width, label="Pass", color="green")
        plt.bar([i - width*0.5 for i in x], category_stats["fail"], width, label="Fail", color="red")
        plt.bar([i + width*0.5 for i in x], category_stats["untested"], width, label="Untested", color="gray")
        plt.bar([i + width*1.5 for i in x], category_stats["incompatible"], width, label="Incompatible", color="lightgray")
        
        # Add pass rate as text
        for i, rate in enumerate(category_stats["pass_rate"]):
            plt.text(i, category_stats["pass"].iloc[i] + 0.5, f"{rate:.1f}%", ha="center")
        
        plt.xlabel("Model Category")
        plt.ylabel("Count")
        plt.title("Validation Status by Model Category")
        plt.xticks(x, categories)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"validation_category_barchart_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Category bar chart saved to {filename}")
    
    def _create_implementation_piechart(self, df: pd.DataFrame, output_dir: str):
        """
        Create a pie chart of implementation types.
        
        Args:
            df: DataFrame with validation data
            output_dir: Directory to save visualizations
        """
        # Filter out incompatible combinations
        df_compatible = df[df["status"] != "incompatible"]
        
        # Count implementation types
        implementation_counts = df_compatible["implementation"].value_counts()
        
        # Create plot
        plt.figure(figsize=(8, 8))
        
        labels = implementation_counts.index.tolist()
        labels = [l.capitalize() if l != "none" else "Not Implemented" for l in labels]
        
        colors = ["green", "orange", "lightgray"]
        
        plt.pie(implementation_counts, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title("Implementation Types (Compatible Combinations Only)")
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"implementation_piechart_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Implementation pie chart saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Model Hardware Validation Tracker")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("--update-status", action="store_true", help="Update status for model-hardware combination")
    group.add_argument("--add-result", help="Add result from JSON file")
    group.add_argument("--generate-report", action="store_true", help="Generate validation report")
    group.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    # Arguments for update-status
    parser.add_argument("--model", help="Model key")
    parser.add_argument("--hardware", help="Hardware key")
    parser.add_argument("--status", choices=["pass", "fail", "untested", "incompatible"], help="Validation status")
    parser.add_argument("--implementation", choices=["real", "mock", "none"], help="Implementation type")
    parser.add_argument("--notes", help="Additional notes")
    
    # Arguments for generate-report
    parser.add_argument("--output", help="Output file for report")
    
    # Arguments for visualize
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    tracker = ModelHardwareValidationTracker()
    
    if args.update_status:
        if not args.model or not args.hardware or not args.status:
            parser.error("--update-status requires --model, --hardware, and --status")
        
        tracker.update_status(
            args.model,
            args.hardware,
            args.status,
            args.implementation,
            args.notes
        )
    
    elif args.add_result:
        tracker.add_test_result(args.add_result)
    
    elif args.generate_report:
        output_file = args.output
        if not output_file:
            # Create default output file
            os.makedirs(CONFIG["report_output_dir"], exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(CONFIG["report_output_dir"], f"validation_report_{timestamp}.md")
        
        tracker.generate_report(output_file)
    
    elif args.visualize:
        tracker.create_visualization(args.output_dir)

if __name__ == "__main__":
    main()
"""