#!/usr/bin/env python3

"""
Generate Abstract Syntax Tree (AST) analysis for test files.

This script analyzes Python test files to generate a comprehensive AST report,
identifying patterns, structures, and potential refactoring opportunities.
"""

import os
import sys
import ast
import json
import glob
import logging
import argparse
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default directories
TEST_DIRS = [
    './fixed_tests',
    './minimal_tests',
    '.'  # Root directory for core test files
]

class TestFileAnalyzer:
    """Analyzer for test file AST structures."""
    
    def __init__(self, test_dirs: List[str] = TEST_DIRS, exclude_patterns: List[str] = None):
        self.test_dirs = test_dirs
        self.exclude_patterns = exclude_patterns or ['__pycache__', '.git', '.bak', '.log']
        
        # Counters and collections
        self.total_files = 0
        self.total_methods = 0
        self.total_classes = 0
        self.class_counts = defaultdict(int)
        self.method_counts = defaultdict(int)
        self.import_counts = defaultdict(int)
        self.class_method_map = defaultdict(list)
        self.method_patterns = defaultdict(int)
        self.test_structure_patterns = []
        self.duplicate_structure_candidates = []
        self.file_complexities = {}
        
    def find_test_files(self) -> List[str]:
        """Find all Python test files in specified directories."""
        test_files = []
        
        for directory in self.test_dirs:
            if not os.path.exists(directory):
                logger.warning(f"Directory {directory} not found, skipping")
                continue
                
            pattern = os.path.join(directory, "test_*.py")
            directory_files = glob.glob(pattern)
            
            # Apply exclusion patterns
            for file_path in directory_files:
                if not any(exclude in file_path for exclude in self.exclude_patterns):
                    test_files.append(file_path)
        
        logger.info(f"Found {len(test_files)} test files")
        return test_files
    
    def parse_file_ast(self, file_path: str) -> Optional[ast.AST]:
        """Parse a file and return its AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file_path)
            return tree
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1  # Base complexity
        
        # Count branches that increase complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And):
                complexity += len(child.values) - 1
        
        return complexity
    
    def get_method_signature(self, node: ast.FunctionDef) -> str:
        """Create a signature for a method based on its structure."""
        # Basic structure: argument count + return type + decorators
        arg_count = len([arg for arg in node.args.args if arg.arg != 'self'])
        
        # Check for common patterns
        has_try_except = any(isinstance(n, ast.Try) for n in ast.walk(node))
        has_assertions = any(isinstance(n, ast.Assert) for n in ast.walk(node))
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        
        # Look for from_pretrained calls
        has_from_pretrained = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and hasattr(child, 'func'):
                func = child.func
                if (isinstance(func, ast.Attribute) and func.attr == 'from_pretrained'):
                    has_from_pretrained = True
                    break
        
        # Pattern signature
        signature = f"args:{arg_count}"
        if has_try_except:
            signature += ",try-except"
        if has_assertions:
            signature += ",assertions"
        if has_return:
            signature += ",return"
        if has_from_pretrained:
            signature += ",from_pretrained"
            
        return signature
    
    def analyze_test_file(self, file_path: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze a test file's AST and extract key information."""
        file_info = {
            "path": file_path,
            "classes": [],
            "imports": [],
            "global_methods": [],
            "complexity": 0,
            "encoding": None,
            "lines_of_code": 0,
            "has_mock_usage": False,
            "from_pretrained_count": 0
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_info["lines_of_code"] = len(f.readlines())
        
        # Track from_pretrained calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node, 'func'):
                func = node.func
                if (isinstance(func, ast.Attribute) and 
                    func.attr == 'from_pretrained'):
                    file_info["from_pretrained_count"] += 1
        
        # Find encoding comment
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, str) and "coding" in value:
                    file_info["encoding"] = value
        
        # Look for mock usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for name in node.names:
                    if "mock" in name.name.lower():
                        file_info["has_mock_usage"] = True
                    
                    # Track import
                    import_name = name.name
                    if isinstance(node, ast.ImportFrom) and node.module:
                        import_name = f"{node.module}.{import_name}"
                    file_info["imports"].append(import_name)
                    self.import_counts[import_name] += 1
        
        # Analyze classes
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self.total_classes += 1
                self.class_counts[node.name] += 1
                
                class_info = {
                    "name": node.name,
                    "methods": [],
                    "attributes": [],
                    "base_classes": [base.id if isinstance(base, ast.Name) else "complex_base" for base in node.bases],
                    "complexity": self.calculate_complexity(node)
                }
                
                file_info["complexity"] += class_info["complexity"]
                
                # Analyze class methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        self.total_methods += 1
                        self.method_counts[item.name] += 1
                        
                        # Skip internal methods
                        if not item.name.startswith('_') or item.name == '__init__':
                            self.class_method_map[node.name].append(item.name)
                        
                        method_info = {
                            "name": item.name,
                            "args": [arg.arg for arg in item.args.args],
                            "complexity": self.calculate_complexity(item),
                            "signature": self.get_method_signature(item)
                        }
                        
                        # Track method signature patterns
                        self.method_patterns[method_info["signature"]] += 1
                        
                        class_info["methods"].append(method_info)
                        file_info["complexity"] += method_info["complexity"]
                    
                    elif isinstance(item, ast.Assign):
                        # Extract class attributes
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_info["attributes"].append(target.id)
                
                file_info["classes"].append(class_info)
            
            # Global methods (not in a class)
            elif isinstance(node, ast.FunctionDef):
                self.total_methods += 1
                self.method_counts[node.name] += 1
                
                method_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "complexity": self.calculate_complexity(node),
                    "signature": self.get_method_signature(node)
                }
                
                self.method_patterns[method_info["signature"]] += 1
                file_info["global_methods"].append(method_info)
                file_info["complexity"] += method_info["complexity"]
        
        # Store file complexity
        self.file_complexities[file_path] = file_info["complexity"]
        
        return file_info
    
    def analyze_all_files(self) -> List[Dict[str, Any]]:
        """Analyze all test files and create a comprehensive report."""
        test_files = self.find_test_files()
        self.total_files = len(test_files)
        
        file_analyses = []
        
        for file_path in test_files:
            tree = self.parse_file_ast(file_path)
            if tree:
                analysis = self.analyze_test_file(file_path, tree)
                file_analyses.append(analysis)
                
                # Track structure patterns for potential duplication
                structure_fingerprint = self._get_file_structure_fingerprint(analysis)
                self.test_structure_patterns.append({
                    "file": file_path,
                    "fingerprint": structure_fingerprint
                })
        
        # Find potential duplicate structures
        self._find_duplicate_structures()
        
        return file_analyses
    
    def _get_file_structure_fingerprint(self, file_analysis: Dict[str, Any]) -> str:
        """Create a structural fingerprint for a file to detect similar files."""
        class_names = sorted([c["name"] for c in file_analysis["classes"]])
        method_counts = defaultdict(int)
        
        for cls in file_analysis["classes"]:
            for method in cls["methods"]:
                method_counts[method["signature"]] += 1
        
        # Create a consistent fingerprint
        fingerprint = f"classes:{len(class_names)}"
        if class_names:
            fingerprint += f",class_pattern:{'-'.join(class_names)}"
        
        fingerprint += f",methods:{sum(method_counts.values())}"
        
        # Add top 3 method signatures
        top_signatures = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_signatures:
            signatures = [f"{sig}:{count}" for sig, count in top_signatures]
            fingerprint += f",signatures:{'-'.join(signatures)}"
        
        return fingerprint
    
    def _find_duplicate_structures(self):
        """Find potentially duplicate or similar file structures."""
        fingerprint_map = defaultdict(list)
        
        for item in self.test_structure_patterns:
            fingerprint_map[item["fingerprint"]].append(item["file"])
        
        # Find fingerprints with multiple files
        for fingerprint, files in fingerprint_map.items():
            if len(files) > 1:
                self.duplicate_structure_candidates.append({
                    "fingerprint": fingerprint,
                    "files": files,
                    "count": len(files)
                })
    
    def generate_report(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        # Sort potential duplicates by count (most duplicated first)
        self.duplicate_structure_candidates.sort(key=lambda x: x["count"], reverse=True)
        
        # Calculate overall statistics
        total_complexity = sum(self.file_complexities.values())
        avg_complexity = total_complexity / self.total_files if self.total_files else 0
        
        avg_methods_per_class = self.total_methods / self.total_classes if self.total_classes else 0
        
        # Most common method names
        common_methods = sorted(self.method_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Most common class names
        common_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Most common imports
        common_imports = sorted(self.import_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Most common method patterns
        common_patterns = sorted(self.method_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Class-method relationships
        class_methods = {k: v for k, v in self.class_method_map.items() if len(v) > 0}
        
        # Find most complex files
        complex_files = sorted(self.file_complexities.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Files with most from_pretrained calls
        from_pretrained_usage = sorted(
            [(a["path"], a["from_pretrained_count"]) for a in file_analyses],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # Calculate from_pretrained implementation types
        from_pretrained_types = {
            "explicit_method": 0,  # Has test_from_pretrained method
            "alternative_method": 0,  # Has method that calls from_pretrained
            "direct_usage": 0,  # Direct calls to from_pretrained
            "pipeline_usage": 0,  # Uses pipeline API
            "no_usage": 0  # No from_pretrained usage
        }
        
        for analysis in file_analyses:
            # Check for explicit test_from_pretrained method
            has_explicit = any(
                method["name"] == "test_from_pretrained" 
                for cls in analysis["classes"] 
                for method in cls["methods"]
            )
            
            # Check for pipeline usage
            has_pipeline = "pipeline(" in open(analysis["path"], 'r').read()
            
            if has_explicit:
                from_pretrained_types["explicit_method"] += 1
            elif analysis["from_pretrained_count"] > 0:
                # Check if there's a test method that uses from_pretrained
                test_method_calls_from_pretrained = False
                for cls in analysis["classes"]:
                    for method in cls["methods"]:
                        if method["name"].startswith("test_") and "from_pretrained" in method["signature"]:
                            test_method_calls_from_pretrained = True
                            break
                
                if test_method_calls_from_pretrained:
                    from_pretrained_types["alternative_method"] += 1
                else:
                    from_pretrained_types["direct_usage"] += 1
            elif has_pipeline:
                from_pretrained_types["pipeline_usage"] += 1
            else:
                from_pretrained_types["no_usage"] += 1
        
        # Create the report
        report = {
            "generated_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "summary": {
                "total_files": self.total_files,
                "total_classes": self.total_classes,
                "total_methods": self.total_methods,
                "total_complexity": total_complexity,
                "avg_complexity": avg_complexity,
                "avg_methods_per_class": avg_methods_per_class
            },
            "from_pretrained_usage": {
                "implementation_types": from_pretrained_types,
                "top_usage": from_pretrained_usage
            },
            "common_patterns": {
                "method_names": common_methods,
                "class_names": common_classes,
                "imports": common_imports,
                "method_patterns": common_patterns
            },
            "complexity": {
                "complex_files": complex_files
            },
            "potential_duplicates": self.duplicate_structure_candidates,
            "class_method_map": class_methods,
            "detailed_analyses": file_analyses
        }
        
        return report
    
    def run_analysis(self, output_file: str = "ast_analysis_report.json", markdown_file: str = "ast_analysis_report.md"):
        """Run the complete analysis process and generate reports."""
        logger.info("Starting test files analysis...")
        
        # Analyze all files
        file_analyses = self.analyze_all_files()
        
        # Generate report
        report = self.generate_report(file_analyses)
        
        # Write JSON report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis complete. Report written to {output_file}")
        
        # Create markdown report
        self.generate_markdown_report(report, markdown_file)
        logger.info(f"Markdown report written to {markdown_file}")
        
        return report
    
    def generate_markdown_report(self, report: Dict[str, Any], output_file: str):
        """Generate a markdown report from the analysis data."""
        summary = report["summary"]
        from_pretrained_usage = report["from_pretrained_usage"]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Test Codebase AST Analysis Report\n\n")
            f.write(f"Generated on: {report['generated_date']}\n\n")
            
            # Summary
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Files Analyzed:** {summary['total_files']}\n")
            f.write(f"- **Total Classes:** {summary['total_classes']}\n")
            f.write(f"- **Total Methods:** {summary['total_methods']}\n")
            f.write(f"- **Average Methods per Class:** {summary['avg_methods_per_class']:.2f}\n")
            f.write(f"- **Total Cyclomatic Complexity:** {summary['total_complexity']}\n")
            f.write(f"- **Average Complexity per File:** {summary['avg_complexity']:.2f}\n\n")
            
            # from_pretrained implementation types
            impl_types = from_pretrained_usage["implementation_types"]
            f.write(f"## from_pretrained Implementation Types\n\n")
            f.write(f"| Implementation Type | Count | Percentage |\n")
            f.write(f"|---------------------|-------|------------|\n")
            
            for impl_type, count in impl_types.items():
                percentage = count / summary['total_files'] * 100
                f.write(f"| {impl_type.replace('_', ' ').title()} | {count} | {percentage:.1f}% |\n")
            
            f.write(f"\n")
            
            # Potential duplicates 
            f.write(f"## Potential Code Duplication\n\n")
            
            if report["potential_duplicates"]:
                f.write(f"Found {len(report['potential_duplicates'])} groups of potentially similar files:\n\n")
                
                for i, dup in enumerate(report["potential_duplicates"][:10]):  # Show top 10
                    f.write(f"### Group {i+1}: {dup['count']} files\n\n")
                    f.write(f"Structural fingerprint: `{dup['fingerprint']}`\n\n")
                    f.write(f"Files:\n")
                    for file in dup["files"]:
                        f.write(f"- `{file}`\n")
                    f.write(f"\n")
                
                if len(report["potential_duplicates"]) > 10:
                    f.write(f"...and {len(report['potential_duplicates']) - 10} more groups\n\n")
            else:
                f.write(f"No duplicate code structures detected.\n\n")
            
            # Common patterns
            f.write(f"## Common Patterns\n\n")
            
            # Method names
            f.write(f"### Most Common Method Names\n\n")
            f.write(f"| Method Name | Occurrences |\n")
            f.write(f"|-------------|-------------|\n")
            for name, count in report["common_patterns"]["method_names"][:10]:
                f.write(f"| `{name}` | {count} |\n")
            f.write(f"\n")
            
            # Class names
            f.write(f"### Most Common Class Names\n\n")
            f.write(f"| Class Name | Occurrences |\n")
            f.write(f"|------------|-------------|\n")
            for name, count in report["common_patterns"]["class_names"][:10]:
                f.write(f"| `{name}` | {count} |\n")
            f.write(f"\n")
            
            # Method patterns
            f.write(f"### Most Common Method Patterns\n\n")
            f.write(f"| Pattern | Occurrences |\n")
            f.write(f"|---------|-------------|\n")
            for pattern, count in report["common_patterns"]["method_patterns"][:10]:
                f.write(f"| `{pattern}` | {count} |\n")
            f.write(f"\n")
            
            # Complex files
            f.write(f"### Most Complex Files\n\n")
            f.write(f"| File Path | Complexity |\n")
            f.write(f"|-----------|------------|\n")
            for path, complexity in report["complexity"]["complex_files"][:10]:
                f.write(f"| `{path}` | {complexity} |\n")
            f.write(f"\n")
            
            # Refactoring recommendations
            f.write(f"## Refactoring Recommendations\n\n")
            
            # Methods that could be standardized
            if report["common_patterns"]["method_patterns"]:
                pattern, count = report["common_patterns"]["method_patterns"][0]
                coverage = count / summary["total_methods"] * 100
                
                f.write(f"1. **Standardize Method Implementations**: The pattern `{pattern}` appears in {count} methods ({coverage:.1f}% of all methods). Consider standardizing this pattern across the codebase.\n\n")
            
            # Duplicate structures
            if report["potential_duplicates"]:
                top_dup = report["potential_duplicates"][0]
                f.write(f"2. **Refactor Similar Files**: The largest group contains {top_dup['count']} similar files. Consider creating a common base class or utility functions.\n\n")
            
            # from_pretrained standardization
            f.write(f"3. **Standardize from_pretrained Testing**: Currently using {len([t for t, c in impl_types.items() if c > 0])} different approaches for testing from_pretrained(). Consider standardizing to the explicit method pattern.\n\n")
            
            # Complex files
            if report["complexity"]["complex_files"]:
                path, complexity = report["complexity"]["complex_files"][0]
                f.write(f"4. **Simplify Complex Files**: The most complex file `{path}` has a complexity score of {complexity}, which is {complexity / summary['avg_complexity']:.1f}x the average. Consider breaking it into smaller components.\n\n")
            
            # Template-based generation
            f.write(f"5. **Template-Based Generation**: Given the structural similarities, use template-based generation for all test files to ensure consistency.\n\n")
            
            # Conclusion
            f.write(f"## Conclusion\n\n")
            f.write(f"The test codebase shows clear patterns that can be leveraged for standardization. While there is some duplication, it appears to be structured and intentional based on the test generator approach. The main opportunity is to standardize the from_pretrained() testing methodology across all model types.\n\n")
            
            # Next steps
            f.write(f"## Next Steps\n\n")
            f.write(f"1. Create standardized templates for each model architecture type\n")
            f.write(f"2. Implement a unified approach to from_pretrained() testing\n")
            f.write(f"3. Reduce complexity in the most complex files\n")
            f.write(f"4. Update the test generator to ensure consistent patterns\n")
            f.write(f"5. Add comprehensive validation for all generated files\n\n")
            
            f.write(f"---\n\n")
            f.write(f"This report was generated using `generate_ast_report.py`.\n")
            f.write(f"To update this report, run:\n```bash\npython generate_ast_report.py\n```\n")


def main():
    parser = argparse.ArgumentParser(description='Generate AST analysis for Python test files')
    parser.add_argument('--dirs', nargs='+', default=TEST_DIRS,
                        help='Directories to analyze (default: {})'.format(', '.join(TEST_DIRS)))
    parser.add_argument('--output', default='ast_analysis_report.json',
                        help='Output JSON file (default: ast_analysis_report.json)')
    parser.add_argument('--markdown', default='ast_analysis_report.md',
                        help='Output Markdown file (default: ast_analysis_report.md)')
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='Patterns to exclude from analysis')
    
    args = parser.parse_args()
    
    analyzer = TestFileAnalyzer(args.dirs, args.exclude)
    analyzer.run_analysis(args.output, args.markdown)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())