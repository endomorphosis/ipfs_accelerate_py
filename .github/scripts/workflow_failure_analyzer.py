#!/usr/bin/env python3
"""
Advanced Workflow Failure Analyzer

This script provides advanced analysis capabilities for GitHub Actions workflow failures.
It can identify common patterns, suggest fixes, and generate detailed reports.
"""

import json
import os
import re
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class FailureCategory(Enum):
    """Categories of workflow failures."""
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    PERMISSION = "permission"
    SYNTAX = "syntax"
    NETWORK = "network"
    TEST = "test"
    BUILD = "build"
    DOCKER = "docker"
    UNKNOWN = "unknown"


@dataclass
class FailurePattern:
    """Pattern for identifying and categorizing failures."""
    category: FailureCategory
    pattern: str
    description: str
    suggested_fix: str
    confidence: float


@dataclass
class AnalysisResult:
    """Result of failure analysis."""
    category: FailureCategory
    confidence: float
    description: str
    suggested_fixes: List[str]
    matched_patterns: List[str]
    root_cause: str
    affected_files: List[str]


class WorkflowFailureAnalyzer:
    """Analyzes GitHub Actions workflow failures and suggests fixes."""
    
    def __init__(self):
        """Initialize the analyzer with failure patterns."""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[FailurePattern]:
        """Initialize failure detection patterns."""
        return [
            # Dependency Issues
            FailurePattern(
                FailureCategory.DEPENDENCY,
                r"ModuleNotFoundError:|No module named|ImportError:|cannot import",
                "Missing Python module dependency",
                "Add the missing module to requirements.txt or install it in the workflow",
                0.95
            ),
            FailurePattern(
                FailureCategory.DEPENDENCY,
                r"Error: Cannot find module|MODULE_NOT_FOUND",
                "Missing Node.js module dependency",
                "Add the missing module to package.json or run npm install",
                0.95
            ),
            FailurePattern(
                FailureCategory.DEPENDENCY,
                r"package .* is not available|there is no package called",
                "Missing R package dependency",
                "Install the missing R package in the workflow",
                0.90
            ),
            
            # Timeout Issues
            FailurePattern(
                FailureCategory.TIMEOUT,
                r"timeout|timed out|exceeded the maximum execution time",
                "Operation or job timeout",
                "Increase the timeout value or optimize the operation",
                0.90
            ),
            
            # Resource Issues
            FailurePattern(
                FailureCategory.RESOURCE,
                r"no space left on device|disk quota exceeded",
                "Insufficient disk space",
                "Add disk cleanup steps or use a runner with more space",
                0.95
            ),
            FailurePattern(
                FailureCategory.RESOURCE,
                r"out of memory|OOM|killed.*memory",
                "Out of memory error",
                "Reduce memory usage or use a runner with more memory",
                0.90
            ),
            
            # Permission Issues
            FailurePattern(
                FailureCategory.PERMISSION,
                r"permission denied|access denied|forbidden|403",
                "Permission or access error",
                "Check file permissions, GitHub token permissions, or use sudo",
                0.85
            ),
            
            # Syntax Issues
            FailurePattern(
                FailureCategory.SYNTAX,
                r"SyntaxError:|invalid syntax|unexpected token",
                "Syntax error in code",
                "Fix the syntax error in the indicated file",
                0.95
            ),
            FailurePattern(
                FailureCategory.SYNTAX,
                r"yaml.*error|invalid yaml|mapping values are not allowed",
                "YAML syntax error",
                "Fix the YAML syntax in the workflow file",
                0.90
            ),
            
            # Network Issues
            FailurePattern(
                FailureCategory.NETWORK,
                r"connection refused|connection timeout|network.*error|failed to connect",
                "Network connectivity issue",
                "Check service availability, add retries, or increase timeout",
                0.80
            ),
            FailurePattern(
                FailureCategory.NETWORK,
                r"unable to resolve host|DNS.*fail|name resolution fail",
                "DNS resolution failure",
                "Check network connectivity or use alternative DNS",
                0.85
            ),
            
            # Test Failures
            FailurePattern(
                FailureCategory.TEST,
                r"test.*failed|assertion.*error|expected.*but got",
                "Test assertion failure",
                "Fix the failing test or update the code to pass the test",
                0.80
            ),
            
            # Build Issues
            FailurePattern(
                FailureCategory.BUILD,
                r"build failed|compilation error|compile.*error",
                "Build or compilation failure",
                "Fix the build errors in the source code",
                0.85
            ),
            
            # Docker Issues
            FailurePattern(
                FailureCategory.DOCKER,
                r"docker.*build.*failed|failed to solve|ERROR \[.*\]",
                "Docker build failure",
                "Review Dockerfile syntax and build context",
                0.85
            ),
            FailurePattern(
                FailureCategory.DOCKER,
                r"docker.*pull.*failed|manifest.*not found",
                "Docker image pull failure",
                "Check image name/tag or network connectivity",
                0.90
            ),
        ]
    
    def analyze_logs(self, log_content: str) -> AnalysisResult:
        """
        Analyze failure logs and categorize the failure.
        
        Args:
            log_content: The raw log content from the failed workflow
            
        Returns:
            AnalysisResult with categorized failure information
        """
        matched_patterns = []
        confidence_scores = []
        categories = []
        
        # Check each pattern against the logs
        for pattern in self.patterns:
            if re.search(pattern.pattern, log_content, re.IGNORECASE):
                matched_patterns.append(pattern)
                confidence_scores.append(pattern.confidence)
                categories.append(pattern.category)
        
        # Determine primary category
        if not matched_patterns:
            category = FailureCategory.UNKNOWN
            confidence = 0.5
            description = "Unable to categorize failure automatically"
            suggested_fixes = [
                "Review the failure logs manually",
                "Check for recent changes that might have caused the failure",
                "Compare with previous successful runs"
            ]
            root_cause = "Unknown - manual investigation required"
        else:
            # Use the pattern with highest confidence
            primary_pattern = max(matched_patterns, key=lambda p: p.confidence)
            category = primary_pattern.category
            confidence = primary_pattern.confidence
            description = primary_pattern.description
            suggested_fixes = [p.suggested_fix for p in matched_patterns]
            root_cause = self._extract_root_cause(log_content, matched_patterns)
        
        # Extract affected files
        affected_files = self._extract_file_references(log_content)
        
        return AnalysisResult(
            category=category,
            confidence=confidence,
            description=description,
            suggested_fixes=suggested_fixes,
            matched_patterns=[p.pattern for p in matched_patterns],
            root_cause=root_cause,
            affected_files=affected_files
        )
    
    def _extract_root_cause(self, log_content: str, patterns: List[FailurePattern]) -> str:
        """Extract the root cause from logs based on matched patterns."""
        # Look for error messages near matched patterns
        lines = log_content.split('\n')
        error_lines = []
        
        for i, line in enumerate(lines):
            if any(re.search(p.pattern, line, re.IGNORECASE) for p in patterns):
                # Include context around the error
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                error_lines.extend(lines[start:end])
        
        if error_lines:
            return '\n'.join(error_lines[:10])  # Limit to 10 lines
        
        return "Unable to extract specific root cause"
    
    def _extract_file_references(self, log_content: str) -> List[str]:
        """Extract file paths mentioned in error logs."""
        # Common patterns for file references
        file_patterns = [
            r'File "([^"]+)"',
            r'in ([^\s]+\.py):',
            r'([^\s]+\.yml).*error',
            r'([^\s]+\.yaml).*error',
            r'Error in ([^\s]+):',
        ]
        
        files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, log_content)
            files.update(matches)
        
        # Filter out system files and keep only project files
        project_files = [
            f for f in files 
            if not f.startswith(('/usr/', '/home/runner/.', '/opt/'))
            and not f.startswith('/')
        ]
        
        return list(project_files)[:10]  # Limit to 10 files
    
    def generate_copilot_prompt(self, analysis: AnalysisResult, 
                                workflow_context: Dict[str, Any]) -> str:
        """
        Generate a detailed prompt for GitHub Copilot to fix the issue.
        
        Args:
            analysis: The analysis result
            workflow_context: Context about the workflow
            
        Returns:
            Formatted prompt for Copilot
        """
        prompt = f"""# Workflow Failure Fix Request

## Failure Classification

- **Category**: {analysis.category.value}
- **Confidence**: {analysis.confidence * 100:.1f}%
- **Description**: {analysis.description}

## Root Cause

```
{analysis.root_cause}
```

## Affected Files

{chr(10).join(f'- `{f}`' for f in analysis.affected_files) if analysis.affected_files else 'No specific files identified'}

## Suggested Fixes

{chr(10).join(f'{i+1}. {fix}' for i, fix in enumerate(analysis.suggested_fixes))}

## Workflow Context

- **Workflow**: {workflow_context.get('workflow_name', 'Unknown')}
- **Branch**: {workflow_context.get('branch', 'Unknown')}
- **Commit**: {workflow_context.get('commit_sha', 'Unknown')[:7]}
- **Run ID**: {workflow_context.get('run_id', 'Unknown')}

## Required Actions

Please:

1. Analyze the root cause in detail
2. Review the affected files
3. Implement the minimal necessary fixes
4. Ensure the fix doesn't break other functionality
5. Add appropriate tests if needed
6. Update documentation if behavior changes

## Implementation Guidelines

- Make minimal, surgical changes
- Follow existing code style and patterns
- Add comments for non-obvious changes
- Test the fix thoroughly
- Consider edge cases and side effects

## Success Criteria

- [ ] Original failure is resolved
- [ ] All existing tests still pass
- [ ] No new errors or warnings introduced
- [ ] Code follows project conventions
- [ ] Changes are well-documented
"""
        return prompt
    
    def generate_report(self, analysis: AnalysisResult, 
                       workflow_context: Dict[str, Any]) -> str:
        """
        Generate a human-readable analysis report.
        
        Args:
            analysis: The analysis result
            workflow_context: Context about the workflow
            
        Returns:
            Formatted markdown report
        """
        confidence_emoji = "🟢" if analysis.confidence > 0.8 else "🟡" if analysis.confidence > 0.6 else "🔴"
        
        report = f"""# Workflow Failure Analysis Report

## Overview

{confidence_emoji} **Confidence Level**: {analysis.confidence * 100:.1f}%

- **Category**: {analysis.category.value.upper()}
- **Description**: {analysis.description}
- **Workflow**: {workflow_context.get('workflow_name', 'Unknown')}
- **Run ID**: {workflow_context.get('run_id', 'Unknown')}

## Root Cause Analysis

```
{analysis.root_cause}
```

## Affected Components

### Files
{chr(10).join(f'- `{f}`' for f in analysis.affected_files) if analysis.affected_files else '_No specific files identified_'}

### Matched Error Patterns
{chr(10).join(f'- {p}' for p in analysis.matched_patterns[:5]) if analysis.matched_patterns else '_No patterns matched_'}

## Recommended Actions

### Immediate Fixes
{chr(10).join(f'{i+1}. {fix}' for i, fix in enumerate(analysis.suggested_fixes))}

### Prevention
{self._get_prevention_advice(analysis.category)}

## Next Steps

1. **Review**: Examine the root cause and affected files
2. **Fix**: Implement the recommended fixes
3. **Test**: Verify the fix resolves the issue
4. **Deploy**: Merge the fix once validated
5. **Monitor**: Watch for recurrence

## Additional Resources

{self._get_resources(analysis.category)}

---
*Generated by Advanced Workflow Failure Analyzer*
*Analysis performed at: {workflow_context.get('timestamp', 'Unknown')}*
"""
        return report
    
    def _get_prevention_advice(self, category: FailureCategory) -> str:
        """Get prevention advice for a failure category."""
        advice = {
            FailureCategory.DEPENDENCY: """
- Use dependency lock files (requirements.txt, package-lock.json)
- Pin dependency versions
- Regularly update dependencies
- Use dependabot for automated updates""",
            
            FailureCategory.TIMEOUT: """
- Set appropriate timeout values
- Optimize long-running operations
- Use caching where possible
- Consider splitting large jobs""",
            
            FailureCategory.RESOURCE: """
- Monitor resource usage
- Clean up temporary files
- Use appropriate runner sizes
- Optimize resource-intensive operations""",
            
            FailureCategory.PERMISSION: """
- Document required permissions
- Use least-privilege principle
- Test with appropriate tokens
- Review security settings""",
            
            FailureCategory.SYNTAX: """
- Use linters and formatters
- Enable pre-commit hooks
- Use IDE syntax checking
- Add syntax validation to CI""",
            
            FailureCategory.NETWORK: """
- Add retry logic
- Use timeout values
- Check service dependencies
- Monitor external services""",
            
            FailureCategory.TEST: """
- Write robust tests
- Use appropriate assertions
- Maintain test data
- Review test coverage""",
            
            FailureCategory.BUILD: """
- Test builds locally
- Keep build tools updated
- Document build requirements
- Use consistent environments""",
            
            FailureCategory.DOCKER: """
- Test Dockerfiles locally
- Use multi-stage builds
- Pin base image versions
- Review build context""",
        }
        return advice.get(category, "No specific prevention advice available")
    
    def _get_resources(self, category: FailureCategory) -> str:
        """Get relevant documentation resources."""
        resources = {
            FailureCategory.DEPENDENCY: """
- [Managing dependencies](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Python dependencies](https://packaging.python.org/en/latest/tutorials/installing-packages/)""",
            
            FailureCategory.TIMEOUT: """
- [Timeout configuration](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idtimeout-minutes)
- [Performance optimization](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)""",
            
            FailureCategory.DOCKER: """
- [Docker build actions](https://github.com/docker/build-push-action)
- [Dockerfile best practices](https://docs.docker.com/develop/dev-best-practices/)""",
        }
        return resources.get(category, "- [GitHub Actions Documentation](https://docs.github.com/en/actions)")


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python workflow_failure_analyzer.py <failure_analysis.json>", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load failure analysis
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
            sys.exit(1)
        
        with open(input_file, 'r') as f:
            failure_data = json.load(f)
        
        # Validate required fields
        required_fields = ['workflow_name', 'failure_details']
        missing_fields = [field for field in required_fields if field not in failure_data]
        if missing_fields:
            print(f"Warning: Missing required fields: {', '.join(missing_fields)}", file=sys.stderr)
        
        # Extract logs
        logs = []
        for detail in failure_data.get('failure_details', []):
            logs.append(detail.get('error_logs', ''))
        log_content = '\n'.join(logs)
        
        if not log_content.strip():
            print("Warning: No failure logs found in the analysis", file=sys.stderr)
        
        # Analyze
        analyzer = WorkflowFailureAnalyzer()
        analysis = analyzer.analyze_logs(log_content)
        
        # Generate report
        workflow_context = {
            'workflow_name': failure_data.get('workflow_name', 'Unknown'),
            'branch': failure_data.get('branch', 'Unknown'),
            'commit_sha': failure_data.get('commit_sha', 'Unknown'),
            'run_id': failure_data.get('run_id', 'Unknown'),
            'timestamp': failure_data.get('failed_at', 'Unknown'),
        }
        
        report = analyzer.generate_report(analysis, workflow_context)
        print(report)
        
        # Save detailed analysis
        with open('detailed_analysis.json', 'w') as f:
            json.dump(asdict(analysis), f, indent=2, default=str)
        
        # Generate Copilot prompt
        copilot_prompt = analyzer.generate_copilot_prompt(analysis, workflow_context)
        with open('copilot_detailed_prompt.md', 'w') as f:
            f.write(copilot_prompt)
        
        print("\n✅ Analysis complete!")
        print(f"- Detailed analysis: detailed_analysis.json")
        print(f"- Copilot prompt: copilot_detailed_prompt.md")
        
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
