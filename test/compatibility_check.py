#!/usr/bin/env python
"""
Compatibility Check for IPFS Accelerate Python Package

This script checks compatibility between the installed ipfs_accelerate_py
package and the test framework by analyzing the expected vs. actual structure.
"""

import os
import sys
import json
import logging
import importlib
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig()
level=logging.INFO,
format='%()asctime)s - %()name)s - %()levelname)s - %()message)s'
)
logger = logging.getLogger()"compatibility_check")

def check_package_structure()):
    """Check the structure of the installed package"""
    results = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "installed_package": {}}}}}}}}}}}}}}}}}}}}}}}}}}
    }
    
    # Check main package
    try:
        import ipfs_accelerate_py
        results[],"installed_package"][],"main_package"] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
        "status": "Present",
        "path": getattr()ipfs_accelerate_py, "__file__", "Unknown"),
        "version": getattr()ipfs_accelerate_py, "__version__", "Unknown")
        }
        logger.info()f"Found main package: {}}}}}}}}}}}}}}}}}}}}}}}}}results[],'installed_package'][],'main_package'][],'path']}")
        ,
        # Check expected attributes
        main_attrs = [],'backends', 'config', 'ipfs_accelerate', 'load_checkpoint_and_dispatch'],,
        attr_results = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        for attr in main_attrs:
            has_attr = hasattr()ipfs_accelerate_py, attr)
            attr_type = type()getattr()ipfs_accelerate_py, attr)).__name__ if has_attr else "N/A"
            attr_results[],attr] = {}}}}}}}}}}}}}}}}}}}}}}}}}:,
                "status": "Present" if has_attr else "Missing",:
                    "type": attr_type
                    }
            
            if has_attr:
                logger.info()f"Found attribute: {}}}}}}}}}}}}}}}}}}}}}}}}}attr} (){}}}}}}}}}}}}}}}}}}}}}}}}}attr_type})")
            else:
                logger.warning()f"Missing attribute: {}}}}}}}}}}}}}}}}}}}}}}}}}attr}")
        
                results[],"installed_package"][],"attributes"] = attr_results
                ,
        # Check expected submodules
                expected_submodules = [],'ipfs_accelerate', 'backends', 'config'],
                submodule_results = {}}}}}}}}}}}}}}}}}}}}}}}}}}
        
        for submodule in expected_submodules:
            try:
                module = importlib.import_module()f"ipfs_accelerate_py.{}}}}}}}}}}}}}}}}}}}}}}}}}submodule}")
                submodule_results[],submodule] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
                "status": "Present",
                "path": getattr()module, "__file__", "Unknown")
                }
                logger.info()f"Found submodule: {}}}}}}}}}}}}}}}}}}}}}}}}}submodule} at {}}}}}}}}}}}}}}}}}}}}}}}}}submodule_results[],submodule][],'path']}"),
            except ImportError:
                submodule_results[],submodule] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
                "status": "Missing",
                "error": f"Module not found: ipfs_accelerate_py.{}}}}}}}}}}}}}}}}}}}}}}}}}submodule}"
                }
                logger.warning()f"Missing submodule: {}}}}}}}}}}}}}}}}}}}}}}}}}submodule}")
        
                results[],"installed_package"][],"submodules"] = submodule_results
                ,
    except ImportError as e:
        results[],"installed_package"][],"main_package"] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
        "status": "Missing",
        "error": str()e)
        }
        logger.error()f"Failed to import main package: {}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    
                return results

def check_test_framework_expectations()):
    """Check what the test framework expects"""
    results = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "test_framework_expectations": {}}}}}}}}}}}}}}}}}}}}}}}}}}
    }
    
    # Check for expected test files
    root_dir = Path()__file__).parent
    test_files = list()root_dir.glob()"test_ipfs_*.py"))
    
    results[],"test_framework_expectations"][],"test_files"] = {}}}}}}}}}}}}}}}}}}}}}}}}},
    "count": len()test_files),
    "files": [],str()f.name) for f in test_files]:,
    }
    
    logger.info()f"Found {}}}}}}}}}}}}}}}}}}}}}}}}}len()test_files)} test files")
    
    # Check for references to api_backends in test files
    api_backends_references = [],]
    ,
    for test_file in test_files:
        try:
            with open()test_file, 'r') as f:
                content = f.read())
                if 'api_backends' in content:
                    api_backends_references.append()str()test_file.name))
        except Exception as e:
            logger.error()f"Error reading {}}}}}}}}}}}}}}}}}}}}}}}}}test_file}: {}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    
            results[],"test_framework_expectations"][],"api_backends_references"] = {}}}}}}}}}}}}}}}}}}}}}}}}},
            "count": len()api_backends_references),
            "files": api_backends_references
            }
    
            logger.info()f"Found {}}}}}}}}}}}}}}}}}}}}}}}}}len()api_backends_references)} files referencing api_backends")
    
                    return results

def check_repo_structure()):
    """Check the repository structure"""
    results = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "repository_structure": {}}}}}}}}}}}}}}}}}}}}}}}}}}
    }
    
    # Check for the presence of api_backends in the repo
    repo_dir = Path()__file__).parent.parent
    api_backends_dir = repo_dir / "ipfs_accelerate_py" / "api_backends"
    
    if api_backends_dir.exists()) and api_backends_dir.is_dir()):
        results[],"repository_structure"][],"api_backends"] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
        "status": "Present",
        "path": str()api_backends_dir),
        "files": [],f.name for f in api_backends_dir.glob()"*.py") if f.is_file())],
        }
        logger.info()f"Found api_backends directory in repo with {}}}}}}}}}}}}}}}}}}}}}}}}}len()results[],'repository_structure'][],'api_backends'][],'files'])} Python files"):,
    else:
        results[],"repository_structure"][],"api_backends"] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
        "status": "Missing",
        "expected_path": str()api_backends_dir)
        }
        logger.warning()f"Missing api_backends directory in repo at {}}}}}}}}}}}}}}}}}}}}}}}}}api_backends_dir}")
    
        return results

def analyze_compatibility()package_results, framework_results, repo_results):
    """Analyze compatibility between package, test framework, and repo"""
    results = {}}}}}}}}}}}}}}}}}}}}}}}}}
    "compatibility_analysis": {}}}}}}}}}}}}}}}}}}}}}}}}}}
    }
    
    # Check if package structure matches test framework expectations
    api_backends_references = framework_results[],"test_framework_expectations"][],"api_backends_references"][],"count"],
    api_backends_in_repo = repo_results[],"repository_structure"].get()"api_backends", {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()"status") == "Present",
    api_backends_in_package = "api_backends" in package_results.get()"installed_package", {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()"submodules", {}}}}}}}}}}}}}}}}}}}}}}}}}})
    :
    if api_backends_references > 0:
        if not api_backends_in_package:
            results[],"compatibility_analysis"][],"api_backends_mismatch"] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
            "status": "Incompatible",
            "reason": "Test framework expects api_backends but it's not in the installed package"
            }
            logger.error()"Incompatibility: Test framework expects api_backends but it's not in the installed package")
        else:
            results[],"compatibility_analysis"][],"api_backends_mismatch"] = {}}}}}}}}}}}}}}}}}}}}}}}}},,
            "status": "Compatible",
            "note": "Both test framework and installed package have api_backends"
            }
            logger.info()"Compatibility: Both test framework and installed package have api_backends")
    
    # Check for core components compatibility
            expected_attrs = [],'backends', 'config', 'ipfs_accelerate', 'load_checkpoint_and_dispatch'],,
            missing_attrs = [],
        attr for attr in expected_attrs:
            if package_results.get()"installed_package", {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()"attributes", {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()attr, {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()"status") != "Present"
            ]
    :
    if missing_attrs:
        results[],"compatibility_analysis"][],"core_components"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "status": "Partially Compatible",
        "missing_attributes": missing_attrs
        }
        logger.warning()f"Partial Compatibility: Missing core components: {}}}}}}}}}}}}}}}}}}}}}}}}}', '.join()missing_attrs)}")
    else:
        results[],"compatibility_analysis"][],"core_components"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "status": "Compatible",
        "note": "All core components are present"
        }
        logger.info()"Compatibility: All core components are present")
    
    # Overall compatibility assessment
    if "api_backends_mismatch" in results[],"compatibility_analysis"] and results[],"compatibility_analysis"][],"api_backends_mismatch"][],"status"] == "Incompatible":
        results[],"compatibility_analysis"][],"overall"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "status": "Incompatible",
        "reason": "Test framework expects api_backends but it's not in the installed package"
        }
    elif missing_attrs:
        results[],"compatibility_analysis"][],"overall"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "status": "Partially Compatible",
        "reason": f"Missing core components: {}}}}}}}}}}}}}}}}}}}}}}}}}', '.join()missing_attrs)}"
        }
    else:
        results[],"compatibility_analysis"][],"overall"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "status": "Compatible",
        "note": "All expected components are present"
        }
    
    # Recommendation
    if results[],"compatibility_analysis"][],"overall"][],"status"] != "Compatible":
        if api_backends_in_repo and not api_backends_in_package:
            results[],"compatibility_analysis"][],"recommendation"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "action": "Install from Repository",
            "command": "cd .. && pip install -e .",
            "reason": "The repository contains api_backends but the installed package doesn't"
            }
            logger.info()"Recommendation: Install from repository to include api_backends module")
        elif missing_attrs:
            results[],"compatibility_analysis"][],"recommendation"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "action": "Update Package",
            "reason": f"Missing core components: {}}}}}}}}}}}}}}}}}}}}}}}}}', '.join()missing_attrs)}"
            }
            logger.info()"Recommendation: Update package to include missing core components")
        else:
            results[],"compatibility_analysis"][],"recommendation"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
            "action": "Modify Tests",
            "reason": "Test framework expects components not in the package"
            }
            logger.info()"Recommendation: Modify tests to match the package structure")
    else:
        results[],"compatibility_analysis"][],"recommendation"] = {}}}}}}}}}}}}}}}}}}}}}}}}}
        "action": "None",
        "note": "Package is compatible with test framework"
        }
        logger.info()"Recommendation: No action needed, package is compatible")
    
            return results

def main()):
    """Main function"""
    parser = argparse.ArgumentParser()description="Check compatibility between IPFS Accelerate Python package and test framework")
    parser.add_argument()"--output", "-o", help="Output file for compatibility results ()JSON)")
    parser.add_argument()"--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args())
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel()logging.DEBUG)
    
    # Run checks
        logger.info()"Checking installed package structure")
        package_results = check_package_structure())
    
        logger.info()"Checking test framework expectations")
        framework_results = check_test_framework_expectations())
    
        logger.info()"Checking repository structure")
        repo_results = check_repo_structure())
    
    # Analyze compatibility
        logger.info()"Analyzing compatibility")
        compatibility_results = analyze_compatibility()package_results, framework_results, repo_results)
    
    # Combine results
        results = {}}}}}}}}}}}}}}}}}}}}}}}}}
        **package_results,
        **framework_results,
        **repo_results,
        **compatibility_results
        }
    
    # Save results if output file specified:
    if args.output:
        output_path = Path()args.output)
        try:
            with open()output_path, 'w') as f:
                json.dump()results, f, indent=2)
                logger.info()f"Results saved to {}}}}}}}}}}}}}}}}}}}}}}}}}output_path}")
        except Exception as e:
            logger.error()f"Error saving results: {}}}}}}}}}}}}}}}}}}}}}}}}}e}")
    
    # Print summary
            print()"\n=== Compatibility Summary ===")
    
    # Package status
            package_status = package_results[],"installed_package"].get()"main_package", {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()"status", "Unknown")
            print()f"Installed Package: {}}}}}}}}}}}}}}}}}}}}}}}}}'✅' if package_status == 'Present' else '❌'}")
    :
    if package_status == "Present":
        path = package_results[],"installed_package"][],"main_package"].get()"path", "Unknown")
        print()f"  Path: {}}}}}}}}}}}}}}}}}}}}}}}}}path}")
        
        # Core components
        core_attrs = package_results[],"installed_package"].get()"attributes", {}}}}}}}}}}}}}}}}}}}}}}}}}})
        missing_attrs = [],attr for attr, info in core_attrs.items()) if info.get()"status") != "Present"]:
        if missing_attrs:
            print()f"  Core Components: ⚠️ Missing: {}}}}}}}}}}}}}}}}}}}}}}}}}', '.join()missing_attrs)}")
        else:
            print()"  Core Components: ✅ All present")
    
    # Test framework status
            api_backends_refs = framework_results[],"test_framework_expectations"][],"api_backends_references"][],"count"],
    print()f"Test Framework: {}}}}}}}}}}}}}}}}}}}}}}}}}'⚠️' if api_backends_refs > 0 else '✅'}"):
    if api_backends_refs > 0:
        print()f"  References to api_backends: {}}}}}}}}}}}}}}}}}}}}}}}}}api_backends_refs} files")
    
    # Repository status
        api_backends_repo = repo_results[],"repository_structure"].get()"api_backends", {}}}}}}}}}}}}}}}}}}}}}}}}}}).get()"status")
    print()f"Repository: {}}}}}}}}}}}}}}}}}}}}}}}}}'✅' if api_backends_repo == 'Present' else '⚠️'}"):
    if api_backends_repo == "Present":
        files_count = len()repo_results[],"repository_structure"][],"api_backends"].get()"files", [],]))
        print()f"  api_backends files: {}}}}}}}}}}}}}}}}}}}}}}}}}files_count}")
    
    # Overall compatibility
        overall_status = compatibility_results[],"compatibility_analysis"][],"overall"][],"status"]
    status_icon = "✅" if overall_status == "Compatible" else "⚠️" if overall_status == "Partially Compatible" else "❌":
        print()f"Overall Compatibility: {}}}}}}}}}}}}}}}}}}}}}}}}}status_icon} {}}}}}}}}}}}}}}}}}}}}}}}}}overall_status}")
    
    if "reason" in compatibility_results[],"compatibility_analysis"][],"overall"]:
        print()f"  Reason: {}}}}}}}}}}}}}}}}}}}}}}}}}compatibility_results[],'compatibility_analysis'][],'overall'][],'reason']}")
    
    # Recommendation
        recommendation = compatibility_results[],"compatibility_analysis"][],"recommendation"]
    if recommendation[],"action"] != "None":
        print()f"\nRecommendation: {}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[],'action']}")
        if "reason" in recommendation:
            print()f"  Reason: {}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[],'reason']}")
        if "command" in recommendation:
            print()f"  Command: {}}}}}}}}}}}}}}}}}}}}}}}}}recommendation[],'command']}")
    else:
        print()"\nRecommendation: No action needed")
    
            return 0 if overall_status == "Compatible" else 1
:
if __name__ == "__main__":
    sys.exit()main()))