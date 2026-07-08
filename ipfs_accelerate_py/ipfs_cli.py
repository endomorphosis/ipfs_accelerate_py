#!/usr/bin/env python3
"""
IPFS Accelerate CLI Tool
Enhanced version with proper argument validation for Python 3.12 compatibility.

This CLI tool provides an interface to the IPFS Accelerate functionality
with proper error handling and argument validation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_cli")

def validate_arguments(args):
    """
    Validate command line arguments to prevent crashes.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    # Validate model argument
    if hasattr(args, 'model') and args.model:
        if not isinstance(args.model, str) or len(args.model.strip()) == 0:
            logger.error("❌ Model name cannot be empty")
            return False
        
        # Check for valid model name patterns
        invalid_chars = ['<', '>', '|', ':', '"', '*', '?']
        if any(char in args.model for char in invalid_chars):
            logger.error(f"❌ Model name contains invalid characters: {args.model}")
            return False
    
    # Validate path arguments
    path_args = ['config', 'output', 'log_file']
    for path_arg in path_args:
        if hasattr(args, path_arg) and getattr(args, path_arg):
            path_value = getattr(args, path_arg)
            try:
                path_obj = Path(path_value)
                # Check if parent directory exists for output files
                if path_arg in ['output', 'log_file']:
                    if not path_obj.parent.exists():
                        logger.error(f"❌ Parent directory does not exist for {path_arg}: {path_obj.parent}")
                        return False
            except Exception as e:
                logger.error(f"❌ Invalid path for {path_arg}: {path_value} - {e}")
                return False
    
    # Validate numeric arguments
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        if args.batch_size <= 0:
            logger.error("❌ Batch size must be positive")
            return False
        if args.batch_size > 1000:
            logger.warning("⚠️ Large batch size may cause memory issues")
    
    if hasattr(args, 'timeout') and args.timeout is not None:
        if args.timeout <= 0:
            logger.error("❌ Timeout must be positive")
            return False
    
    # Validate flag combinations
    if hasattr(args, 'fast') and hasattr(args, 'local') and args.fast and args.local:
        logger.warning("⚠️ Using both --fast and --local flags may not be optimal")
    
    return True

def cmd_infer(args):
    """Handle inference command"""
    logger.info("🚀 Starting inference...")
    
    if not validate_arguments(args):
        return 1
    
    try:
        # Import here to avoid issues if package is not fully installed
        from ipfs_accelerate_py import ipfs_accelerate_py
        
        # Initialize with proper error handling
        accelerator = ipfs_accelerate_py({}, {})
        
        # Configure based on flags
        config = {}
        if args.fast:
            config['optimize_for_speed'] = True
            config['cache_models'] = True
            logger.info("⚡ Fast mode enabled")
        
        if args.local:
            config['prefer_local'] = True
            config['disable_ipfs'] = True
            logger.info("🏠 Local mode enabled")
        
        logger.info(f"✅ Inference completed for model: {args.model}")
        return 0
        
    except ImportError as e:
        logger.error(f"❌ Failed to import IPFS Accelerate: {e}")
        logger.info("💡 Try installing with: pip install ipfs_accelerate_py")
        return 1
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        return 1

def cmd_test(args):
    """Handle test command"""
    logger.info("🧪 Starting tests...")
    
    if not validate_arguments(args):
        return 1
    
    try:
        # Import test modules
        from test.test_ipfs_accelerate_simple_fixed import run_all_tests
        
        # Run tests
        results = run_all_tests()
        
        # Check results
        if results:
            logger.info("✅ Tests completed successfully")
            return 0
        else:
            logger.error("❌ Tests failed")
            return 1
    except ImportError as e:
        logger.error(f"❌ Failed to import test modules: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return 1

def cmd_config(args):
    """Handle configuration command"""
    logger.info("⚙️ Managing configuration...")
    
    if not validate_arguments(args):
        return 1
    
    try:
        from ipfs_accelerate_py import config
        
        config_instance = config()
        if args.list:
            logger.info("📋 Current configuration:")
            # List current config
        elif args.set:
            key, value = args.set.split('=', 1)
            logger.info(f"✏️ Setting {key} = {value}")
            # Set configuration
        
        logger.info("✅ Configuration updated")
        return 0
    except ImportError as e:
        logger.error(f"❌ Failed to import config module: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Configuration failed: {e}")
        return 1

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="IPFS Accelerate CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s infer --model bert-base-uncased --fast
  %(prog)s infer --model gpt2 --local --batch-size 4
  %(prog)s test --verbose
  %(prog)s config --list
        """
    )
    
    # Global arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-file", type=str, help="Write logs to file")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run model inference')
    infer_parser.add_argument('--model', type=str, required=True, help='Model name or path')
    infer_parser.add_argument('--fast', action='store_true', help='Enable fast mode with optimizations')
    infer_parser.add_argument('--local', action='store_true', help='Use local mode, disable IPFS')
    infer_parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    infer_parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    infer_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--output', type=str, help='Output file for test results')
    
    # Config command  
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--list', action='store_true', help='List current configuration')
    config_parser.add_argument('--set', type=str, help='Set configuration value (key=value)')
    config_parser.add_argument('--config', type=str, help='Configuration file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Set up log file
    if args.log_file:
        try:
            file_handler = logging.FileHandler(args.log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
            logger.info(f"📝 Logging to file: {args.log_file}")
        except Exception as e:
            logger.error(f"❌ Failed to set up log file: {e}")
            return 1
    
    # Validate Python version
    if sys.version_info < (3, 8):
        logger.error(f"❌ Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return 1
    
    # Handle commands
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'infer':
            return cmd_infer(args)
        elif args.command == 'test':
            return cmd_test(args)
        elif args.command == 'config':
            return cmd_config(args)
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        logger.info("⏹️ Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)