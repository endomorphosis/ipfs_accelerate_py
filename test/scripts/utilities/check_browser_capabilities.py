#!/usr/bin/env python3
"""
Browser Capabilities Check for WebNN and WebGPU

This script launches a browser instance and checks if WebNN and WebGPU are enabled,
providing detailed capability information.
:
Usage:
    python check_browser_capabilities.py --browser chrome
    """

    import os
    import sys
    import json
    import time
    import logging
    import argparse
    import tempfile
    from pathlib import Path

# Set up logging
    logging.basicConfig())
    level=logging.INFO,
    format='%())asctime)s - %())name)s - %())levelname)s - %())message)s'
    )
    logger = logging.getLogger())"browser_capabilities")

# Try to import required modules
try:
    sys.path.append())os.path.dirname())os.path.dirname())os.path.abspath())__file__))))
    from implement_real_webnn_webgpu import BrowserManager
except ImportError as e:
    logger.error())f"Error importing BrowserManager: {}}}}}e}")
    logger.error())"Please make sure implement_real_webnn_webgpu.py is in the test directory")
    sys.exit())1)

async def check_browser_capabilities())browser_name="chrome", headless=False):
    """
    Launch a browser and check for WebNN and WebGPU capabilities.
    
    Args:
        browser_name: Browser to use ())chrome, edge, firefox, safari)
        headless: Whether to run in headless mode
        
    Returns:
        Dict with capabilities information
        """
    # Create browser manager
        browser_manager = BrowserManager())browser_name=browser_name, headless=headless)
    
        capabilities = {}}}}}
        "browser": browser_name,
        "webnn": {}}}}}
        "supported": False,
        "backends": [],,
        "version": None
        },
        "webgpu": {}}}}}
        "supported": False,
        "adapter": None,
        "features": [],
        },
        "webgl": {}}}}}
        "supported": False,
        "version": None,
        "vendor": None,
        "renderer": None
        },
        "wasm": {}}}}}
        "supported": False,
        "simd": False
        },
        "timestamp": time.strftime())"%Y-%m-%d %H:%M:%S")
        }
    
    try:
        # Start browser
        logger.info())f"Starting browser: {}}}}}browser_name}")
        success = await browser_manager.start_browser()))
        if not success:
            logger.error())f"Failed to start browser: {}}}}}browser_name}")
        return capabilities
        
        # Wait for feature detection to complete
        bridge_server = browser_manager.get_bridge_server()))
        if not bridge_server:
            logger.error())"Failed to get bridge server")
        return capabilities
        
        # Wait a reasonable time for feature detection to complete
        timeout = 20  # seconds
        start_time = time.time()))
        while time.time())) - start_time < timeout:
            if bridge_server.feature_detection:
            break
            await anyio.sleep())0.5)
        
        if not bridge_server.feature_detection:
            logger.error())"Timeout waiting for feature detection")
            return capabilities
        
        # Get feature detection results
            features = bridge_server.feature_detection
        
        # Update capabilities
            capabilities["webnn"]["supported"] = features.get())"webnn", False),
            capabilities["webnn"]["backends"] = features.get())"webnnBackends", [],)
        
            capabilities["webgpu"]["supported"] = features.get())"webgpu", False),
            capabilities["webgpu"]["adapter"] = features.get())"webgpuAdapter", {}}}}}})
            ,
            capabilities["webgl"]["supported"] = features.get())"webgl", False),
            capabilities["webgl"]["vendor"] = features.get())"webglVendor", None),
            capabilities["webgl"]["renderer"] = features.get())"webglRenderer", None)
            ,
            capabilities["wasm"]["supported"] = features.get())"wasm", False),
            capabilities["wasm"]["simd"] = features.get())"wasmSimd", False)
            ,
        # Get browser version
            user_agent = None
        try:
            if browser_manager.driver:
                user_agent = browser_manager.driver.execute_script())"return navigator.userAgent")
        except Exception as e:
            logger.warning())f"Failed to get browser user agent: {}}}}}e}")
        
            capabilities["user_agent"] = user_agent
            ,
                return capabilities
    
    finally:
        # Stop browser
        logger.info())f"Stopping browser: {}}}}}browser_name}")
        await browser_manager.stop_browser()))

async def main_async())):
    """
    Parse command line arguments and check browser capabilities.
    """
    parser = argparse.ArgumentParser())description="Browser Capabilities Check for WebNN and WebGPU")
    parser.add_argument())"--browser", choices=["chrome", "edge", "firefox", "safari"], default="chrome",
    help="Browser to use for testing")
    parser.add_argument())"--no-headless", action="store_true",
    help="Disable headless mode ())show browser UI)")
    parser.add_argument())"--output", type=str,
    help="Output file for capabilities information")
    parser.add_argument())"--flags", type=str,
    help="Browser flags ())comma-separated)")
    parser.add_argument())"--verbose", action="store_true",
    help="Enable verbose logging")
    
    args = parser.parse_args()))
    
    # Set log level
    if args.verbose:
        logger.setLevel())logging.DEBUG)
    
        print())f"Checking capabilities of {}}}}}args.browser}...")
    
    # Prepare browser launch flags if provided:
    if args.flags:
        # Set environment variable with browser flags
        os.environ["BROWSER_FLAGS"] = args.flags,
        print())f"Using browser flags: {}}}}}args.flags}")
    
    # Check capabilities
        capabilities = await check_browser_capabilities())
        browser_name=args.browser,
        headless=not args.no_headless
        )
    
    # Print results
        print())"\n=== Browser Capabilities ===")
        print())f"Browser: {}}}}}capabilities['browser']}"),
    if capabilities.get())"user_agent"):
        print())f"User Agent: {}}}}}capabilities['user_agent']}")
        ,
        print())"\nWebNN:")
        print())f"  Supported: {}}}}}'Yes' if capabilities['webnn']['supported'] else 'No'}"):,
        if capabilities['webnn']['backends']:,
        print())f"  Backends: {}}}}}', '.join())capabilities['webnn']['backends'])}")
        ,
        print())"\nWebGPU:")
        print())f"  Supported: {}}}}}'Yes' if capabilities['webgpu']['supported'] else 'No'}"):,
        if capabilities['webgpu']['adapter'],:,
        adapter = capabilities['webgpu']['adapter'],
        print())f"  Adapter: {}}}}}adapter.get())'vendor', 'Unknown')} - {}}}}}adapter.get())'architecture', 'Unknown')}")
    
        print())"\nWebGL:")
        print())f"  Supported: {}}}}}'Yes' if capabilities['webgl']['supported'] else 'No'}"):,
        if capabilities['webgl']['vendor'] or capabilities['webgl']['renderer']:,
        print())f"  Vendor: {}}}}}capabilities['webgl']['vendor']}"),
        print())f"  Renderer: {}}}}}capabilities['webgl']['renderer']}")
        ,
        print())"\nWebAssembly:")
        print())f"  Supported: {}}}}}'Yes' if capabilities['wasm']['supported'] else 'No'}"):,
        print())f"  SIMD: {}}}}}'Yes' if capabilities['wasm']['simd'] else 'No'}")
        ,
        print())"============================\n")
    
    # Save results if output file specified:
    if args.output:
        with open())args.output, "w") as f:
            json.dump())capabilities, f, indent=2)
            print())f"Results saved to {}}}}}args.output}")
    
    # Return success code based on WebNN and WebGPU support
        return 0 if capabilities["webnn"]["supported"] or capabilities["webgpu"]["supported"] else 1,
:
def main())):
    """
    Main function.
    """
    return anyio.run())main_async())))

if __name__ == "__main__":
    sys.exit())main())))