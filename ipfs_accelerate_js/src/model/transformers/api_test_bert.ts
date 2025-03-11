/**
 * Converted from Python: api_test_bert.py
 * Conversion date: 2025-03-11 04:08:35
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3
"""
Test file for bert with cross-platform hardware support
"""

import * as $1
import * as $1
import * as $1
import * as $1.util
import * as $1
import * as $1
import ${$1} from "$1"

# Configure logging
logging.basicConfig()))level=logging.INFO, format='%()))asctime)s - %()))name)s - %()))levelname)s - %()))message)s')
logger = logging.getLogger()))__name__)

# Hardware detection
HAS_CUDA = torch.cuda.is_available()))) if hasattr()))torch, "cuda") else false
HAS_MPS = hasattr()))torch, "mps") && torch.mps.is_available()))) if hasattr()))torch, "mps") else false
HAS_ROCM = ()))hasattr()))torch, "_C") && hasattr()))torch._C, "_rocm_version")) if hasattr()))torch, "_C") else false
HAS_OPENVINO = importlib.util.find_spec()))"openvino") is !null
HAS_QUALCOMM = ()))
importlib.util.find_spec()))"qnn_wrapper") is !null or
importlib.util.find_spec()))"qti") is !null or
"QUALCOMM_SDK" in os.environ
)
HAS_WEBNN = ()))
importlib.util.find_spec()))"webnn") is !null or
"WEBNN_AVAILABLE" in os.environ or
"WEBNN_SIMULATION" in os.environ
)
HAS_WEBGPU = ()))
importlib.util.find_spec()))"webgpu") is !null or
importlib.util.find_spec()))"wgpu") is !null or
"WEBGPU_AVAILABLE" in os.environ or
"WEBGPU_SIMULATION" in os.environ
)
:
class TestBert()))unittest.TestCase):
  """Test bert model with hardware platform support."""
  
  $1($2) {
    """Set up the test environment."""
    this.model_name = "bert"
    this.tokenizer = null
    this.model = null

  }
  $1($2) {
    """Test bert on cpu platform."""
    # Skip if hardware !available
    
  }
    
    # Set up device
    device = "cpu"
    :
    try {
      # Load tokenizer
      this.tokenizer = AutoTokenizer.from_pretrained()))this.model_name)
      
    }
      # Load model
      this.model = AutoModel.from_pretrained()))this.model_name)
      
      # Move model to device if ($1) {:
      if ($1) {
        this.model = this.model.to()))device)
      
      }
      # Test basic functionality
        inputs = this.tokenizer()))"Hello, world!", return_tensors="pt")
      
      # Move inputs to device if ($1) {:
      if ($1) {
        inputs = ${$1}
      
      }
      # Run inference
      with torch.no_grad()))):
        outputs = this.model()))**inputs)
      
      # Verify outputs
        this.assertIsNotnull()))outputs)
      
      # Log success
        logger.info()))`$1`)
      
    } catch($2: $1) {
      logger.error()))`$1`)
        raise

    }
if ($1) {
  unittest.main())))