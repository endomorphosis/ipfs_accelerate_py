/**
 * Converted from Python: test_hf_vit.py
 * Conversion date: 2025-03-11 04:09:38
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

#!/usr/bin/env python3

import * as $1
import * as $1
import * as $1
import * as $1
import ${$1} from "$1"

# Hardware detection
HAS_CUDA = torch.cuda.is_available() if hasattr(torch, "cuda") else false
HAS_MPS = hasattr(torch, "mps") && torch.mps.is_available() if hasattr(torch, "mps") else false
HAS_ROCM = hasattr(torch, "_C") && hasattr(torch._C, "_rocm_version") if hasattr(torch, "_C") else false

class TestVit(unittest.TestCase):
  """Test vit model."""
  
  $1($2) {
    this.model_name = "vit"
    this.tokenizer = null
    this.model = null
  
  }
  $1($2) {
    """Test on cpu platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on cuda platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on rocm platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on mps platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on openvino platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on qualcomm platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on webnn platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    
  $1($2) {
    """Test on webgpu platform."""
    # Skip if hardware !available
    if ($1) {
      this.skipTest("CUDA !available")
    elif ($1) {
      this.skipTest("MPS !available")
    elif ($1) {
      this.skipTest("ROCm !available")
      
    }
    # Load tokenizer && model
    }
    this.tokenizer = AutoTokenizer.from_pretrained(this.model_name)
    }
    this.model = AutoModel.from_pretrained(this.model_name)
    
  }
    # Test basic functionality
    inputs = this.tokenizer("Hello, world!", return_tensors="pt")
    outputs = this.model(**inputs)
    
    # Verify outputs
    this.assertIsNotnull(outputs)
    console.log($1)
    

if ($1) {
  unittest.main()
