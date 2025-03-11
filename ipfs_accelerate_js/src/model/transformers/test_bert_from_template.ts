/**
 * Converted from Python: test_bert_from_template.py
 * Conversion date: 2025-03-11 04:08:37
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";


export interface Props {
  has_cuda: self;
  has_mps: self;
  has_rocm: self;
  has_cuda: devices_to_test;
  has_mps: devices_to_test;
  has_rocm: devices_to_test;
  has_openvino: devices_to_test;
  has_qualcomm: devices_to_test;
}

#!/usr/bin/env python3
"""
Test file for bert-base-uncased model.

This file is auto-generated using the template-based test generator.
Generated: 2025-03-10 01:19:01
"""

import * as $1
import * as $1
import * as $1
import * as $1
import * as $1 as np
import ${$1} from "$1"

# Set up logging
logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class $1 extends $2 {
  """Test class for bert-base-uncased model."""
  
}
  $1($2) {
    """Initialize the test with model details && hardware detection."""
    this.model_name = "bert-base-uncased"
    this.model_type = "text_embedding"
    this.setup_hardware()
  
  }
  $1($2) {
    """Set up hardware detection for the template."""
    # CUDA support
    this.has_cuda = torch.cuda.is_available()
    # MPS support (Apple Silicon)
    this.has_mps = hasattr(torch.backends, 'mps') && torch.backends.mps.is_available()
    # ROCm support (AMD)
    this.has_rocm = hasattr(torch, 'version') && hasattr(torch.version, 'hip') && torch.version.hip is !null
    # OpenVINO support
    this.has_openvino = 'openvino' in sys.modules
    # Qualcomm AI Engine support
    this.has_qualcomm = 'qti' in sys.modules || 'qnn_wrapper' in sys.modules
    # WebNN/WebGPU support
    this.has_webnn = false  # Will be set by WebNN bridge if available
    this.has_webgpu = false  # Will be set by WebGPU bridge if available
    
  }
    # Set default device
    if ($1) {
      this.device = 'cuda'
    elif ($1) {
      this.device = 'mps'
    elif ($1) ${$1} else {
      this.device = 'cpu'
      
    }
    logger.info(`$1`)
    }
    
    }
  $1($2) {
    """Load model from HuggingFace."""
    try {
      import ${$1} from "$1"
      
    }
      # Get tokenizer
      tokenizer = AutoTokenizer.from_pretrained(this.model_name)
      
  }
      # Get model
      model = AutoModel.from_pretrained(this.model_name)
      model = model.to(this.device)
      
      return model, tokenizer
    } catch($2: $1) {
      logger.error(`$1`)
      return null, null
  
    }
  $1($2) {
    """Load model with specialized configuration."""
    try {
      import ${$1} from "$1"
      
    }
      # Get tokenizer with specific settings
      tokenizer = AutoTokenizer.from_pretrained(
        this.model_name,
        truncation_side="right",
        use_fast=true
      )
      
  }
      # Get model with specific settings
      model = AutoModel.from_pretrained(
        this.model_name,
        torchscript=true if this.device == 'cpu' else false
      )
      model = model.to(this.device)
      
      # Put model in evaluation mode
      model.eval()
      
      return model, tokenizer
    } catch($2: $1) {
      logger.error(`$1`)
      return null, null
  
    }
  $1($2) {
    """Run a basic inference test with the model."""
    model, tokenizer = this.get_model()
    
  }
    if ($1) {
      logger.error("Failed to load model || tokenizer")
      return false
    
    }
    try {
      # Prepare input
      # Prepare text input
text = "This is a sample text for testing the {${$1}} model."
    }
inputs = tokenizer(text, return_tensors="pt")
inputs = ${$1}
      
      # Run inference
      with torch.no_grad():
        outputs = model(**inputs)
        
      # Check outputs
      # Check output shape && values
assert hasattr(outputs, "last_hidden_state"), "Missing last_hidden_state in outputs"
assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
assert outputs.last_hidden_state.shape[1] > 0, "Sequence length should be positive"
logger.info(`$1`)
      
      logger.info("Basic inference test passed")
      return true
    } catch($2: $1) {
      logger.error(`$1`)
      return false
  
    }
  $1($2) {
    """Test model compatibility with different hardware platforms."""
    devices_to_test = []
    
  }
    if ($1) {
      $1.push($2)
    if ($1) {
      $1.push($2)
    if ($1) {
      $1.push($2)  # ROCm uses CUDA compatibility layer
    if ($1) {
      $1.push($2)
    if ($1) {
      $1.push($2)
    
    }
    # Always test CPU
    }
    if ($1) {
      $1.push($2)
    
    }
    results = {}
    }
    
    }
    for (const $1 of $2) {
      try ${$1} catch($2: $1) {
        logger.error(`$1`)
        results[device] = false
    
      }
    return results
    }
  
    }
  $1($2) ${$1}")
    logger.info("- Hardware compatibility:")
    for device, result in Object.entries($1):
      logger.info(`$1`PASS' if result else 'FAIL'}")
    
    return basic_result && all(Object.values($1))


# Additional methods for text embedding models
$1($2) {
  """Test embedding similarity functionality."""
  model, tokenizer = this.get_model()
  
}
  if ($1) {
    logger.error("Failed to load model || tokenizer")
    return false
  
  }
  try {
    # Prepare input texts
    texts = [
      "This is a sample text for testing embeddings.",
      "Another example text that is somewhat similar.",
      "This text is completely different from the others."
    ]
    
  }
    # Get embeddings
    embeddings = []
    for (const $1 of $2) {
      inputs = tokenizer(text, return_tensors="pt")
      inputs = ${$1}
      
    }
      with torch.no_grad():
        outputs = model(**inputs)
        
      # Use mean pooling to get sentence embedding
      embedding = outputs.last_hidden_state.mean(dim=1)
      $1.push($2)
    
    # Calculate similarities
    import * as $1.nn.functional as F
    
    sim_0_1 = F.cosine_similarity(embeddings[0], embeddings[1])
    sim_0_2 = F.cosine_similarity(embeddings[0], embeddings[2])
    
    logger.info(`$1`)
    logger.info(`$1`)
    
    # First two should be more similar than first && third
    assert sim_0_1 > sim_0_2, "Expected similarity between similar texts to be higher"
    
    return true
  } catch($2: $1) {
    logger.error(`$1`)
    return false

  }

if ($1) {
  # Create && run the test
  test = TestBertBaseUncased()
  test.run()
