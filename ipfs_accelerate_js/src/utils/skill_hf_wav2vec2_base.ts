/**
 * Converted from Python: skill_hf_wav2vec2_base.py
 * Conversion date: 2025-03-11 04:08:38
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

#!/usr/bin/env python3
"""
Skill implementation for wav2vec2-base with hardware platform support
"""

import * as $1
import * as $1
import * as $1
import * as $1 as np
import ${$1} from "$1"

class $1 extends $2 {
  """Skill for wav2vec2-base model with hardware platform support."""
  
}
  $1($2) {
    """Initialize the skill."""
    this.model_id = model_id
    this.device = device || this.get_default_device()
    this.tokenizer = null
    this.model = null
    
  }
  $1($2) {
    """Get the best available device."""
    # Check for CUDA
    if ($1) {
    return "cuda"
    }
    
  }
    # Check for MPS (Apple Silicon)
    if ($1) {
      if ($1) {
      return "mps"
      }
    
    }
    # Default to CPU
    return "cpu"
  
  $1($2) {
    """Load the model && tokenizer based on modality."""
    if ($1) {
      # Determine model modality
      modality = "audio"
      
    }
      # Load appropriate tokenizer/processor && model based on modality
      if ($1) {
        this.processor = AutoFeatureExtractor.from_pretrained(this.model_id)
        this.model = AutoModelForAudioClassification.from_pretrained(this.model_id)
      elif ($1) {
        this.processor = AutoImageProcessor.from_pretrained(this.model_id)
        this.model = AutoModelForImageClassification.from_pretrained(this.model_id)
      elif ($1) {
        this.processor = AutoProcessor.from_pretrained(this.model_id)
        this.model = AutoModel.from_pretrained(this.model_id)
      elif ($1) ${$1} else {
        # Default to text
        this.tokenizer = AutoTokenizer.from_pretrained(this.model_id)
        this.model = AutoModel.from_pretrained(this.model_id)
      
      }
      # Move to device
      }
      if ($1) {
        this.model = this.model.to(this.device)
  
      }
  $1($2) {
    """Process the input text && return the output."""
    # Ensure model is loaded
    this.load_model()
    
  }
    # Tokenize
      }
    inputs = this.tokenizer(text, return_tensors="pt")
      }
    
  }
    # Move to device
    if ($1) {
      inputs = {}k: v.to(this.device) for k, v in Object.entries($1)}
    
    }
    # Run inference
    with torch.no_grad():
      outputs = this.model(**inputs)
    
    # Convert to numpy for consistent output
      last_hidden_state = outputs.last_hidden_state.cpu().numpy()
    
    # Return formatted results
      return {}
      "model": this.model_id,
      "device": this.device,
      "last_hidden_state_shape": last_hidden_state.shape,
      "embedding": last_hidden_state.mean(axis=1).tolist(),
      }

# Factory function to create skill instance
$1($2) {
  """Create a skill instance."""
      return Wav2Vec2BaseSkill(model_id=model_id, device=device)