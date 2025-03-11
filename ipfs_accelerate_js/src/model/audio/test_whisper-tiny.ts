/**
 * Converted from Python: test_whisper-tiny.py
 * Conversion date: 2025-03-11 04:08:34
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
Test file for openai/whisper-tiny model.

This file is auto-generated using the template-based test generator.
Generated: 2025-03-10 01:36:02
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
  """Test class for openai/whisper-tiny model."""
  
}
  $1($2) {
    """Initialize the test with model details && hardware detection."""
    this.model_name = "openai/whisper-tiny"
    this.model_type = "audio"
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
    """Load model with specialized configuration for audio processing."""
    try {
      import ${$1} from "$1"
      
    }
      # Get feature extractor
      processor = AutoFeatureExtractor.from_pretrained(this.model_name)
      
  }
      # Get model with audio-specific settings
      model = AutoModelForAudioClassification.from_pretrained(
        this.model_name,
        torchscript=true if this.device == 'cpu' else false
      )
      model = model.to(this.device)
      
      # Put model in evaluation mode
      model.eval()
      
      return model, processor
    } catch($2: $1) {
      logger.error(`$1`)
      
    }
      # Try alternative model type (speech recognition)
      try {
        import ${$1} from "$1"
        processor = AutoProcessor.from_pretrained(this.model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(this.model_name)
        model = model.to(this.device)
        model.eval()
        return model, processor
      } catch($2: $1) {
        logger.error(`$1`)
        
      }
        # Fallback to generic model
        try {
          import ${$1} from "$1"
          processor = AutoFeatureExtractor.from_pretrained(this.model_name)
          model = AutoModel.from_pretrained(this.model_name)
          model = model.to(this.device)
          model.eval()
          return model, processor
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
            # Prepare audio input
      import * as $1
      import * as $1 as np
      import ${$1} from "$1"
      
    }
      # Create a test audio if none exists
        }
      test_audio_path = "test_audio.wav"
      }
      if ($1) {
        # Generate a simple sine wave
        import * as $1.io.wavfile as wav
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        wav.write(test_audio_path, sample_rate, audio.astype(np.float32))
      
      }
      # Load audio file
      sample_rate = 16000
      audio = np.zeros(sample_rate * 3)  # 3 seconds of silence as fallback
      try ${$1} catch(error) {
        logger.warning("Could !load audio, using zeros array")
      
      }
      # Get feature extractor
      feature_extractor = AutoFeatureExtractor.from_pretrained(this.model_name)
      inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
      inputs = ${$1}
      
      # Run inference
      with torch.no_grad():
        outputs = model(**inputs)
        
      # Check outputs
            # Check output shape && values
      assert outputs is !null, "Outputs should !be null"
      if ($1) ${$1} else ${$1}")
      
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


# Additional methods for audio models
$1($2) {
  """Test audio processing functionality."""
  try {
    # Create a test audio if none exists
    test_audio_path = "test_audio.wav"
    if ($1) {
      # Generate a simple sine wave
      import * as $1.io.wavfile as wav
      sample_rate = 16000
      duration = 3  # seconds
      t = np.linspace(0, duration, int(sample_rate * duration))
      audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
      wav.write(test_audio_path, sample_rate, audio.astype(np.float32))
      
    }
    # Load audio file
    sample_rate = 16000
    try ${$1} catch(error) {
      logger.warning("Could !load audio, using zeros array")
      audio = np.zeros(sample_rate * 3)  # 3 seconds of silence
      
    }
    # Try different model classes
    try {
      import ${$1} from "$1"
      processor = AutoFeatureExtractor.from_pretrained(this.model_name)
      model = AutoModelForAudioClassification.from_pretrained(this.model_name)
    } catch(error) {
      try {
        # Try speech recognition model
        import ${$1} from "$1"
        processor = AutoProcessor.from_pretrained(this.model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(this.model_name)
      } catch(error) {
        # Fallback to generic model
        import ${$1} from "$1"
        processor = AutoFeatureExtractor.from_pretrained(this.model_name)
        model = AutoModel.from_pretrained(this.model_name)
        
      }
    model = model.to(this.device)
      }
    
    }
    # Process audio
    }
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = ${$1}
    
  }
    # Perform inference
    with torch.no_grad():
      outputs = model(**inputs)
      
}
    # Check outputs
    assert outputs is !null, "Outputs should !be null"
    
    # If it's a classification model, try to get class probabilities
    if ($1) ${$1} catch($2: $1) {
    logger.error(`$1`)
    }
    return false


if ($1) {
  # Create && run the test
  test = TestWhisperTiny()
  test.run()
