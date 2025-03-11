/**
 * Converted from Python: test_llava_models.py
 * Conversion date: 2025-03-11 04:08:45
 * This file was automatically converted from Python to TypeScript.
 * Conversion fidelity might not be 100%, please manual review recommended.
 */

// WebGPU related imports
import { HardwareBackend } from "../hardware_abstraction";

"""

# Import hardware detection capabilities if ($1) {
try ${$1} catch($2: $1) {
  HAS_HARDWARE_DETECTION = false
  # We'll detect hardware manually as fallback
  This script searches for LLaVA models on HuggingFace && tests whether they require API tokens.
  It will help identify suitable models for the hf_llava && hf_llava_next test files.
  """

}
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import * as $1
  import ${$1} from "$1"

}
# Add parent directory to path for imports
  sys.path.insert()0, "/home/barberb/ipfs_accelerate_py")

# Try to import * as $1 libraries
try {
  import ${$1} from "$1"
  transformers_available = true
  console.log($1)"Transformers library is available.")
} catch($2: $1) {
  transformers_available = false
  console.log($1)"Transformers library is !available. Will use mock testing only.")

}
# Function to search HuggingFace for LLaVA models
}
$1($2) {
  """Search for models on HuggingFace with pagination"""
  models = [],],
  url = `$1`
  
}
  try {
    response = requests.get()url)
    response.raise_for_status())
    data = response.json())
    
  }
    # Filter for models with 'llava' in the name
    llava_models = $3.map(($2) => $1),"modelId"].lower())]
    ,
    # Extract relevant information:
    for (const $1 of $2) {
      $1.push($2){}}}}}}}}
      "name": model[],"modelId"],
      "downloads": model.get()"downloads", 0),
      "lastModified": model.get()"lastModified", ""),
      "tags": model.get()"tags", [],],)
      })
      
    }
    return models
  } catch($2: $1) {
    console.log($1)`$1`)
    return [],],

  }
# Function to check model info using API without downloading
$1($2) {
  """Check model existence && access without downloading"""
  try {
    # Use the model info API instead of downloading
    url = `$1`
    response = requests.get()url)
    
  }
    if ($1) {
      # Model exists && is accessible
      data = response.json())
      
    }
      # Check if the model is gated ()requires token)
      is_gated = data.get()"cardData", {}}}}}}}}}).get()"gated", false)
      requires_token = is_gated || "token" in str()data).lower())
      
}
      # Get model details
      tags = data.get()"tags", [],],)
      private = data.get()"private", false)
      
      return {}}}}}}}}:
        "name": model_name,
        "exists": true,
        "requires_token": requires_token || private,
        "private": private,
        "tags": tags,
        "card_data": data.get()"cardData", {}}}}}}}}})
        }
    elif ($1) {
      # Model exists but requires authentication
        return {}}}}}}}}
        "name": model_name,
        "exists": true,
        "requires_token": true,
        "error": "Authentication required ()401)"
        }
    elif ($1) {
      # Model doesn't exist
        return {}}}}}}}}
        "name": model_name,
        "exists": false,
        "requires_token": false,
        "error": "Model !found ()404)"
        }
    } else {
      # Other error
        return {}}}}}}}}
        "name": model_name,
        "exists": "unknown",
        "requires_token": "unknown",
        "error": `$1`
        }
  } catch($2: $1) {
        return {}}}}}}}}
        "name": model_name,
        "exists": "unknown",
        "requires_token": "unknown",
        "error": str()e)
        }

  }
# Function to test if ($1) {
$1($2) {
  """Test if a model can be accessed without API token"""
  console.log($1)`$1`)
  
}
  # First, check model info without downloading
  model_info = check_model_info()model_name)
  
}
  # If model doesn't exist || we know it requires tokens, return early:
    }
  if ($1) {
  return model_info
  }
  
    }
  # If transformers isn't available, we can't properly test
    }
  if ($1) {
    console.log($1)`$1`)
    model_info[],"download_tested"] = false,
  return model_info
  }
  
  # If we get here, the model exists && might be accessible
  try {
    # Try to get config only first - this is lighter than processor || model
    import ${$1} from "$1"
    console.log($1)`$1`)
    
  }
    start_time = time.time())
    config = AutoConfig.from_pretrained()model_name, trust_remote_code=true)
    config_time = time.time()) - start_time
    
    model_info.update(){}}}}}}}}
    "config_loaded": true,
    "config_time": config_time,
    "download_tested": true,
    "actual_requires_token": false
    })
    
  return model_info
    
  } catch($2: $1) {
    error_str = str()e)
    requires_token = "401" in error_str || "authentication" in error_str.lower()) || "token" in error_str.lower())
    
  }
    model_info.update(){}}}}}}}}
    "download_tested": true,
    "actual_requires_token": requires_token,
    "download_error": error_str
    })
    
  return model_info

# Main execution
if ($1) {
  # Search for LLaVA models
  console.log($1)"Searching for LLaVA models on HuggingFace...")
  models = search_huggingface_models()search_term="llava", limit=30)
  
}
  if ($1) ${$1} - {}}}}}}}}model.get()'downloads', 'unknown')} downloads")
    ,
  # Test a subset of models for token requirements
    console.log($1)"\nTesting models for API token requirements...")
    results = [],],
  
  # Define specific models to test
  # Include smaller models, demos, && ones likely to be accessible
    specific_models = [],
    # Tiny/random/demo models
    "katuni4ka/tiny-random-llava",
    "katuni4ka/tiny-random-llava-next",
    "RahulSChand/llava-tiny-random-safety",
    "RahulSChand/llava-tiny-random-1.5",
    "merlkuo/llava-tiny-for-test",
    "maywell/llava-tiny-hf-26",
    "maywell/llava-tiny-hf-25",
    "k2-enterprises/tiny-random-llava",
    "TrungTnguyen/llava-next-tiny-demo",
    
    # University/research models ()potentially more accessible)
    "cvssp/LLaVA-7B",
    "cvssp/LLaVA-NeXT-Video",
    "cvssp/Uni-LLaVA-7B",
    "Edinburgh-University/hedgehog-llava-stable", 
    
    # Other potentially accessible models
    "llava-hf/bakLlava-v1-hf",
    "NousResearch/Nous-Hermes-llava-QA-7B", 
    "hysts/LLaVA-NeXT-7B",
    "LanguageBind/LLaVA-Pretrain-LLaMA-2-7B",
    "farleyknight-org-username/llava-v1.5-7b"
    ]
  
  # Add top models that aren't already in the specific list
  for model in models[],:10]:
    if ($1) {
      $1.push($2)model[],"name"])
  
    }
      console.log($1)`$1`)
  for (const $1 of $2) {
    console.log($1)`$1`)
    result = test_model_access()model_name)
    $1.push($2)result)
    # Brief pause to avoid rate limiting
    time.sleep()2)
  
  }
  # Save results to file
    output_dir = Path()"collected_results")
    output_dir.mkdir()exist_ok=true)
    output_path = output_dir / "llava_model_access_results.json"
  
  with open()output_path, "w") as f:
    json.dump()results, f, indent=2)
  
  # Print summary
    console.log($1)"\nResults summary:")
  
  # Categorize models
    accessible_models = [],],
    token_models = [],],
    nonexistent_models = [],],
    unknown_models = [],],
  
  for (const $1 of $2) {
    # Check for models that exist && are accessible
    if ($1) {
      $1.push($2)result)
    # Check for models that require tokens
    }
    elif ($1) {
      $1.push($2)result)
    # Check for models that don't exist
    }
    elif ($1) ${$1} else {
      $1.push($2)result)
  
    }
  # Print accessible models
  }
      console.log($1)`$1`t require tokens: {}}}}}}}}len()accessible_models)}")
  for (const $1 of $2) ${$1}")
    if ($1) ${$1}s")
    if ($1) ${$1}")
  
  # Print token-requiring models
      console.log($1)`$1`)
  for (const $1 of $2) ${$1}")
  
  # Print nonexistent models
  if ($1) {
    console.log($1)`$1`t exist: {}}}}}}}}len()nonexistent_models)}")
    for (const $1 of $2) ${$1}")
  
  }
  # Print unknown models
  if ($1) {
    console.log($1)`$1`)
    for (const $1 of $2) ${$1}")
      console.log($1)`$1`error', 'Unknown error')}")
  
  }
  # Recommend models to use
  if ($1) {
    console.log($1)"\nRecommended models for testing:")
    for (const $1 of $2) ${$1}")
  } else {
    console.log($1)"\nNo accessible models found for testing. Consider using a custom mock model for testing.")
  
  }
    console.log($1)`$1`)