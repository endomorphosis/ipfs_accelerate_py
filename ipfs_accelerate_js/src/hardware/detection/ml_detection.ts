/**
 * Machine Learning capabilities detection
 */

export interface MLCapabilities {
  webnn: {
    supported: boolean;
    features: string[];
  };
}

export async function detectMLCapabilities(): Promise<MLCapabilities> {
  // Check for WebNN support
  const isWebNNSupported = 'ml' in navigator;
  
  return {
    webnn: {
      supported: isWebNNSupported,
      features: isWebNNSupported ? ['basic'] : []
    }
  };
}
