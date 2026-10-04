/**
 * React Hooks for IPFS Accelerate JavaScript SDK
 * 
 * This file provides React hooks for easy integration of the SDK in React applications.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  WebAccelerator, 
  createAccelerator, 
  detectCapabilities, 
  HardwareBackendType,
  ModelType,
  ModelConfig
} from './ipfs_accelerate_js_index';
import { Model } from './ipfs_accelerate_js_model_loader';

/**
 * Hook for loading a model
 */
export function useModel(options: {
  modelId: string;
  modelType?: ModelType;
  autoLoad?: boolean;
  autoHardwareSelection?: boolean;
  fallbackOrder?: HardwareBackendType[];
  config?: ModelConfig;
}) {
  const { 
    modelId, 
    modelType = 'text', 
    autoLoad = true, 
    autoHardwareSelection = true, 
    fallbackOrder,
    config 
  } = options;
  
  const [model, setModel] = useState<Model | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'loaded' | 'error'>('idle');
  const [error, setError] = useState<Error | null>(null);
  const acceleratorRef = useRef<WebAccelerator | null>(null);
  
  // Initialize accelerator
  useEffect(() => {
    let mounted = true;
    
    const initAccelerator = async () => {
      try {
        const newAccelerator = await createAccelerator({
          autoDetectHardware: autoHardwareSelection,
          fallbackOrder
        });
        
        if (mounted) {
          acceleratorRef.current = newAccelerator;
          
          // Auto-load model if requested
          if (autoLoad) {
            loadModel();
          }
        } else {
          // Clean up if component unmounted
          await newAccelerator.dispose();
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setStatus('error');
        }
      }
    };
    
    initAccelerator();
    
    return () => {
      mounted = false;
      
      // Cleanup accelerator on unmount
      if (acceleratorRef.current) {
        acceleratorRef.current.dispose().catch(console.error);
      }
    };
  }, []);
  
  // Load model function
  const loadModel = useCallback(async () => {
    if (!acceleratorRef.current) {
      setError(new Error('Accelerator not initialized'));
      setStatus('error');
      return;
    }
    
    if (status === 'loading') {
      return;
    }
    
    setStatus('loading');
    setError(null);
    
    try {
      const modelLoader = acceleratorRef.current['modelLoader'];
      
      if (!modelLoader) {
        throw new Error('Model loader not available');
      }
      
      const loadedModel = await modelLoader.loadModel({
        modelId,
        modelType,
        autoSelectHardware,
        fallbackOrder,
        config
      });
      
      if (!loadedModel) {
        throw new Error(`Failed to load model ${modelId}`);
      }
      
      setModel(loadedModel);
      setStatus('loaded');
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setStatus('error');
    }
  }, [modelId, modelType, autoHardwareSelection, fallbackOrder, config, status]);
  
  // Switch backend function
  const switchBackend = useCallback(async (newBackend: HardwareBackendType) => {
    if (!model) {
      setError(new Error('Model not loaded'));
      return false;
    }
    
    try {
      return await model.switchBackend(newBackend);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      return false;
    }
  }, [model]);
  
  return {
    model,
    status,
    error,
    loadModel,
    switchBackend
  };
}

/**
 * Hook for hardware capabilities
 */
export function useHardwareInfo() {
  const [capabilities, setCapabilities] = useState<any>(null);
  const [isReady, setIsReady] = useState(false);
  const [optimalBackend, setOptimalBackend] = useState<HardwareBackendType | null>(null);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    let mounted = true;
    
    const detectHardware = async () => {
      try {
        const detected = await detectCapabilities();
        
        if (mounted) {
          setCapabilities(detected);
          setOptimalBackend(detected.optimalBackend);
          setIsReady(true);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      }
    };
    
    detectHardware();
    
    return () => {
      mounted = false;
    };
  }, []);
  
  return {
    capabilities,
    isReady,
    optimalBackend,
    error
  };
}

/**
 * Hook for P2P network status
 */
export function useP2PStatus() {
  const [isEnabled, setIsEnabled] = useState(false);
  const [peerCount, setPeerCount] = useState(0);
  const [networkHealth, setNetworkHealth] = useState(0);
  const [error, setError] = useState<Error | null>(null);
  const p2pManagerRef = useRef<any>(null);
  
  // Enable P2P network
  const enableP2P = useCallback(async () => {
    try {
      // This would be implemented with actual P2P functionality
      // For now, it's a placeholder
      
      // Simulate network connection
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Update state
      setIsEnabled(true);
      setPeerCount(5); // Sample value
      setNetworkHealth(0.8); // 80% health
      
      return true;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      return false;
    }
  }, []);
  
  // Disable P2P network
  const disableP2P = useCallback(async () => {
    try {
      // This would disconnect from the P2P network
      
      // Simulate disconnection
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Update state
      setIsEnabled(false);
      setPeerCount(0);
      setNetworkHealth(0);
      
      return true;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      return false;
    }
  }, []);
  
  return {
    isEnabled,
    peerCount,
    networkHealth,
    enableP2P,
    disableP2P,
    error
  };
}

/**
 * Hook for using the acceleration functionality
 */
export function useAcceleration(options: {
  modelId: string;
  modelType: ModelType;
  backend?: HardwareBackendType;
  autoInitialize?: boolean;
}) {
  const { modelId, modelType, backend, autoInitialize = true } = options;
  
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [capabilities, setCapabilities] = useState<any>(null);
  const acceleratorRef = useRef<WebAccelerator | null>(null);
  
  // Initialize accelerator
  useEffect(() => {
    let mounted = true;
    
    if (autoInitialize) {
      initializeAccelerator();
    }
    
    async function initializeAccelerator() {
      try {
        const newAccelerator = await createAccelerator({
          preferredBackend: backend
        });
        
        if (mounted) {
          acceleratorRef.current = newAccelerator;
          setCapabilities(newAccelerator.getCapabilities());
          setIsReady(true);
        } else {
          // Clean up if component unmounted
          await newAccelerator.dispose();
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      }
    }
    
    return () => {
      mounted = false;
      
      // Cleanup accelerator on unmount
      if (acceleratorRef.current) {
        acceleratorRef.current.dispose().catch(console.error);
      }
    };
  }, [autoInitialize, backend]);
  
  // Acceleration function
  const accelerate = useCallback(async (input: any, config?: any) => {
    if (!acceleratorRef.current) {
      throw new Error('Accelerator not initialized');
    }
    
    if (!isReady) {
      throw new Error('Accelerator not ready');
    }
    
    return await acceleratorRef.current.accelerate({
      modelId,
      modelType,
      input,
      config: {
        backend,
        ...config
      }
    });
  }, [modelId, modelType, backend, isReady]);
  
  // Manual initialization
  const initialize = useCallback(async () => {
    if (isReady) {
      return true;
    }
    
    try {
      const newAccelerator = await createAccelerator({
        preferredBackend: backend
      });
      
      acceleratorRef.current = newAccelerator;
      setCapabilities(newAccelerator.getCapabilities());
      setIsReady(true);
      
      return true;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      return false;
    }
  }, [backend, isReady]);
  
  return {
    accelerate,
    initialize,
    isReady,
    error,
    capabilities
  };
}

// Export a complete component
export function ModelProcessor(props: {
  modelId: string;
  modelType: ModelType;
  input?: any;
  onResult?: (result: any) => void;
  onError?: (error: Error) => void;
  children?: React.ReactNode;
}) {
  const { modelId, modelType, input, onResult, onError, children } = props;
  
  const [result, setResult] = useState<any>(null);
  const [processing, setProcessing] = useState(false);
  
  const { accelerate, isReady, error } = useAcceleration({
    modelId,
    modelType
  });
  
  // Process input when it changes
  useEffect(() => {
    if (input && isReady && !processing) {
      processInput();
    }
    
    async function processInput() {
      setProcessing(true);
      
      try {
        const accelerationResult = await accelerate(input);
        setResult(accelerationResult.result);
        onResult?.(accelerationResult.result);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        onError?.(error);
      } finally {
        setProcessing(false);
      }
    }
  }, [input, isReady, processing, accelerate, onResult, onError]);
  
  // Handle errors
  useEffect(() => {
    if (error) {
      onError?.(error);
    }
  }, [error, onError]);
  
  // Render children with props
  if (typeof children === 'function') {
    return children({
      result,
      processing,
      isReady,
      error
    });
  }
  
  return children || null;
}