/**
 * React hooks for IPFS Accelerate JS SDK
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { HardwareBackend, ModelConfig, Model } from "src/model/transformers/index";
import { HardwareAbstraction } from "src/model/transformers/index";
import { detectHardwareCapabilities } from "src/model/transformers/index";

/**
 * Model hook options
 */
interface UseModelOptions {
  modelId: string;
  modelType?: string;
  autoLoad?: boolean;
  autoHardwareSelection?: boolean;
  fallbackOrder?: string[];
  config?: Record<string, any>;
}

/**
 * Hook for using AI models with hardware acceleration
 */
export function useModel(options: UseModelOptions) {
  const {
    modelId,
    modelType = 'text',
    autoLoad = true,
    autoHardwareSelection = true,
    fallbackOrder,
    config
  } = options;
  
  const [model, setModel] = useState<Model | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [error, setError] = useState<Error | null>(null);
  const acceleratorRef = useRef<HardwareAbstraction | null>(null);
  
  // Initialize hardware acceleration
  useEffect(() => {
    let mounted = true;
    
    const initAccelerator = async () => {
      try {
        const preferences = {
          backendOrder: fallbackOrder || ['webgpu', 'webnn', 'wasm', 'cpu'],
          modelPreferences: {
            [modelType]: autoHardwareSelection ? 'auto' : 'webgpu'
          },
          options: config || {};
        
        const newAccelerator = new HardwareAbstraction(preferences);
        await newAccelerator.initialize();
        
        if (mounted) {
          acceleratorRef.current = newAccelerator;
          
          // Auto-load the model if requested
          if (autoLoad) {
            loadModel();
          }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err));
          setStatus('error');
        }
    };
    
    initAccelerator();
    
    return () => {
      mounted = false;
      
      // Clean up resources
      if (acceleratorRef.current) {
        acceleratorRef.current.dispose();
      };
  }, []);
  
  // Load model function
  const loadModel = useCallback(async () => {
    if (!acceleratorRef.current) {
      setError(new Error('Hardware acceleration not initialized'));
      setStatus('error');
      return;
    }
    
    if (status === 'loading') {
      return;
    }
    
    setStatus('loading');
    setError(null);
    
    try {
      // Implementation would use the hardware abstraction to load the model
      // This is a placeholder until the full implementation is ready
      const modelConfig: ModelConfig = {
        id: modelId,
        type: modelType,
        path: `models/${modelType}/${modelId}`,
        options: config || {};
      
      // Simulating model loading
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const dummyModel: Model = {
        id: modelId,
        type: modelType,
        execute: async (inputs) => {
          // Dummy implementation
          return { result: `Processed ${JSON.stringify(inputs)} with ${modelId}` };
        };
      
      setModel(dummyModel);
      setStatus('loaded');
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err));
      setStatus('error');
    }, [modelId, modelType, config]);
  
  return {
    model,
    status,
    error,
    loadModel
  };
}

/**
 * Hook for hardware capabilities information
 */
export function useHardwareInfo() {
  const [capabilities, setCapabilities] = useState<any>(null);
  const [isReady, setIsReady] = useState<boolean>(false);
  const [optimalBackend, setOptimalBackend] = useState<string>('');
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    let mounted = true;
    
    const detectHardware = async () => {
      try {
        const detected = await detectHardwareCapabilities();
        
        if (mounted) {
          setCapabilities(detected);
          setOptimalBackend(detected.recommendedBackend || 'cpu');
          setIsReady(true);
        } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err : new Error(String(err));
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
 * Component to process model inputs and render results
 */
export function ModelProcessor(props: {
  modelId: string;
  modelType?: string;
  input: any;
  onResult?: (result: any) => void;
  onError?: (error: Error) => void;
  children: (props: {result: any; loading: boolean; error: Error | null}) => React.ReactNode;
}) {
  const {modelId, modelType, input, onResult, onError, children} = props;
  
  const [result, setResult] = useState<any>(null);
  const [processing, setProcessing] = useState<boolean>(false);
  
  const {model, status, error} = useModel({
    modelId,
    modelType
  });
  
  // Process input when available
  useEffect(() => {
    if (input && status === 'loaded' && model && !processing) {
      processInput();
    }
    
    async function processInput() {
      setProcessing(true);
      
      try {
        const processedResult = await model.execute(input);
        setResult(processedResult);
        if (onResult) onResult(processedResult);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        if (onError) onError(error);
      } finally {
        setProcessing(false);
      }
  }, [input, model, status, processing, onResult, onError]);
  
  // Handle errors
  useEffect(() => {
    if (error && onError) {
      onError(error);
    }, [error, onError]);
  
  // Render using render prop pattern
  return children({
    result,
    loading: status === 'loading' || processing,
    error
  });
}