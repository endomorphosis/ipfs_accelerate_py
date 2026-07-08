/**
 * WebGPU Optimizer
 * Advanced optimization for WebGPU tensor operations
 */

import { Tensor } from '../../../tensor/tensor';
import { WebGPUBackend } from '../backend';
import { WebGPUOperationFusion, FusionOpType, FusionPattern, FusionConfig } from './operation_fusion';
import { 
  getSpecializedMatmulShader, 
  getSpecializedElementwiseShader,
  getSpecializedActivationShader,
  getSpecializedTransposeShader,
  getBrowserOptimizedShader,
  SpecializedShaderOptions
} from './specialized_shaders';

/**
 * WebGPU optimizer configuration
 */
export interface WebGPUOptimizerConfig {
  /** Whether to enable operation fusion */
  enableOperationFusion?: boolean;
  
  /** Whether to enable specialized shaders */
  enableSpecializedShaders?: boolean;
  
  /** Whether to enable browser-specific optimizations */
  enableBrowserOptimizations?: boolean;
  
  /** Whether to enable tensor memory optimizations */
  enableMemoryOptimizations?: boolean;
  
  /** Maximum operations to track in sequence for fusion */
  maxOperationTrackingSequence?: number;
  
  /** Configuration for operation fusion */
  fusionConfig?: FusionConfig;
  
  /** Options for specialized shaders */
  shaderOptions?: SpecializedShaderOptions;
}

/**
 * Default optimizer configuration
 */
const DEFAULT_CONFIG: WebGPUOptimizerConfig = {
  enableOperationFusion: true,
  enableSpecializedShaders: true,
  enableBrowserOptimizations: true,
  enableMemoryOptimizations: true,
  maxOperationTrackingSequence: 10,
  fusionConfig: {
    maxFusionLength: 5,
    enableAutoFusion: true,
    enabledPatterns: [
      FusionPattern.LinearActivation,
      FusionPattern.ElementWiseChain,
      FusionPattern.BinaryUnary
    ]
  },
  shaderOptions: {
    workgroupSize: 256,
    useSpecializedLayout: true,
    useFastMath: true
  }
};

/**
 * Advanced WebGPU optimization manager
 */
export class WebGPUOptimizer {
  /** WebGPU backend reference */
  private backend: WebGPUBackend;
  
  /** Configuration options */
  private config: WebGPUOptimizerConfig;
  
  /** Operation fusion utility */
  private fusionEngine: WebGPUOperationFusion;
  
  /** Browser type for optimizations */
  private browserType: 'chrome' | 'firefox' | 'safari' | 'edge' | 'unknown';
  
  /** Recent operation sequence for tracking fusion opportunities */
  private recentOperations: {
    opType: FusionOpType,
    inputs: string[],
    output: string
  }[] = [];
  
  /** Cached shader modules */
  private shaderCache: Map<string, any> = new Map();
  
  /**
   * Constructor
   * @param backend WebGPU backend
   * @param config Optimizer configuration
   */
  constructor(backend: WebGPUBackend, config: WebGPUOptimizerConfig = {}) {
    this.backend = backend;
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    // Initialize fusion engine
    this.fusionEngine = new WebGPUOperationFusion(
      backend, 
      this.config.fusionConfig
    );
    
    // Detect browser type
    this.browserType = this.detectBrowserType();
    
    // Initialize browser-specific options
    this.initializeBrowserOptions();
  }
  
  /**
   * Detect the current browser type
   * @returns Browser type string
   */
  private detectBrowserType(): 'chrome' | 'firefox' | 'safari' | 'edge' | 'unknown' {
    if (typeof navigator === 'undefined') {
      return 'unknown';
    }
    
    const ua = navigator.userAgent;
    
    if (ua.includes('Firefox')) {
      return 'firefox';
    }
    
    if (ua.includes('Edg/')) {
      return 'edge';
    }
    
    if (ua.includes('Chrome')) {
      return 'chrome';
    }
    
    if (ua.includes('Safari') && !ua.includes('Chrome')) {
      return 'safari';
    }
    
    return 'unknown';
  }
  
  /**
   * Initialize browser-specific optimization options
   */
  private initializeBrowserOptions(): void {
    if (!this.config.enableBrowserOptimizations) {
      return;
    }
    
    // Set browser-specific shader options
    if (this.config.shaderOptions) {
      this.config.shaderOptions.browserOptimized = this.browserType as any;
    }
    
    // Adjust fusion settings for different browsers
    if (this.config.fusionConfig) {
      // Firefox benefits from smaller fusion chains
      if (this.browserType === 'firefox') {
        this.config.fusionConfig.maxFusionLength = 3;
      }
      
      // Safari does well with longer chains on Apple Silicon
      if (this.browserType === 'safari') {
        this.config.fusionConfig.maxFusionLength = 8;
      }
    }
  }
  
  /**
   * Track an operation for potential fusion
   * @param opType Operation type
   * @param inputs Input tensor IDs
   * @param output Output tensor ID
   */
  trackOperation(
    opType: FusionOpType,
    inputs: string[],
    output: string
  ): void {
    if (!this.config.enableOperationFusion) {
      return;
    }
    
    // Add to recent operations
    this.recentOperations.push({
      opType,
      inputs,
      output
    });
    
    // Trim if exceeding max length
    if (this.recentOperations.length > this.config.maxOperationTrackingSequence!) {
      this.recentOperations.shift();
    }
    
    // Check for fusion opportunities
    this.detectFusionOpportunities();
  }
  
  /**
   * Detect potential fusion opportunities in recent operations
   */
  private detectFusionOpportunities(): void {
    if (this.recentOperations.length < 2) {
      return;
    }
    
    // Look for operation chains where output of one is input to the next
    for (let length = 2; length <= Math.min(this.recentOperations.length, this.config.fusionConfig!.maxFusionLength!); length++) {
      const sequence = this.recentOperations.slice(-length);
      
      // Check if this is a valid chain
      let validChain = true;
      for (let i = 1; i < sequence.length; i++) {
        // Check if the output of the previous operation is an input to this one
        const prevOutput = sequence[i-1].output;
        if (!sequence[i].inputs.includes(prevOutput)) {
          validChain = false;
          break;
        }
      }
      
      if (validChain) {
        // Extract operation types
        const opTypes = sequence.map(op => op.opType);
        
        // Check if this sequence can be fused
        if (this.fusionEngine.canFuse(opTypes)) {
          // This sequence can be fused - log it for potential future execution
          console.log(`Fusion opportunity detected: ${opTypes.join(' -> ')}`);
          // In a real implementation, we would register this fusion pattern
          // and potentially modify the backend to use it in the future
        }
      }
    }
  }
  
  /**
   * Get a specialized shader for an operation
   * @param operation Operation type
   * @param shapeInfo Shape information for specialization
   * @returns Specialized shader source
   */
  getSpecializedShader(
    operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
    shapeInfo?: any
  ): string {
    if (!this.config.enableSpecializedShaders) {
      throw new Error('Specialized shaders are disabled');
    }
    
    // Generate cache key
    let cacheKey = operation;
    if (shapeInfo) {
      cacheKey += JSON.stringify(shapeInfo);
    }
    
    // Check cache
    if (this.shaderCache.has(cacheKey)) {
      return this.shaderCache.get(cacheKey);
    }
    
    // Use browser-specific optimizations if enabled
    if (this.config.enableBrowserOptimizations && this.browserType !== 'unknown') {
      const shader = this.getBrowserOptimizedShader(operation, shapeInfo);
      this.shaderCache.set(cacheKey, shader);
      return shader;
    }
    
    // Otherwise use generic specialized shaders
    let shader: string;
    switch (operation) {
      case 'matmul':
        shader = getSpecializedMatmulShader(
          shapeInfo?.M || 1024,
          shapeInfo?.K || 1024,
          shapeInfo?.N || 1024,
          this.config.shaderOptions
        );
        break;
      
      case 'add':
      case 'subtract':
      case 'multiply':
      case 'divide':
        shader = getSpecializedElementwiseShader(operation, this.config.shaderOptions);
        break;
      
      case 'relu':
      case 'sigmoid':
      case 'tanh':
      case 'softmax':
        shader = getSpecializedActivationShader(operation, this.config.shaderOptions);
        break;
      
      case 'transpose':
        shader = getSpecializedTransposeShader(this.config.shaderOptions);
        break;
      
      default:
        throw new Error(`Unsupported operation for specialization: ${operation}`);
    }
    
    // Cache and return
    this.shaderCache.set(cacheKey, shader);
    return shader;
  }
  
  /**
   * Execute a fusion sequence if possible
   * @param operations Operation sequence to execute
   * @param inputs Input tensors
   * @param shapeInfo Optional shape information to help with specialization
   * @returns Output tensor, or null if fusion not possible
   */
  async tryExecuteFusion<T>(
    operations: FusionOpType[],
    inputs: Tensor<T>[],
    shapeInfo?: any
  ): Promise<Tensor<T> | null> {
    if (!this.config.enableOperationFusion) {
      return null;
    }
    
    // Check if this sequence can be fused
    if (!this.fusionEngine.canFuse(operations)) {
      return null;
    }
    
    try {
      console.log(`Executing fused operations: ${operations.join(' -> ')}`);
      
      // Execute the fused operation
      const result = await this.fusionEngine.executeFusedOperations(inputs, operations);
      
      // For performance analysis
      if (result) {
        console.log(`Successfully executed fused operations: ${operations.join(' -> ')}`);
      }
      
      return result;
    } catch (error) {
      console.warn('Fusion execution failed:', error);
      // If fusion fails, we should fall back to executing operations one by one
      // This is handled by the caller
      return null;
    }
  }
  
  /**
   * Optimize a tensor operation by choosing the best implementation
   * @param operation Operation type
   * @param inputs Input tensors
   * @param shapeInfo Shape information for optimization
   * @returns Optimized output tensor
   */
  async optimizeOperation<T>(
    operation: FusionOpType,
    inputs: Tensor<T>[],
    shapeInfo?: any
  ): Promise<Tensor<T> | null> {
    // Track this operation
    const inputIds = inputs.map(t => this.getTensorId(t));
    const outputId = `output_${Date.now()}`;
    this.trackOperation(operation, inputIds, outputId);
    
    // First, check if we can fuse with previous operations
    // Look for recent operations that may be fusible with this one
    if (this.config.enableOperationFusion && this.recentOperations.length > 0) {
      // Try to find chains of operations where the output of one is the input to this one
      const potentialChains: FusionOpType[][] = [];
      
      // Start with just this operation
      potentialChains.push([operation]);
      
      // Try to build chains of increasing length by looking at recent operations
      for (let length = 2; length <= Math.min(this.recentOperations.length + 1, this.config.fusionConfig!.maxFusionLength!); length++) {
        // Get the most recent operations, excluding the current one
        const recentOps = this.recentOperations.slice(-length + 1);
        
        // Build the operation chain
        const opChain: FusionOpType[] = recentOps.map(op => op.opType);
        opChain.push(operation);
        
        // Check if this is a valid chain
        let validChain = true;
        for (let i = 1; i < recentOps.length; i++) {
          if (!recentOps[i].inputs.includes(recentOps[i-1].output)) {
            validChain = false;
            break;
          }
        }
        
        // For the current operation, we need to check if it uses the output of the last recent op
        if (validChain && recentOps.length > 0) {
          const lastRecentOutputId = recentOps[recentOps.length - 1].output;
          if (!inputIds.some(id => id === lastRecentOutputId)) {
            validChain = false;
          }
        }
        
        if (validChain) {
          potentialChains.push(opChain);
        }
      }
      
      // Try the longest chain first
      potentialChains.sort((a, b) => b.length - a.length);
      
      for (const chain of potentialChains) {
        // Skip single operation chains - we'll handle those below
        if (chain.length === 1) continue;
        
        // Check if this chain can be fused
        if (this.fusionEngine.canFuse(chain)) {
          // Prepare the correct input tensors
          // This is simplified - in a real implementation, we would need to track
          // and use the actual input tensors for each operation in the chain
          const chainInputs = [...inputs];
          
          // Try to execute the fusion
          try {
            const result = await this.tryExecuteFusion(chain, chainInputs, shapeInfo);
            if (result) {
              return result;
            }
          } catch (error) {
            console.warn(`Failed to execute fusion chain ${chain.join(' -> ')}:`, error);
            // Continue to next approach
          }
        }
      }
    }
    
    // If fusion with previous operations didn't work, try specialized shader for this operation
    if (this.config.enableSpecializedShaders && 
        ['matmul', 'add', 'subtract', 'multiply', 'divide', 'relu', 'sigmoid', 'tanh', 'softmax', 'transpose'].includes(operation)) {
      try {
        // Get the specialized shader
        const shaderSource = this.getSpecializedShader(operation as any, shapeInfo);
        
        // Get the WebGPU device
        const device = (this.backend as any).device;
        if (!device) {
          return null;
        }
        
        // Create an output tensor
        let outputShape: number[];
        
        // Determine output shape based on operation
        if (operation === 'matmul') {
          outputShape = [inputs[0].shape[0], inputs[1].shape[1]];
        } else if (['reshape', 'transpose'].includes(operation)) {
          // For reshape, the second input typically contains the target shape
          outputShape = inputs.length > 1 ? [...inputs[1].shape] : [...inputs[0].shape];
        } else {
          // For element-wise operations
          outputShape = [...inputs[0].shape];
        }
        
        const outputTensor = new Tensor<T>(
          outputShape,
          null,
          {
            dataType: inputs[0].dataType,
            backend: 'webgpu',
            device: this.backend.id
          }
        );
        
        // Here, we would create and execute the specialized shader pipeline using shaderSource
        // However, this requires complex WebGPU setup that would duplicate much of executeFusedOperations
        
        // For now, since all the shader implementations are already in executeFusedOperations,
        // we'll just delegate to it by treating this as a single-operation "fusion"
        try {
          const fusedResult = await this.tryExecuteFusion([operation], inputs, shapeInfo);
          if (fusedResult) {
            return fusedResult;
          }
        } catch (error) {
          console.warn(`Failed to execute specialized implementation for ${operation}:`, error);
          // Continue to next approach
        }
      } catch (error) {
        console.warn('Specialized shader generation failed:', error);
        // Continue to next approach
      }
    }
    
    // If no optimization was applied, return null
    // This signals to the backend that it should use the standard implementation
    return null;
  }
  
  /**
   * Gets a unique ID for a tensor
   * @param tensor Tensor to get ID for
   * @returns Unique tensor ID
   */
  private getTensorId<T>(tensor: Tensor<T>): string {
    return `tensor_${tensor.shape.join('x')}_${tensor.dataType}_${Date.now()}`;
  }
  
  /**
   * Get browser-specific optimized shader
   * @param operation Operation type
   * @param shapeInfo Shape information for optimization
   * @returns Optimized shader for the current browser
   */
  private getBrowserOptimizedShader(
    operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
    shapeInfo?: any
  ): string {
    // Apply browser-specific optimizations based on browser type
    switch (this.browserType) {
      case 'chrome':
        return this.getChromeOptimizedShader(operation, shapeInfo);
      case 'firefox':
        return this.getFirefoxOptimizedShader(operation, shapeInfo);
      case 'safari':
        return this.getSafariOptimizedShader(operation, shapeInfo);
      case 'edge':
        return this.getEdgeOptimizedShader(operation, shapeInfo);
      default:
        // Default to the standard specialized shaders
        return getBrowserOptimizedShader(operation, this.browserType as any);
    }
  }
  
  /**
   * Get Chrome-optimized shader
   * Chrome WebGPU is optimized for larger workgroups and has good support for compute shaders
   * @param operation Operation type
   * @param shapeInfo Shape information
   * @returns Chrome-optimized shader
   */
  private getChromeOptimizedShader(
    operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
    shapeInfo?: any
  ): string {
    // Create Chrome-specific options
    const options: SpecializedShaderOptions = {
      browserOptimized: 'chrome',
      workgroupSize: 256, // Chrome handles larger workgroups well
      useFastMath: true,  // Chrome benefits from fast math optimizations
      algorithmSwitchThreshold: 1024 * 1024 * 4 // Higher threshold for tiling in Chrome
    };
    
    // Chrome benefits from larger tiles in matrix operations
    if (operation === 'matmul') {
      // Adjust tile size based on matrix dimensions
      const M = shapeInfo?.M || 1024;
      const K = shapeInfo?.K || 1024;
      const N = shapeInfo?.N || 1024;
      
      // For very large matrices, use tiling approach in Chrome
      if (M >= 2048 || N >= 2048) {
        options.workgroupSize = 512; // Chrome can handle larger workgroups 
      }
      
      return getSpecializedMatmulShader(M, K, N, options);
    }
    
    // For element-wise and activation operations, Chrome works well with optimized implementations
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      return getSpecializedElementwiseShader(operation as any, options);
    }
    
    if (['relu', 'sigmoid', 'tanh', 'softmax'].includes(operation)) {
      return getSpecializedActivationShader(operation as any, options);
    }
    
    if (operation === 'transpose') {
      return getSpecializedTransposeShader(options);
    }
    
    // Fallback to default browser optimized shader
    return getBrowserOptimizedShader(operation, 'chrome');
  }
  
  /**
   * Get Firefox-optimized shader
   * Firefox benefits from smaller workgroups and has excellent audio processing performance
   * @param operation Operation type
   * @param shapeInfo Shape information
   * @returns Firefox-optimized shader
   */
  private getFirefoxOptimizedShader(
    operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
    shapeInfo?: any
  ): string {
    // Create Firefox-specific options
    const options: SpecializedShaderOptions = {
      browserOptimized: 'firefox',
      workgroupSize: 128, // Firefox works better with smaller workgroups
      useFastMath: true,  // Fast math approximations work well in Firefox
      algorithmSwitchThreshold: 512 * 512 // Lower threshold for tiling in Firefox
    };
    
    // Firefox benefits from smaller tiles in matrix operations
    if (operation === 'matmul') {
      const M = shapeInfo?.M || 1024;
      const K = shapeInfo?.K || 1024;
      const N = shapeInfo?.N || 1024;
      
      // Firefox works best with 8x8 workgroups for matrices
      options.workgroupSize = 64;
      
      return getSpecializedMatmulShader(M, K, N, options);
    }
    
    // Firefox is optimized for audio processing, so operations like FFT
    // and audio-related operations get special treatment
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      // Use vector operations where possible in Firefox
      return getSpecializedElementwiseShader(operation as any, options);
    }
    
    if (['relu', 'sigmoid', 'tanh', 'softmax'].includes(operation)) {
      return getSpecializedActivationShader(operation as any, options);
    }
    
    if (operation === 'transpose') {
      // Firefox benefits from smaller workgroups for transpose
      options.workgroupSize = 8;
      return getSpecializedTransposeShader(options);
    }
    
    // Fallback to default browser optimized shader
    return getBrowserOptimizedShader(operation, 'firefox');
  }
  
  /**
   * Get Safari-optimized shader
   * Safari/WebKit on Apple Silicon has specific optimization patterns
   * @param operation Operation type
   * @param shapeInfo Shape information
   * @returns Safari-optimized shader
   */
  private getSafariOptimizedShader(
    operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
    shapeInfo?: any
  ): string {
    // Create Safari-specific options for Apple Silicon
    const options: SpecializedShaderOptions = {
      browserOptimized: 'safari',
      workgroupSize: 512, // Apple Silicon can handle very large workgroups
      useFastMath: false, // Apple GPUs benefit from higher precision
      algorithmSwitchThreshold: 2048 * 2048 // Higher threshold for Apple GPUs
    };
    
    // Safari WebGPU benefits from optimizations for Apple Silicon
    if (operation === 'matmul') {
      const M = shapeInfo?.M || 1024;
      const K = shapeInfo?.K || 1024;
      const N = shapeInfo?.N || 1024;
      
      // Apple Silicon benefits from larger workgroups
      options.workgroupSize = 1024;
      
      return getSpecializedMatmulShader(M, K, N, options);
    }
    
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      return getSpecializedElementwiseShader(operation as any, options);
    }
    
    if (['relu', 'sigmoid', 'tanh', 'softmax'].includes(operation)) {
      return getSpecializedActivationShader(operation as any, options);
    }
    
    if (operation === 'transpose') {
      return getSpecializedTransposeShader(options);
    }
    
    // Fallback to default browser optimized shader
    return getBrowserOptimizedShader(operation, 'safari');
  }
  
  /**
   * Get Edge-optimized shader
   * Edge (Chromium-based) benefits from similar optimizations to Chrome with a few tweaks
   * @param operation Operation type
   * @param shapeInfo Shape information
   * @returns Edge-optimized shader
   */
  private getEdgeOptimizedShader(
    operation: 'matmul' | 'add' | 'subtract' | 'multiply' | 'divide' | 'relu' | 'sigmoid' | 'tanh' | 'softmax' | 'transpose',
    shapeInfo?: any
  ): string {
    // Create Edge-specific options (similar to Chrome but with tweaks)
    const options: SpecializedShaderOptions = {
      browserOptimized: 'edge',
      workgroupSize: 256,
      useFastMath: true,
      algorithmSwitchThreshold: 1024 * 1024 * 2
    };
    
    // Edge tends to work well with WebNN for neural network operations
    // but for WebGPU operations, it benefits from similar optimizations to Chrome
    if (operation === 'matmul') {
      const M = shapeInfo?.M || 1024;
      const K = shapeInfo?.K || 1024;
      const N = shapeInfo?.N || 1024;
      
      return getSpecializedMatmulShader(M, K, N, options);
    }
    
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      return getSpecializedElementwiseShader(operation as any, options);
    }
    
    if (['relu', 'sigmoid', 'tanh', 'softmax'].includes(operation)) {
      return getSpecializedActivationShader(operation as any, options);
    }
    
    if (operation === 'transpose') {
      return getSpecializedTransposeShader(options);
    }
    
    // Fallback to default browser optimized shader
    return getBrowserOptimizedShader(operation, 'edge');
  }
  
  /**
   * Garbage collect shader caches if they grow too large
   */
  garbageCollect(): void {
    // Clear shader cache if it grows too large
    if (this.shaderCache.size > 100) {
      this.shaderCache.clear();
    }
    
    // Reset operation tracking
    this.recentOperations = [];
  }
  
  /**
   * Get the optimal memory layout for a tensor based on browser and operation
   * Different browsers have different optimal memory access patterns
   * @param operation Operation type
   * @param shape Tensor shape
   * @returns Optimal layout configuration
   */
  getOptimalMemoryLayout(
    operation: FusionOpType,
    shape: number[]
  ): { 
    rowMajor: boolean, 
    paddedShape?: number[], 
    alignment: number 
  } {
    // Default layout properties
    const defaultLayout = {
      rowMajor: true,   // Row-major is most common
      alignment: 4,     // 4-byte alignment is standard
    };
    
    // Apply browser-specific memory layout optimizations
    switch (this.browserType) {
      case 'chrome':
        return this.getChromeOptimalLayout(operation, shape);
      case 'firefox':
        return this.getFirefoxOptimalLayout(operation, shape);
      case 'safari':
        return this.getSafariOptimalLayout(operation, shape);
      case 'edge':
        return this.getEdgeOptimalLayout(operation, shape);
      default:
        return defaultLayout;
    }
  }
  
  /**
   * Get Chrome-optimized memory layout
   * @param operation Operation type
   * @param shape Tensor shape
   * @returns Optimal layout for Chrome
   */
  private getChromeOptimalLayout(
    operation: FusionOpType,
    shape: number[]
  ): { rowMajor: boolean, paddedShape?: number[], alignment: number } {
    // Chrome WebGPU works well with aligned buffer sizes
    // Chrome prefers column-major layout for some matrix operations
    
    if (operation === 'matmul') {
      // For matrix multiplication, Chrome does better with column-major for large matrices
      const isLarge = shape[0] > 1024 || shape[1] > 1024;
      
      if (isLarge) {
        // For large matrices, use column-major layout
        return {
          rowMajor: false,
          alignment: 16, // 16-byte alignment works well for Chrome with large matrices
          paddedShape: this.getPaddedShape(shape, 16) // Pad to multiple of 16
        };
      } else {
        // For smaller matrices, use row-major layout
        return {
          rowMajor: true,
          alignment: 8, // 8-byte alignment is sufficient for smaller matrices
          paddedShape: this.getPaddedShape(shape, 8)
        };
      }
    }
    
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      // Element-wise operations benefit from contiguous memory layout
      return {
        rowMajor: true,
        alignment: 16,
        paddedShape: this.getPaddedShape(shape, 16)
      };
    }
    
    if (['relu', 'sigmoid', 'tanh', 'gelu', 'silu'].includes(operation)) {
      // Activation functions also benefit from alignment
      return {
        rowMajor: true,
        alignment: 16,
        paddedShape: this.getPaddedShape(shape, 16)
      };
    }
    
    // Default Chrome layout
    return {
      rowMajor: true,
      alignment: 8,
      paddedShape: this.getPaddedShape(shape, 8)
    };
  }
  
  /**
   * Get Firefox-optimized memory layout
   * @param operation Operation type
   * @param shape Tensor shape
   * @returns Optimal layout for Firefox
   */
  private getFirefoxOptimalLayout(
    operation: FusionOpType,
    shape: number[]
  ): { rowMajor: boolean, paddedShape?: number[], alignment: number } {
    // Firefox WebGPU works best with smaller alignments
    // Firefox prefers row-major layout in most cases
    
    if (operation === 'matmul') {
      // Firefox does better with row-major layout for matrix multiplication
      return {
        rowMajor: true,
        alignment: 8, // 8-byte alignment for Firefox
        paddedShape: this.getPaddedShape(shape, 8)
      };
    }
    
    // Firefox works well with audio processing operations
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      // Element-wise operations in Firefox work best with smaller alignments
      return {
        rowMajor: true,
        alignment: 4, // 4-byte alignment is sufficient
        paddedShape: this.getPaddedShape(shape, 4)
      };
    }
    
    // Default Firefox layout
    return {
      rowMajor: true,
      alignment: 4,
      paddedShape: this.getPaddedShape(shape, 4)
    };
  }
  
  /**
   * Get Safari-optimized memory layout
   * @param operation Operation type
   * @param shape Tensor shape
   * @returns Optimal layout for Safari
   */
  private getSafariOptimalLayout(
    operation: FusionOpType,
    shape: number[]
  ): { rowMajor: boolean, paddedShape?: number[], alignment: number } {
    // Safari on Apple Silicon benefits from larger alignments and Metal-friendly layouts
    
    if (operation === 'matmul') {
      // For matrix multiplication, Safari does best with column-major layout
      return {
        rowMajor: false, // Column-major works better for Metal
        alignment: 16,   // 16-byte alignment for Metal
        paddedShape: this.getPaddedShape(shape, 16, 4) // Pad to multiple of 16, with 4-element padding
      };
    }
    
    if (['add', 'subtract', 'multiply', 'divide'].includes(operation)) {
      // Element-wise operations
      return {
        rowMajor: true,
        alignment: 16, // Apple GPUs like 16-byte alignment
        paddedShape: this.getPaddedShape(shape, 16)
      };
    }
    
    // Default Safari layout
    return {
      rowMajor: true,
      alignment: 16,
      paddedShape: this.getPaddedShape(shape, 16)
    };
  }
  
  /**
   * Get Edge-optimized memory layout
   * @param operation Operation type
   * @param shape Tensor shape
   * @returns Optimal layout for Edge
   */
  private getEdgeOptimalLayout(
    operation: FusionOpType,
    shape: number[]
  ): { rowMajor: boolean, paddedShape?: number[], alignment: number } {
    // Edge (Chromium-based) has similar characteristics to Chrome
    
    if (operation === 'matmul') {
      // For matrix multiplication
      const isLarge = shape[0] > 1024 || shape[1] > 1024;
      
      if (isLarge) {
        return {
          rowMajor: false, // Column-major for large matrices
          alignment: 16,
          paddedShape: this.getPaddedShape(shape, 16)
        };
      } else {
        return {
          rowMajor: true, // Row-major for smaller matrices
          alignment: 8,
          paddedShape: this.getPaddedShape(shape, 8)
        };
      }
    }
    
    // Default Edge layout
    return {
      rowMajor: true,
      alignment: 8,
      paddedShape: this.getPaddedShape(shape, 8)
    };
  }
  
  /**
   * Calculate padded shape for optimal memory alignment
   * @param shape Original tensor shape
   * @param alignment Alignment value (4, 8, 16, etc.)
   * @param padWith Optional padding element count for dimensions
   * @returns Padded shape
   */
  private getPaddedShape(
    shape: number[],
    alignment: number = 4,
    padWith: number = 1
  ): number[] {
    // For vectors and matrices, pad last dimension to alignment
    if (shape.length <= 2) {
      const result = [...shape];
      const lastDim = result[result.length - 1];
      
      // Compute padding to next multiple of alignment
      const remainder = lastDim % alignment;
      if (remainder !== 0) {
        result[result.length - 1] = lastDim + (alignment - remainder);
      }
      
      return result;
    }
    
    // For higher dimensional tensors, pad each dimension individually
    return shape.map((dim, index) => {
      // Last dimension is padded to alignment
      if (index === shape.length - 1) {
        const remainder = dim % alignment;
        return remainder === 0 ? dim : dim + (alignment - remainder);
      }
      
      // Other dimensions are padded by padWith
      const remainder = dim % padWith;
      return remainder === 0 ? dim : dim + (padWith - remainder);
    });
  }
}