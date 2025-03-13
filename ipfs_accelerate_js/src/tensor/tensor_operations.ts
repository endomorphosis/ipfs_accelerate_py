/**
 * Tensor operations for IPFS Accelerate
 */
import { Tensor } from '../interfaces';

export class TensorOperations {
  /**
   * Creates a new tensor with the given shape and data
   */
  static createTensor(shape: number[], data: Float32Array | Int32Array | Uint8Array, dtype: string = 'float32'): Tensor {
    return {
      shape,
      data,
      dtype
    };
  }
  
  /**
   * Creates a tensor filled with zeros
   */
  static zeros(shape: number[], dtype: string = 'float32'): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    let data: Float32Array | Int32Array | Uint8Array;
    
    if (dtype === 'float32') {
      data = new Float32Array(size);
    } else if (dtype === 'int32') {
      data = new Int32Array(size);
    } else if (dtype === 'uint8') {
      data = new Uint8Array(size);
    } else {
      throw new Error(`Unsupported dtype: ${dtype}`);
    }
    
    return {
      shape,
      data,
      dtype
    };
  }
  
  /**
   * Creates a tensor filled with ones
   */
  static ones(shape: number[], dtype: string = 'float32'): Tensor {
    const tensor = this.zeros(shape, dtype);
    
    if (dtype === 'float32') {
      const data = tensor.data as Float32Array;
      data.fill(1);
    } else if (dtype === 'int32') {
      const data = tensor.data as Int32Array;
      data.fill(1);
    } else if (dtype === 'uint8') {
      const data = tensor.data as Uint8Array;
      data.fill(1);
    }
    
    return tensor;
  }
  
  /**
   * Creates a tensor filled with random values
   */
  static random(shape: number[], dtype: string = 'float32'): Tensor {
    const tensor = this.zeros(shape, dtype);
    const size = shape.reduce((a, b) => a * b, 1);
    
    if (dtype === 'float32') {
      const data = tensor.data as Float32Array;
      for (let i = 0; i < size; i++) {
        data[i] = Math.random();
      }
    } else if (dtype === 'int32') {
      const data = tensor.data as Int32Array;
      for (let i = 0; i < size; i++) {
        data[i] = Math.floor(Math.random() * 100);
      }
    } else if (dtype === 'uint8') {
      const data = tensor.data as Uint8Array;
      for (let i = 0; i < size; i++) {
        data[i] = Math.floor(Math.random() * 256);
      }
    }
    
    return tensor;
  }
  
  /**
   * Gets the size of a tensor
   */
  static size(tensor: Tensor): number {
    return tensor.shape.reduce((a, b) => a * b, 1);
  }
  
  /**
   * Reshapes a tensor to a new shape
   */
  static reshape(tensor: Tensor, newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    const oldSize = this.size(tensor);
    
    if (newSize !== oldSize) {
      throw new Error(`Cannot reshape tensor of size ${oldSize} to size ${newSize}`);
    }
    
    return {
      shape: newShape,
      data: tensor.data,
      dtype: tensor.dtype
    };
  }
}

export class TensorSharingManager {
  private sharedTensors: Map<string, Tensor> = new Map();
  private refCounts: Map<string, number> = new Map();
  
  /**
   * Shares a tensor for reuse across models
   */
  shareTensor(id: string, tensor: Tensor): void {
    this.sharedTensors.set(id, tensor);
    this.refCounts.set(id, (this.refCounts.get(id) || 0) + 1);
  }
  
  /**
   * Gets a shared tensor by ID
   */
  getTensor(id: string): Tensor | null {
    return this.sharedTensors.get(id) || null;
  }
  
  /**
   * Releases a shared tensor
   */
  releaseTensor(id: string): void {
    if (!this.refCounts.has(id)) {
      return;
    }
    
    const count = this.refCounts.get(id)! - 1;
    if (count <= 0) {
      this.sharedTensors.delete(id);
      this.refCounts.delete(id);
    } else {
      this.refCounts.set(id, count);
    }
  }
  
  /**
   * Gets all shared tensor IDs
   */
  getSharedTensorIds(): string[] {
    return Array.from(this.sharedTensors.keys());
  }
  
  /**
   * Clears all shared tensors
   */
  clearAll(): void {
    this.sharedTensors.clear();
    this.refCounts.clear();
  }
}

// Singleton instance
export const tensorSharingManager = new TensorSharingManager();
