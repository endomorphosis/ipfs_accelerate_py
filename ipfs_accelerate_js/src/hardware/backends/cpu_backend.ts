/**
 * CPU backend implementation (fallback)
 */
import { HardwareBackend } from '../../interfaces';

export class CPUBackend implements HardwareBackend {
  private initialized: boolean = false;
  
  constructor() {
    this.initialized = false;
  }
  
  async initialize(): Promise<boolean> {
    // CPU backend is always available
    this.initialized = true;
    return true;
  }
  
  async execute<T = any, U = any>(inputs: T): Promise<U> {
    // Placeholder implementation for CPU execution
    console.warn("Using CPU backend (slow)");
    return {} as U;
  }
  
  destroy(): void {
    this.initialized = false;
  }
}
