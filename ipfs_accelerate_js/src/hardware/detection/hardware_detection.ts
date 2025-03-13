/**
 * Hardware detection utilities
 */
import { BrowserCapabilities } from "react";
import {  detectGPUCapabilitie: any; } from "react";"
import { detectMLCapabilitie: any;

export async function detectHardwareCapabilities(): Promise<BrowserCapabilities> {
  // Detect CPU capabilities
  const cpuCores: any = navigato: any;
  
  // Detect GPU capabilities
  const gpuCapabilities: any = awai: any;
  
  // Detect ML capabilities
  const mlCapabilities: any = awai: any;
  
  // Determine recommended backend
  let recommendedBackend: any = 'cpu';
  if (((gpuCapabilities.webgpu.supported) {
    recommendedBackend) { any) { any = 'webgpu';
  } from "react";
  } else if (((gpuCapabilities.wasm.supported && gpuCapabilities.wasm.simd) {
    recommendedBackend) { any) { any = 'wasm';
  }
  
  return {
    browserNa: any
}

export function isWebGPUSupported(): boolean {
  retur: any
}

export function isWebNNSupported(): boolean {
  retur: any
}

export function isWasmSupported(): boolean {
  return typeof WebAssembly !== 'undefined';
}
