/**
 * IPFS Accelerate API Backends
 * 
 * This module provides a collection of API backends for various LLM providers.
 */

// Export types
export * from './types';

// Export base class
export * from './base';

// Export utilities
export * from './utils/device_mapper';

// Re-export all backends
export * from './claude';
export * from './groq';
export * from './hf_tei';
export * from './hf_tgi';
export * from './ollama';
export * from './gemini';
export * from './openai';
export * from './openai_mini';
export * from './hf_tei_unified';
export * from './hf_tgi_unified';
export * from './sample_backend';
export * from './ovms';
export * from './vllm';
export * from './vllm_unified';
export * from './opea';
export * from './s3_kit';
export * from './llvm';

import { BaseApiBackend } from './base';
import { Groq } from './groq';
import { Gemini } from './gemini';
import { Ollama } from './ollama';
import { HfTei } from './hf_tei';
import { HfTgi } from './hf_tgi';
import { HfTeiUnified } from './hf_tei_unified';
import { HfTgiUnified } from './hf_tgi_unified';
import { SampleBackend } from './sample_backend';
import { Claude } from './claude';
import { OpenAI } from './openai';
import { OpenAiMini } from './openai_mini';
import { OVMS } from './ovms';
import { VLLM } from './vllm';
import { VllmUnified } from './vllm_unified';
import { OPEA } from './opea';
import { S3Kit } from './s3_kit';
import { LLVM } from './llvm';

// Export a registry of all backends
export const apiBackends = {
  // Register backends as they are implemented
  claude: Claude,
  groq: Groq,
  gemini: Gemini,
  ollama: Ollama,
  openai: OpenAI,
  openai_mini: OpenAiMini,
  hf_tei: HfTei,
  hf_tgi: HfTgi,
  hf_tei_unified: HfTeiUnified,
  hf_tgi_unified: HfTgiUnified,
  sample_backend: SampleBackend,
  ovms: OVMS,
  vllm: VLLM,
  vllm_unified: VllmUnified,
  opea: OPEA,
  s3_kit: S3Kit,
  llvm: LLVM
};

// Function to get a backend by name
export function getBackend(name: string) {
  if (name in apiBackends) {
    return apiBackends[name as keyof typeof apiBackends];
  }
  throw new Error(`API backend ${name} not found`);
}

/**
 * Create an API backend instance by name
 */
export function createApiBackend(
  name: string, 
  resources: Record<string, any> = {}, 
  metadata: Record<string, any> = {}
): BaseApiBackend | null {
  try {
    const BackendClass = getBackend(name);
    return new BackendClass(resources, metadata);
  } catch (e) {
    console.error(`API backend "${name}" not found. Available backends: ${Object.keys(apiBackends).join(', ')}`);
    return null;
  }
}

/**
 * Get available API backend names
 */
export function getAvailableBackends(): string[] {
  return Object.keys(apiBackends);
}

/**
 * Find compatible backend for a model name
 */
export function findCompatibleBackend(
  modelName: string,
  resources: Record<string, any> = {}, 
  metadata: Record<string, any> = {}
): BaseApiBackend | null {
  for (const [name, BackendClass] of Object.entries(apiBackends)) {
    const backend = new BackendClass(resources, metadata);
    if (backend.isCompatibleModel(modelName)) {
      return backend;
    }
  }
  
  return null;
}