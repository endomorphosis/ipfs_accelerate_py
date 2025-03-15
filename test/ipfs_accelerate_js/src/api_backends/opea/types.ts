/**
 * Type definitions for OpenAI Proxy API Extension (OPEA)
 */

import { ApiRequestOptions } from '../types';

export interface OPEARequestData {
  model?: string;
  messages?: Array<{ role: string; content: string | any[]; [key: string]: any }>;
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string | string[];
  stream?: boolean;
  [key: string]: any;
}

export interface OPEAResponse {
  choices?: Array<{
    message?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    text?: string;
    delta?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    index?: number;
    finish_reason?: string;
    [key: string]: any;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    [key: string]: any;
  };
  id?: string;
  created?: number;
  object?: string;
  model?: string;
  [key: string]: any;
}

export interface OPEAStreamChunk {
  choices?: Array<{
    delta?: {
      content?: string;
      role?: string;
      [key: string]: any;
    };
    index?: number;
    finish_reason?: string | null;
    [key: string]: any;
  }>;
  id?: string;
  created?: number;
  model?: string;
  object?: string;
  [key: string]: any;
}

export interface OPEARequestOptions extends ApiRequestOptions {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string | string[];
  [key: string]: any;
}