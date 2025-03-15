import { ApiMetadata, ChatMessage, ChatOptions, BaseStreamChunk } from '../types';

export interface OpenAiMiniOptions {
  apiUrl?: string;
  maxRetries?: number;
  requestTimeout?: number;
  useRequestQueue?: boolean;
  debug?: boolean;
}

export interface OpenAiMiniChatRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string | string[];
}

export interface OpenAiMiniChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: {
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAiMiniStreamChunk extends BaseStreamChunk {
  id: string;
  model: string;
  created: number;
  delta: {
    role?: string;
    content?: string;
  };
  finish_reason: string | null;
}

export interface OpenAiMiniFileUploadOptions {
  purpose?: string;
  fileName?: string;
}

export interface OpenAiMiniFileResponse {
  id: string;
  object: string;
  created_at: number;
  filename: string;
  purpose: string;
  bytes: number;
  status: string;
  status_details: string | null;
}

export interface OpenAiMiniAudioOptions {
  model?: string;
  prompt?: string;
  response_format?: string;
  temperature?: number;
  language?: string;
}

export interface OpenAiMiniTTSOptions {
  model?: string;
  voice?: string;
  speed?: number;
  response_format?: string;
}

export interface OpenAiMiniImageOptions {
  model?: string;
  size?: string;
  quality?: string;
  n?: number;
  response_format?: string;
  style?: string;
}

export interface OpenAiMiniImageResponse {
  created: number;
  data: {
    url?: string;
    b64_json?: string;
    revised_prompt?: string;
  }[];
}

export type OpenAiMiniApiMetadata = ApiMetadata & {
  openai_mini_api_key?: string;
};