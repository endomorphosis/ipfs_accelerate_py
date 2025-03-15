import { Message } from '../types';

export interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message?: {
      role: string;
      content: string;
      tool_calls?: Array<{
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    delta?: {
      content?: string;
      role?: string;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAIRequest {
  model: string;
  messages?: Message[];
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  functions?: Array<{
    name: string;
    description?: string;
    parameters: {
      type: string;
      properties: {
        [key: string]: any;
      };
      required?: string[];
    };
  }>;
  tools?: Array<{
    type: string;
    function?: {
      name: string;
      description?: string;
      parameters: {
        type: string;
        properties: {
          [key: string]: any;
        };
        required?: string[];
      };
    };
  }>;
  tool_choice?: string | {
    type: string;
    function?: {
      name: string;
    };
  };
}

export interface OpenAIEmbeddingRequest {
  model: string;
  input: string | string[];
  encoding_format?: string;
}

export interface OpenAIEmbeddingResponse {
  object: string;
  data: Array<{
    object: string;
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface OpenAIModerationRequest {
  input: string;
  model?: string;
}

export interface OpenAIModerationResponse {
  id: string;
  model: string;
  results: Array<{
    flagged: boolean;
    categories: {
      [key: string]: boolean;
    };
    category_scores: {
      [key: string]: number;
    };
  }>;
}

export interface OpenAIImageRequest {
  model?: string;
  prompt: string;
  n?: number;
  size?: string;
  style?: string;
  quality?: string;
  response_format?: string;
}

export interface OpenAIImageResponse {
  created: number;
  data: Array<{
    url?: string;
    b64_json?: string;
    revised_prompt?: string;
  }>;
}

export interface OpenAISpeechRequest {
  model: string;
  input: string;
  voice: string;
  response_format?: string;
  speed?: number;
}

export interface OpenAITranscriptionRequest {
  model: string;
  file: any;
  language?: string;
  prompt?: string;
  response_format?: string;
  temperature?: number;
}

export interface OpenAITranscriptionResponse {
  text: string;
}