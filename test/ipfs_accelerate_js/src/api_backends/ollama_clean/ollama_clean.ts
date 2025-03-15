import { BaseApiBackend } from '../base';
import { ApiMetadata, ApiRequestOptions, Message, ChatCompletionResponse, StreamChunk } from '../types';
import { OllamaCleanResponse, OllamaCleanRequest } from './types';

export class OllamaClean extends BaseApiBackend {

  private apiEndpoint: string = 'https://api.ollamaclean.com/v1/chat';

  constructor(resources: Record<string, any> = {}, metadata: ApiMetadata = {}) {
    super(resources, metadata);
  }


}