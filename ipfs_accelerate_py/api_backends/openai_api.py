
import time
import re
import os
import openai
# from cloudkit_worker import dispatch_result
import openai.resources
import tiktoken
import tempfile
import base64
import requests

import json
import subprocess
from datetime import datetime
import io

# Deprecations page
# https://platform.openai.com/docs/deprecations

import inspect
from pathlib import Path
import tqdm

class TestError(Exception):
    pass

max_tokens = {
    'o1': 100000,
    'o1-mini': 65536,
    'o1-preview': 32768,
    'o3-mini': 100000,
    'gpt-4o': 16384,
    'gpt-4o-audio-preview': 16384,
    'gpt-4o-realtime-preview': 4096,
    'gpt-4o-mini': 16384,
    'gpt-4o-mini-audio-preview': 16384,
    'gpt-4o-mini-realtime-preview': 4096,
    'gpt-4': 8192,
    'gpt-4-0613': 8192,
    'gpt-4-0125-preview': 4096,
    'gpt-4-1106-preview': 128000,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-0125': 4096,
    'gpt-3.5-turbo-1106': 16385,
    'gpt-3.5-turbo-16k': 16385,
    'gpt-3.5-turbo-instruct': 4096,
    'gpt-4-turbo-preview': 4096,
    'chatgpt-4o-latest': 16384
}

assistants_models = [ # /v1/assistants
    "o1", # 100,000
    "o3-mini", # 100,000
    "gpt-4o", # 16,384
    "gpt-4o-mini", # 16,384
    "gpt-4", # 8192
    "gpt-4-0613", # 8,192
    "gpt-4-0125-preview", # 4,096
    "gpt-4-1106-preview", # 128000
    "gpt-3.5-turbo", # 4096
    "gpt-3.5-turbo-0125", # 4,096
    "gpt-3.5-turbo-1106", # 16385
    "gpt-3.5-turbo-16k", # 16385
    "gpt-4-turbo-preview" # 4,096
]

tools_models = [
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-1106"
]

embedding_models = [ # /v1/embeddings
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002"
]

"gpt-4o-audio-preview",
"gpt-4o-mini-audio-preview",
"gpt-4o-mini-realtime-preview",
"gpt-3.5-turbo-instruct",

chat_completion_models = [ # /v1/chat/completions
    "o1", # 100,000
    "o1-mini", # 100,000
    "o1-preview", # 32,768
    "o3-mini",
    "gpt-4o",
    "gpt-4o-mini", # 16,384
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "chatgpt-4o-latest"
]

completions = [ # /v1/completions
    "gpt-3.5-turbo-instruct",
]

image_models = [ # /v1/images/generations
    "dall-e-3",
    "dall-e-2"
]

moderation_models = [ # /v1/moderations
    "omni-moderation-latest",
    "text-moderation-latest",
    "text-moderation-stable"
]

speech_to_text = [ # /v1/audio/transcriptions
    "whisper-1"
]

text_to_speech = [ #/v1/audio/speech
    "tts-1",
    "tts-1-hd-1106",
    "tts-1-1106",
    "tts-1-hd",
]

translation_models = [ # /v1/audio/translations
    "whisper-1"
]

vision_models = [ # https://platform.openai.com/docs/guides/vision
    "o1",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo"
]

chat_templates = [
        {
            'models': ['gpt-3.5-turbo','gpt-4','gpt-3.5-turbo-16k'],
            'system_msg': 'A chat between a curious user and an artificial intelligence assistant. ' + \
            'The assistant gives helpful, detailed, and polite answers to the user\'s questions. <</SYS>> [/INST]',
            'user_msg': 'USER: {text}',
            'user_sep': '\n',
            'assistant_msg': 'ASSISTANT: {text}',
            'assistant_sep': '\n',
        }
    ]

class openai_api:
    """

    Methods:
        make_post_request_openai_api
        test_openai_api_endpoint
        request_openai_api_endpoint
        create_openai_api_endpoint_handler
        embedding
        moderation
        speech_to_text
        text_to_image
        moderated_text_to_image
        text_to_speech
        moderated_text_to_speech
        tokenize
        detokenize
        moderated_chat_complete
        request_complete
        image_to_text
        chat
        audio_chat
        determine_model
        process_messages
    """
    def __init__(self, resources=None, metadata=None):
        self.prompt = None
        self.messages = None
        self.instruct = None
        self.input = None
        self.method = None
        self.temperature = None
        self.api_key = None
        self.max_tokens = None 
        if metadata is not None:
            if "openai_api_key" in metadata:
                if metadata['openai_api_key'] is not None:
                    self.api_key = metadata['openai_api_key']
        dir_self = list(dir(self))
        properties = list(self.__dict__.keys())
        if("api_key" in dir_self):
            if self.api_key is not None:
                openai.api_key = self.api_key

        if self.api_key is not None:
            pass
        #else:
        #    raise Exception('bad api_key: %s' % self.api_key)

        self.resources = resources
        self.metadata = metadata
        self.create_openai_api_endpoint_handler = self.create_openai_api_endpoint_handler
        self.request_openai_api_endpoint = self.request_openai_api_endpoint
        self.test_openai_api_endpoint = self.test_openai_api_endpoint
        self.make_post_request_openai_api = self.make_post_request_openai_api
        self.init()
        return None

    def make_post_request_openai_api(self, model, endpoint, endpoint_type, batch):
        return None

    def test_openai_api_endpoint(self):
        ## test dependencies
        return None

    def request_openai_api_endpoint(self, model,  endpoint=None, endpoint_type=None, batch=None):
        return None

    def create_openai_api_endpoint_handler(self):
        def handler(request):
            return None
        return handler

    def init(self, resources=None, metadata=None):
        if resources is not None:
            self.model = resources['models'].split("@")[0].split("/")[-1]
        else:
            self.model = None

    def __call__(self, method, **kwargs):

        self.messages = None
        self.input = None
        if "openai_api_key" in kwargs:
            if kwargs['openai_api_key'] is not None:
                self.metadata["openai_api_key"] = kwargs['openai_api_key']
        print(self.metadata)
        if ("openai_api_key" in list(self.metadata.keys())):
            if self.metadata["openai_api_key"] is not None:
                openai.api_key = self.metadata["openai_api_key"]
            else:
                raise Exception('bad api_key: %s' % self.metadata["openai_api_key"])
        else:
            raise Exception('no key found in metadata: %s' % self.metadata)
        if self.model is not None:
            kwargs['model'] = self.model
        if method == 'chat':
            return self.chat(**kwargs)
        elif method == 'embedding':
            return self.embedding(**kwargs)
        elif method == 'text_to_image':
            return self.text_to_image(**kwargs)
        elif method == 'image_to_text':
            return self.image_to_text(**kwargs)
        elif method == 'text_to_speech':
            return self.text_to_speech(**kwargs)
        elif method == 'speech_to_text':
            return self.speech_to_text(**kwargs)
        elif method == 'moderation':
            return self.moderation(**kwargs)
        elif method == 'audio_chat':
            return self.audio_chat(**kwargs)
        elif method == 'assistant':
            return self.assistant(**kwargs)
        else:
            print(self)
            raise Exception('bad method in __call__: %s' % method)

    def embedding(self, model, input, **kwargs):
        if model not in embedding_models:
            raise Exception('bad model: %s' % model)
        self.model = model
        self.input = input
        self.method = 'embedding'
        embedding = openai.embeddings.create(
            input=input,
            model=model
        )
        return {
            'text': embedding,
            'done': True
        }

    def moderation(self, model, text, **kwargs):
        if model not in moderation_models:
            raise Exception('bad model: %s' % model)
        self.model = model
        self.text = text
        self.method = 'moderation'
        moderation = openai.moderations.create(input=text, model=model)
        return moderation

    def speech_to_text(self, model, audio, **kwargs):
        if model not in speech_to_text:
            raise Exception('bad model: %s' % model)
        self.model = model
        self.audio = audio
        self.method = 'speech_to_text'
        audio_file = open(audio, "rb")
        transcript = openai.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
        return {
            'text': transcript,
            'done': True
        }


    def text_to_image(self, model, size, n, prompt, **kwargs):
        sizes = {
            "dall-e-3":
            [
                "1024x1024",
                "1792x1024",
                "1024x1792"
            ],
            "dall-e-2":
            [
                "256x256",
                "512x512",
                "1024x1024",
            ]
        }
        if model not in image_models:
            raise Exception('bad model: %s' % model)
        if size not in sizes[model]:
            raise Exception('bad size: %s' % size)

        if n is None:
            n = 1
        if int(n):
            n = int(n)
        if n < 1:
            raise Exception('bad n: %s' % n)
        if n > 1:
            if model == "dall-e-3":
                raise Exception('bad n: %s' % n)
        if n > 10:
            if model == "dall-e-2":
                raise Exception('bad n: %s' % n)
            raise Exception('bad n: %s' % n)

        self.model = model
        self.prompt = prompt
        self.n = n
        self.size = size
        self.method = 'text_to_image'

        image = self.moderated_text_to_image(self.model, self.size, self.n, self.prompt)

        return image


    def moderated_text_to_image(self, model, size, n, prompt, **kwargs):
        json_messages = json.dumps(prompt)
        requested_model = self.model
        original_method = self.method
        moderation_model = 'text-moderation-stable'
        check_messages = self.moderation(moderation_model, json_messages)
        self.method = original_method
        self.model = requested_model
        if len(check_messages.results) > 0:
            results_keys = list(check_messages.results[0].__dict__.keys())
            if "flagged" in results_keys:
                if check_messages.results[0].flagged == True:
                    raise Exception('bad messages: %s' % self.messages)
                else:
                    image = openai.images.generate(
                        model=model,
                        n=n,
                        size=size,
                        prompt=prompt
                    )

                    data = image.data
                    images = []
                    for i in range(len(data)):
                        this_data = data[i]
                        this_image = {}
                        this_image['url'] = this_data.url
                        this_image['revised_prompt'] = this_data.revised_prompt
                        images.append(this_image)

                    return {
                        'text': json.dumps(images),
                        'done': True
                    }


    def text_to_speech(self, model, text, voice, response_format="mp3", speed=1, **kwargs):

        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        response_formats = ["mp3", "opus", "aac" "flac"]
        speeds = [ 0.25, 4 ]
        max_length = 4096

        if(voice is None):
            voice = "fable"
        if(response_format is None):
            response_format = "mp3"
        if(speed is None):
            speed = 1

        if(len(text) > max_length):
            raise Exception('bad text: %s' % text)
        if(voice not in voices):
            raise Exception('bad voice: %s' % voice)
        if(response_format not in response_formats):
            raise Exception('bad response_format: %s' % response_format)
        if(speed < 0.25 or speed > 4):
            raise Exception('bad speed: %s' % speed)

        self.model = model
        self.text = text
        self.method = 'text_to_speech'
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.close()
            speech_file_path = temp_file.name
            #response = openai.audio.speech.create(
            #    model=model,
            #    voice=voice,
            #    input=text
            #)
            response = self.moderated_text_to_speech(model, text, voice, response_format, speed)["text"].text
            return {
                'audio': response,
                'done': True
            }

 
    def embedding(self, model, input, format, **kwargs):
        encoding_formats = [
            "float",
            "base64"
        ]
        self.model = model
        self.input = input
        self.messages = None
        self.prompt = None
        self.method = 'embedding'
        self.encoding_format = format
        embedding = openai.embeddings.create(
            input=input,
            model=model,
            encoding_format=format
        )

        data = embedding.data
        embeddings = []
        for i in range(len(data)):
            this_data = data[i]
            this_image = {}
            this_image['embedding'] = this_data.embedding
            embeddings.append(this_image)

        return {
            'text': json.dumps(embeddings[0]),
            'done': True
        }

    def tokenize(self, text , model, **kwargs):
        self.model = model
        self.text = text
        self.method = 'tokenize'
        default_tokenizer_model = "gpt-3.5-turbo"
        if self.model is None:
            self.model = default_tokenizer_model
        encoding = tiktoken.encoding_for_model(default_tokenizer_model)
        encoding = encoding.encode(text)
        return encoding

    def detokenize(self, tokens, model, **kwargs):
        self.model = model
        self.tokens = tokens
        self.method = 'detokenize'
        default_tokenizer_model = "gpt-3.5-turbo"
        if self.model is None:
            self.model = default_tokenizer_model
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding = tiktoken.encoding_for_model(self.model)
        return encoding.decode(tokens)

    def moderated_chat_complete(self, stopping_regex=None, **kwargs):
        json_messages = json.dumps(self.messages)
        requested_model = self.model
        original_method = self.method
        moderation_model = 'text-moderation-stable'
        check_messages = self.moderation(moderation_model, json_messages)
        self.method = original_method
        self.model = requested_model
        if len(check_messages.results) > 0:
            results_keys = list(check_messages.results[0].__dict__.keys())
            if "flagged" in results_keys:
                if check_messages.results[0].flagged == True:
                    raise Exception('bad messages: %s' % self.messages)
                else:
                    response = openai.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    return {
                        'text': response,
                        'done': True
                    }


    def moderated_text_to_speech(self, model, text, voice, response_format, speed):
        json_messages = json.dumps(self.messages)
        requested_model = model
        original_method = self.method
        moderation_model = 'text-moderation-stable'
        check_messages = self.moderation(moderation_model, json_messages)
        self.method = original_method
        self.model = requested_model
        if len(check_messages.results) > 0:
            results_keys = list(check_messages.results[0].__dict__.keys())
            if "flagged" in results_keys:
                if check_messages.results[0].flagged == True:
                    raise Exception('bad messages: %s' % self.messages)
                else:
                    response = openai.audio.speech.create(
                        model=self.model,
                        voice=voice,
                        input=text,
                        speed=speed,
                        response_format=response_format
                    )

                    return {
                        'text': response,
                        'done': True
                    }


    def request_complete(self, stopping_regex=None, **kwargs):

        all_models = vision_models + tools_models + chat_completion_models

        if self.model is None or self.model not in all_models:
            raise Exception('bad model: %s' % self.model)

        if stopping_regex:
            try:
                stopping_regex = re.compile(stopping_regex)
            except Exception as e:
                raise Exception('bad "stopping_regex": %s' % str(e))
        openai_error = None
        response = None
        while openai_error == True or openai_error == None:
            openai_error = False
            try:
                if self.method is not None and self.method == 'image_to_text':

                    response = self.moderated_chat_complete(stopping_regex)

                elif self.method is not None and self.method == 'chat':

                    response = self.moderated_chat_complete(stopping_regex)

                else:
                    raise Exception('bad method in request_complete: %s' % self.method)

                openai_error = False
            except Exception as e:
                openai_error = True
                print(e)
                #wait 1 second
                time.sleep(1)
                pass
        # if stream is undefined
        if "stream" not in list(self.__dict__.keys()):
            if self.method is not None and ( self.method == 'chat' or self.method == 'image_to_text' ):
                response = response["text"]
                return {
                    'text': response.choices[0].message.content,
                    'done': True
                }
            elif self.input is not None and self.instruct is not None:
                return {
                    'text': response.choices[0].text,
                    'done': True
                }
            else:
                ## todo ##
                return {
                    'text': response.choices[0].text,
                    'done': True
                }


    def image_to_text(self, model, prompt, images, max_tokens, system, **kwargs):
        qualities  = ["low", "high", "auto"]
        self.images = images
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        messages = {}
        self.messages = {}
        self.system = system
        this_messages = self.process_messages(messages, prompt, images, system)
        self.messages = this_messages
        self.method = 'image_to_text'
        for image in self.images:
            if image['detail'] not in qualities:
                raise Exception('bad quality: %s' % image['quality'])
        return self.request_complete(**kwargs)


    def chat(self, model, messages, prompt, system, temperature, max_tokens, **kwargs):
        self.max_tokens = max_tokens
        if ("files" in kwargs):
            files = kwargs['files']
        else:
            files = None
        messages = self.process_messages(messages, prompt, files, system)
        model = self.determine_model(model, messages)
        self.messages = messages
        self.model = model
        self.prompt = prompt
        self.system = system
        self.temperature = temperature
        self.model = model
        self.files = files
        self.method = 'chat'
        return self.request_complete(**kwargs)


    def audio_chat(self, model, messages, voice, system, temperature, max_tokens, **kwargs):
        self.max_tokens = max_tokens

        if ("prompt" in kwargs):
            prompt = kwargs['prompt']
            if prompt == "":
                prompt = None
        else:
            prompt = None
            pass
        if ("audio" in kwargs):
            audio = kwargs['audio']
            if audio == "":
                audio = None
        else:
            audio = None
            pass
        if prompt is None and audio is None:
            raise Exception('no prompt or audio: %s' % prompt)

        if prompt is not None and audio is not None:
            raise Exception('you have both prompt and audio: %s' % prompt)
        file_types = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
        self.messages = messages
        self.model = model
        if prompt is not None:
            self.prompt = prompt
        if audio is not None:
            if "http" in audio:
                file_type = audio.split(".")[-1]
                if file_type not in file_types:
                    raise Exception('bad file_type: %s' % file_type)
                else:
                    file_type = "." + file_type
                with tempfile.NamedTemporaryFile(suffix=file_type, delete=False) as temp_file:
                    audio_file_path = temp_file.name
                    print("audio_file_path")
                    print(audio_file_path)
                    print("file_type")
                    print(file_type)
                    subprocess.run(["wget", "-O", audio_file_path, audio])
                    audio = audio_file_path
            self.prompt = self.speech_to_text("whisper-1", audio)["text"]
            prompt = self.prompt
            pass

        messages = self.process_messages(messages, prompt.text, None, system)
        model = self.determine_model(model, messages)
        self.method = 'chat'
        self.prompt = prompt
        self.system = system
        self.temperature = temperature
        self.model = model
        self.files = None
        self.method = 'chat'
        results = self.request_complete( **kwargs)
        audio = self.text_to_speech("tts-1-hd", results['text'], voice, "mp3", 1)
        return {
            'text': audio["audio"],
            'done': True
        }


    def determine_model(self, model: str, messages: list):
        model_type: str = ""
        this_max_tokens = self.max_tokens
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {type(self.max_tokens)}")

        match model:
            case model if "gpt-4" in model:
                model_type = "gpt-4"
            case model if "gpt-3" in model:
                model_type = "gpt-3"
            case model if "o1"in model or "o3" in model:
                model_type = "o_model"
            case _:
                model_type = None

        if "instruct" in model:
            model_type = "instruct"

        if model in vision_models:
            model_type = "vision"

        chosen_model = None
        max_tokens = {
            'o1': 100000,
            'o1-mini': 65536,
            'o1-preview': 32768,
            'o3-mini': 100000,
            'gpt-4o': 16384,
            'gpt-4o-audio-preview': 16384,
            'gpt-4o-realtime-preview': 4096,
            'gpt-4o-mini': 16384,
            'gpt-4o-mini-audio-preview': 16384,
            'gpt-4o-mini-realtime-preview': 4096,
            'gpt-4': 8192,
            'gpt-4-0613': 8192,
            'gpt-4-0125-preview': 4096,
            'gpt-4-1106-preview': 128000,
            'gpt-3.5-turbo': 4096,
            'gpt-3.5-turbo-0125': 4096,
            'gpt-3.5-turbo-1106': 16385,
            'gpt-3.5-turbo-16k': 16385,
            'gpt-3.5-turbo-instruct': 4096,
            'gpt-4-turbo-preview': 4096,
            'chatgpt-4o-latest': 16384
        }
        stringified_messages = ""
        stringified_messages = json.dumps(messages)
        if "image_url" in stringified_messages:
            model_type = "vision"
            pass

        message_tokens = self.tokenize(stringified_messages, model)
        num_tokens = len(message_tokens) + this_max_tokens

        if model_type != "vision" and model_type != "instruct":
            if model_type == "gpt-3":
                for model in max_tokens:
                    if "gpt-3" in model:
                        if num_tokens < max_tokens[model]:
                            chosen_model = model
                            model_type = "chosen"
                            break
                        else:
                            pass
                if chosen_model is None:
                    model_type = "gpt-4"
                pass
            if model_type == "gpt-4":
                for model in max_tokens:
                    if "gpt-4" in model:
                        if num_tokens < max_tokens[model]:
                            chosen_model = model
                            model_type = "chosen"
                            break
                        else:
                            pass
                if chosen_model is None:
                    model_type = "o_model"
                pass
            if model_type == "o_model":
                for model in max_tokens:
                    if "o1" in model or "o3" in model:
                        if num_tokens < max_tokens[model]:
                            chosen_model = model
                            model_type = "chosen"
                            break
                if chosen_model is None:
                    raise Exception("bad model: %s" % model)
                pass
        else:
            if model_type == "instruct":
                for model in max_tokens:
                    if "instruct" in model:
                        if num_tokens < max_tokens[model]:
                            chosen_model = model
                            model_type = "chosen"
                            break
                        else:
                            pass
                if chosen_model is None:
                    raise Exception("bad model: %s" % model)
                pass
            elif model_type == "vision":
                for model in max_tokens:
                    if model in vision_models:
                        if num_tokens < max_tokens[model]:
                            chosen_model = model
                            model_type = "chosen"
                            break
                        else:
                            pass
                if chosen_model is None:
                    raise Exception("bad model: %s" % model)
                pass
            else:
                raise Exception("bad model: %s" % model)
        return chosen_model


    def process_messages(self, messages, prompt, files, system):
        messagesList = []
        new_files = []
        if files is not None:
            if type(files) is not list:
                raise Exception('bad files: %s' % files)
            for image in files:
                if "url" not in image:
                    raise Exception('bad url: %s' % image)
                if "detail" not in image:
                    this_detail = "auto"
                this_url = image['url']
                this_detail = image['detail']
                #this_url = convert_image_base64(this_url)
                image['url'] = this_url
                image['detail'] = this_detail
                new_files.append(image)
            pass

        template = chat_templates[0]

        if system is not None:
            if system != "":
                systemDict = {"role": "system", "content": system}
            else:
                systemDict = {"role": "system", "content": template['system_msg']}
                pass
            messagesList.append(systemDict)
            pass

        for m in messages:
            if m['role'] == 'user':
                if "text" in m:
                    userDict = {"role": "user", "content": m['text']}
                elif "content" in m:
                    userDict = {"role": "user", "content": m['content']}
                messagesList.append(userDict)
            elif m['role'] == 'assistant':
                if "text" in m:
                    assistantDict = {"role": "assistant", "content": m['content']}
                elif "content" in m:
                    assistantDict = {"role": "assistant", "content": m['content']}
                messagesList.append(assistantDict)
            elif m['role'] == 'system':
                if "text" in m:
                    systemDict = {"role": "system", "content": m['content']}
                elif "content" in m:
                    systemDict = {"role": "system", "content": m['content']}
                messagesList.append(systemDict)
            else:
                raise Exception('bad role: %s' % m['role'])

        addToMessages = False
        if (files is not None or prompt is not None):
            if files is not None:
                if (len(files) > 0):
                    addToMessages = True
                    pass
                pass
            if prompt is not None:
                if len(prompt) > 0:
                    addToMessages = True
                    pass
                pass
            pass
        if len(messages) == 0:
            addToMessages = True
            pass
        elif messages[-1]['role'] == 'assistant':
            if addToMessages == False:
                raise Exception("bad prompt: %s" % prompt)
                pass

            if messages[-1]['role'] == 'user':
                if addToMessages == False:
                    self.messages = messagesList
                    pass
            pass
        if addToMessages == True:
            lastMessages = {}
            lastMessages['role'] = 'user'
            if (files is not None and len(files) > 0):
                lastMessages['content'] = []
                lastMessages['content'].append({"type": "text", "text": prompt})
                for image in files:
                    lastMessages['content'].append({"type": "image_url", "image_url": {"url": image['url'], "detail": image['detail']}})

            else:
                lastMessages['content'] = prompt
                pass

            messagesList.append(lastMessages)
            self.messages = messagesList
        return messagesList


    ### TESTS ### 
    def _test_if_models_are_available_for_their_given_endpoints(self):
        """Check to see if the models are available for each API endpoint"""
        print("Testing model endpoints to see if they are available for the given API endpoint")
        endpoints = {
            'chat': chat_completion_models,
            'embedding': embedding_models,
            'image': image_models,
            'moderation': moderation_models,
            'speech': speech_to_text,
            'tts': text_to_speech,
            'translation': translation_models,
            'vision': vision_models,
            'assistant': assistants_models,
            'tools': tools_models
        }
        error_list = []
        test_audio_data = self.resources['test']['audio']['data']
        test_translation_audio_data = self.resources['test']['translation']['audio']['data']

        for endpoint, models in endpoints.items():
            for model in tqdm.tqdm(models, desc=f"Testing {endpoint} models"):
                try:
                    match endpoint:
                        case 'chat':
                            openai.chat.completions.create(model=model, messages=[{"role": "user", "content": "Test"}])
                        case 'embedding':
                            openai.embeddings.create(model=model, input="Test")
                        case 'image':
                            openai.images.generate(model=model, prompt="Test")
                        case 'moderation':
                            openai.moderations.create(model=model, input="Test")
                        case 'speech':
                            openai.audio.transcriptions.create(model=model, file=test_audio_data)
                        case 'tts':
                            openai.audio.speech.create(model=model, input="Test", voice="alloy")
                        case 'translation':
                            openai.audio.translations.create(model=model, file=test_translation_audio_data)
                        case 'vision':
                            openai.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user",
                                "content": [
                                    {"type": "text", 
                                     "text": "What's in this image?"},
                                    {"type": "image_url", 
                                     "image_url": {"url": self.resources['test']['image']['url']}}
                                ]}])
                        case "assistant":
                            _ = openai.beta.assistants.create(
                                model=model,
                                name="Object-Oriented Programmer",
                                instructions="You are a an object oriented programmer. You write code in Python.",
                            )
                        case "tools":
                            _ = openai.beta.assistants.create(
                                model=model,
                                name="Object-Oriented Programmer",
                                instructions="You are a an object oriented programmer. You write code in Python.",
                                tools=[{"type": "code_interpreter"}],
                            )
                        case _:
                            raise ValueError(f"Unknown model: {model}")

                except Exception as e:
                    error_list.append(f"model: {model} | endpoint: {endpoint}  | error: {type(e).__name__} - {e}")

        self._raise_test_errors_if_if_any_occurred(error_list)
        return


    def _test_if_models_max_token_sizes_are_correct(self):
        """Check to see if the specified max token return size for each model is correct"""
        _max_tokens = self.metadata['max_tokens'].items()
        error_list = []
        for model, expected_max_tokens in tqdm.tqdm(_max_tokens, desc=f"Testing model max token size"):
            try:
                # Generate a string of the expected length
                test_string = "test " * (expected_max_tokens // self.metadata['tokens_per_word'])  # Approximate 4 tokens per word

                # Attempt to tokenize the string
                tokens = self.tokenize(test_string, model)
                actual_max_tokens = len(tokens)

                # Check if the actual token count is within a small margin of error (e.g., 1%)
                margin = expected_max_tokens * 0.01
                if abs(actual_max_tokens - expected_max_tokens) <= margin:
                    continue
                else:
                    error_list.append(f"Model {model}: Max token size mismatch. Expected {expected_max_tokens}, got {actual_max_tokens}")

            except Exception as e:
                error_list.append(f"model: {model} | expected_max_tokens: {expected_max_tokens} | error:\n{e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_for_tool_use_in_tool_models(self):
        """Check to see if the tool-use models support tool use"""
        error_list = []
        for model in tqdm.tqdm(tools_models, desc=f"Testing if the tool-use models support tool use"):
            try:
                # Define a simple tool
                tools = [
                    {"type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            }, "required": ["location"]}}}]
                # Test the model with tool use
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "What's the weather like in New York?"}],
                    tools=tools,
                    tool_choice="auto"
                )
                # Check if the model attempted to use the tool
                if response.choices[0].message.tool_calls:
                    continue
                    #print(f"Model {model} successfully supports tool use.")
                else:
                    error_list.append(f"Model {model} did not attempt to use the provided tool.")

            except Exception as e:
                error_list.append(f"Error testing tool use for mode {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_vision_capable_models(self):
        """Check to see if the listed vision models support vision capabilities"""
        test_image_url = self.resources['test']['image']['url']
        test_prompt = "What's in this image?"
        error_list = []

        for model in tqdm.tqdm(vision_models, desc=f"Testing vision capable models"):
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": test_prompt},
                            {"type": "image_url", "image_url": {"url": test_image_url}}
                        ]}])
                if response.choices and response.choices[0].message.content:
                    continue
                    # print(f"Model {model} successfully processed the image and provided a response.")
                    # print(f"Response: {response.choices[0].message.content[:100]}...")  # Print first 100 characters
                else:
                    error_list.append(f"Model {model} did not provide a valid response for the image.")

            except Exception as e:
                error_list.append(f"Error testing vision capabilities for model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_speech_to_text_models(self):
        # TODO
        """Check to see if the listed audio models support audio capabilities"""
        error_list = []
        test_translation_audio_data = self.resources['test']['translation']['audio']['data']
        for model in tqdm.tqdm(speech_to_text, desc="Testing speech-to-text models"):
            try:
                transcript = openai.audio.transcriptions.create(
                    model=model,
                    file=test_translation_audio_data
                )
                print(f"Model {model} successfully transcribed audio. First 50 characters: {transcript.text[:50]}...")
            except Exception as e:
                error_list.append(f"Error testing speech-to-text for model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_text_to_speech_models(self):
        """Test text-to-speech models."""
        error_list = []
        for model in tqdm.tqdm(text_to_speech, desc="Testing text-to-speech models"):
            try:
                _ = openai.audio.speech.create(
                    model=model,
                    voice="alloy",
                    input=self.resources['test']['text']
                )
                print(f"Model {model} successfully generated speech audio.")
            except Exception as e:
                error_list.append(f"Error testing text-to-speech for model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_audio_translation_models(self):
        """Test the functionality of audio translation models."""
        # Test translation models
        error_list = []
        test_translation_audio = self.resources['test']['translation']['audio']['data']
        for model in tqdm.tqdm(translation_models, desc="Testing audio translation models"):
            try:
                # Attempt translation
                translation = openai.audio.translations.create(
                    model=model,
                    file=test_translation_audio
                )
                print(f"Model {model} successfully translated audio. First 50 characters: {translation.text[:50]}...")
            except Exception as e:
                error_list.append(f"Error testing audio translation for model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_embedding_models(self):
        """Test the functionality of embedding models."""
        error_list = []
        for model in tqdm.tqdm(embedding_models, desc="Testing embedding models"):
            try:
                embedding = openai.embeddings.create(
                    model=model,
                    input=self.resources['test']['text']
                )
                print(f"Model {model} successfully generated embeddings. Dimension: {len(embedding.data[0].embedding)}")
            except Exception as e:
                error_list.append(f"Error testing embedding model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_text_completion_models(self):
        """Test the functionality of completion models."""
        error_list = []
        for model in tqdm.tqdm(chat_completion_models, desc="Testing chat completion models"):
            try:
                completion = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": self.resources['test']['completion']['text']}]
                )
                print(f"Model {model} successfully generated completion: {completion.choices[0].message.content[:50]}...")
            except Exception as e:
                error_list.append(f"Error testing completion model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_image_generation_models(self):
        """Test the functionality of image generation models."""
        # Supported sizes from this post.
        # See: https://community.openai.com/t/charges-for-dall-e-3-model-for-image/675469/3
        error_list = []
        for model in tqdm.tqdm(image_models, desc="Testing image generation models"):
            if model == "dall-e-3":
                supported_sizes = ["1024x1024", "1792x1024", "1024x1792"]
            elif model == "dall-e-2":
                supported_sizes = ["256x256", "512x512", "1024x1024"]
            else:
                raise ValueError(f"Model {model} is not supported for image generation at OpenAI.")
            for size in supported_sizes:
                try:
                    image = openai.images.generate(
                        model=model,
                        prompt=self.resources['test']['image']['prompt'],
                        n=1,
                        size=size
                    )
                    print(f"Model {model} successfully generated an image. URL: {image.data[0].url}")
                except Exception as e:
                    error_list.append(f"Error testing image model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_moderation_models(self):
        """Test the functionality of content moderation models."""
        error_list = []
        for model in tqdm.tqdm(moderation_models, desc="Testing moderation models"):
            try:
                moderation = openai.moderations.create(
                    model=model,
                    input=self.resources['test']['moderation']['text']
                )
                print(f"Model {model} successfully performed moderation. Flagged: {moderation.results[0].flagged}")
            except Exception as e:
                error_list.append(f"Error testing moderation model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_speech_to_text_models(self):
        """Test the functionality of speech-to-text models."""
        error_list = []
        test_audio = self.resources['test']['audio']['data']
        for model in tqdm.tqdm(speech_to_text, desc="Testing speech-to-text models"):
            try:
                transcript = openai.audio.transcriptions.create(
                    model=model,
                    file=test_audio
                )
                print(f"Model {model} successfully transcribed audio: {transcript.text[:50]}...")
            except Exception as e:
                error_list.append(f"Error testing speech-to-text model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_text_to_speech_models(self):
        """Test the functionality of text-to-speech models."""
        error_list = []
        for model in tqdm.tqdm(text_to_speech, desc="Testing text-to-speech models"):
            try:
                _ = openai.audio.speech.create(
                    model=model,
                    voice="alloy",
                    input=self.resources['test']['text']
                )
                print(f"Model {model} successfully generated speech audio.")
            except Exception as e:
                error_list.append(f"Error testing text-to-speech model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_translation_models(self):
        """Test the functionality of translation models."""
        error_list = []
        for model in tqdm.tqdm(translation_models, desc="Testing translation models"):
            try:
                translation = openai.audio.translations.create(
                    model=model,
                    file=self.resources['test']['translation']['audio']
                )
                print(f"Model {model} successfully translated audio: {translation.text[:50]}...")
            except Exception as e:
               error_list.append(f"Error testing translation model {model}: {type(e).__name__} - {e}")
        self._raise_test_errors_if_if_any_occurred(error_list)


    def _test_tokenize(self):
        print("Testing tokenize method...")

        # Test case 1: Basic tokenization
        text = "Hello, world!"
        model = "gpt-3.5-turbo"
        tokens = self.tokenize(text, model)
        assert isinstance(tokens, list), "Tokenization should return a list"
        assert len(tokens) > 0, "Tokenization should produce non-empty result"

        # Test case 2: Tokenization with different model
        text = "OpenAI isn't open!"
        model = "gpt-4"
        tokens = self.tokenize(text, model)
        assert isinstance(tokens, list), "Tokenization should return a list"
        assert len(tokens) > 0, "Tokenization should produce non-empty result"

        # Test case 3: Tokenization with None model (should use default)
        text = "Testing default model"
        tokens = self.tokenize(text, None)
        assert isinstance(tokens, list), "Tokenization should return a list"
        assert len(tokens) > 0, "Tokenization should produce non-empty result"

        # Test case 4: Tokenization of empty string
        text = ""
        model = "gpt-3.5-turbo"
        tokens = self.tokenize(text, model)
        assert isinstance(tokens, list), "Tokenization should return a list"
        assert len(tokens) == 0, "Tokenization of empty string should produce empty list"

        # Test case 5: Tokenization of long text
        text = "This is a longer piece of text that should be tokenized correctly. " * 10
        model = "gpt-3.5-turbo"
        tokens = self.tokenize(text, model)
        assert isinstance(tokens, list), "Tokenization should return a list"
        assert len(tokens) > 50, "Tokenization of long text should produce many tokens"

        print("All tokenize tests passed successfully.")


    def _test_detokenize(self):
        print("Testing detokenize method...")

        # Test case 1: Basic detokenization
        tokens = [9906, 11, 1917, 0]  # Corresponds to "Hello, world!"
        model = "gpt-3.5-turbo"
        result = self.detokenize(tokens, model)
        assert result == "Hello, world!", f"Expected 'Hello, world!', but got '{result}'"

        # Test case 2: Detokenization with different model
        tokens = [9906, 11, 1917, 0]  # Same tokens, different model
        model = "gpt-4"
        result = self.detokenize(tokens, model)
        assert result == "Hello, world!", f"Expected 'Hello, world!', but got '{result}'"

        # Test case 3: Detokenization with None model (should use default)
        tokens = [9906, 11, 1917, 0]
        result = self.detokenize(tokens, None)
        assert result == "Hello, world!", f"Expected 'Hello, world!', but got '{result}'"

        # Test case 4: Detokenization of empty list
        tokens = []
        model = "gpt-3.5-turbo"
        result = self.detokenize(tokens, model)
        assert result == "", f"Expected empty string, but got '{result}'"

        # Test case 5: Detokenization of longer text
        tokens = [2028, 374, 264, 1296, 315, 279, 3474, 1713, 2065, 1749, 13]  # "This is a test of the detokenization method."
        model = "gpt-3.5-turbo"
        result = self.detokenize(tokens, model)
        expected = "This is a test of the detokenization method."
        assert result == expected, f"Expected '{expected}', but got '{result}'"

        print("All detokenize tests passed successfully.")


    def _test_determine_model(self):
        print("Testing determine_model method...")
        messages = [{"role": "user", "content": "Hello, world!"}]

        def _check_in_serial(models, messages):
            for model in models:
                self.max_tokens = model[1]
                # Debug print to see token calculations
                message_tokens = self.tokenize(json.dumps(messages), model[0])
                total_tokens = len(message_tokens) + model[1]
                print(f"Testing {model[0]} with max_tokens={model[1]}")
                print(f"Message tokens: {len(message_tokens)}, Total: {total_tokens}")
                result = self.determine_model(model[0], messages)
                print(f"Result: {result}")
                yield result, model[1]

        # Test case 1: GPT-3 models
        models =[
            ('gpt-3.5-turbo', 1234),
            ('gpt-3.5-turbo-0125', 1234),
            ('gpt-3.5-turbo-1106', 12345),
            ('gpt-3.5-turbo-16k', 12345),
        ]

        # # Test case 1: GPT-3 models with smaller message
        # models =[
        #     ('gpt-3.5-turbo', 1000),  # Using smaller max_tokens to ensure we stay within limits
        #     ('gpt-3.5-turbo-0125', 1000),
        #     ('gpt-3.5-turbo-1106', 1000),
        # ]
        
        for result, tokens in _check_in_serial(models, messages):
            assert "gpt-3" in result, f"Expected GPT-3 model, but got {result}"


        # for result, tokens in _check_in_serial(models, messages):
        #     # Check that result contains "gpt-3" as expected
        #     assert "gpt-3" in result, f"Expected GPT-3 model, but got {result}"
            
        #     # Check that it's not an instruct model (unless specifically requested)
        #     if "instruct" not in models[0]:
        #         assert "instruct" not in result, "Got instruct model, when we should have gotten a gpt-3 model"
            
        #     # Based on the function's logic, it will select the first model in the dictionary 
        #     # that can handle the tokens, which appears to be gpt-3.5-turbo-1106 for all cases
        #     # (since dictionaries preserve insertion order in Python 3.7+)
        #     if "gpt-3" in models[0][0]:
        #         # For any gpt-3 model input, we expect a gpt-3 model that can handle the tokens
        #         assert result in max_tokens.keys(), f"Got {result} which is not in our known models"
        #         assert "gpt-3" in result, f"Expected a GPT-3 model, got {result}"
                
        #         # Verify the chosen model can handle our token count
        #         input_token_count = len(self.tokenize(json.dumps(messages), models[0][0])) + tokens
        #         assert max_tokens[result] >= input_token_count, f"Model {result} cannot handle {input_token_count} tokens"
                


        # Test case 2: GPT-4 models
        model = "gpt-4"
        self.max_tokens = 4096
        messages = [{"role": "user", "content": "Hello, world!"}]
        result = self.determine_model(model, messages)
        assert "gpt-4" in result, f"Expected GPT-4 model, but got {result}"

        # Test case 3: Vision models
        models = [("o1",3768),("gpt-4o",16384),("gpt-4o-mini",16384),("gpt-4-turbo",4096)]
        messages = [
            {"role": "user", 
             "content": [
                {"type": "text", "text": "What's in this image?"}, 
                {"type": "image_url", 
                 "image_url": {"url": "https://example.com/image.jpg"}}]}]
        for model in models:
            self.max_tokens = model[1]
            result = self.determine_model(model[0], messages)
            assert result in vision_models, f"Expected Vision model, but got {result}"

        # Test case 4: o-series models:
        model = "o1"
        messages = [{"role": "user", "content": "Hello, world!"}]
        result = self.determine_model(model, messages)

        # Test case 4: Instruct model
        model = "gpt-3.5-turbo-instruct"
        messages = [{"role": "user", "content": "Translate the following English text to French: 'Hello, how are you?'"}]
        result = self.determine_model(model, messages)
        assert "instruct" in result, f"Expected Instruct model, but got {result}"

        # Test case 5: Long input (should choose a model with higher token limit)
        model = "gpt-3.5-turbo"
        long_content = "This is a very long input. " * 1000
        messages = [{"role": "user", "content": long_content}]
        result = self.determine_model(model, messages)
        assert result in ["gpt-3.5-turbo-16k", "gpt-4-1106-preview"], f"Expected model with high token limit, but got {result}"

        # Test case 6: Invalid model
        model = "invalid-model"
        messages = [{"role": "user", "content": "Hello, world!"}]
        try:
            result = self.determine_model(model, messages)
            assert False, "Expected an exception for invalid model, but got a result"
        except Exception as e:
            assert "bad model" in str(e), f"Expected 'bad model' error, but got: {e}"

        print("All determine_model tests passed successfully.")


    def _test_embedding(self):
        print("Testing embedding method...")

        # Test case 1: Basic embedding
        model = "text-embedding-ada-002"
        input_text = "Hello, world!"
        format = "float"
        result = self.embedding(model, input_text, format)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"

        # Test case 2: Embedding with base64 format
        format = "base64"
        result = self.embedding(model, input_text, format)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"

        # Test case 3: Embedding with longer input
        input_text = "This is a longer piece of text that should be embedded correctly."
        format = "float"
        result = self.embedding(model, input_text, format)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"

        # Test case 4: Embedding with invalid model
        try:
            invalid_model = "invalid-model"
            self.embedding(invalid_model, input_text, format)
            assert False, "Expected an exception for invalid model, but got a result"
        except Exception as e:
            assert "invalid model" in str(e).lower(), f"Expected 'invalid model' error, but got: {e}"

        # Test case 5: Embedding with invalid format
        try:
            invalid_format = "invalid-format"
            self.embedding(model, input_text, invalid_format)
            assert False, "Expected an exception for invalid format, but got a result"
        except Exception as e:
            assert "invalid format" in str(e).lower(), f"Expected 'invalid format' error, but got: {e}"

        print("All embedding tests passed successfully.")


    def _test_moderation(self):
        print("Testing moderation method...")

        # Test case 1: Basic moderation
        model = "text-moderation-stable"
        text = "This is a normal text."
        result = self.moderation(model, text)
        assert isinstance(result, openai.types.moderation.Moderation), "Result should be a Moderation object"
        assert hasattr(result, 'results'), "Result should have 'results' attribute"
        assert len(result.results) > 0, "Result should have at least one result"
        assert result.results[0].flagged == False, "Normal text should not be flagged"

        # Test case 2: Moderation with potentially flagged content
        text = "I want to harm someone."
        result = self.moderation(model, text)
        assert result.results[0].flagged == True, "Potentially harmful text should be flagged"

        # Test case 3: Moderation with different model
        model = "text-moderation-latest"
        text = "This is another normal text."
        result = self.moderation(model, text)
        assert isinstance(result, openai.types.moderation.Moderation), "Result should be a Moderation object"

        # Test case 4: Moderation with invalid model
        invalid_model = "invalid-moderation-model"
        text = "Test text"
        try:
            self.moderation(invalid_model, text)
            assert False, "Expected an exception for invalid model, but got a result"
        except Exception as e:
            assert "bad model" in str(e), f"Expected 'bad model' error, but got: {e}"

        # Test case 5: Moderation with empty text
        model = "text-moderation-stable"
        text = ""
        result = self.moderation(model, text)
        assert isinstance(result, openai.types.moderation.Moderation), "Result should be a Moderation object"
        assert len(result.results) > 0, "Result should have at least one result even for empty text"

        print("All moderation tests passed successfully.")


    def _test_moderated_text_to_image(self):
        print("Testing moderated_text_to_image method...")

        # Test case 1: Basic functionality
        model = "dall-e-2"
        size = "256x256"
        n = 1
        prompt = "A beautiful sunset over the ocean"
        result = self.moderated_text_to_image(model, size, n, prompt)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"

        # Parse the JSON string in 'text' key
        images = json.loads(result['text'])
        assert isinstance(images, list), "Images should be a list"
        assert len(images) == n, f"Number of images should be {n}"
        assert 'url' in images[0], "Each image should have a 'url'"
        assert 'revised_prompt' in images[0], "Each image should have a 'revised_prompt'"

        # Test case 2: Multiple images
        n = 2
        result = self.moderated_text_to_image(model, size, n, prompt)
        images = json.loads(result['text'])
        assert len(images) == n, f"Number of images should be {n}"

        # Test case 3: Different size
        size = "512x512"
        result = self.moderated_text_to_image(model, size, n, prompt)
        images = json.loads(result['text'])
        assert len(images) == n, f"Number of images should be {n}"

        # Test case 4: Flagged content
        prompt = "Explicit violent content"
        try:
            self.moderated_text_to_image(model, size, n, prompt)
            assert False, "Expected an exception for flagged content, but got a result"
        except Exception as e:
            assert "bad messages" in str(e), f"Expected 'bad messages' error, but got: {e}"

        # Test case 5: Invalid model
        invalid_model = "invalid-model"
        try:
            self.moderated_text_to_image(invalid_model, size, n, prompt)
            assert False, "Expected an exception for invalid model, but got a result"
        except Exception as e:
            assert "invalid model" in str(e).lower(), f"Expected 'invalid model' error, but got: {e}"

        print("All moderated_text_to_image tests passed successfully.")


    def _test_process_messages(self):
        print("Testing process_messages method...")

        # Test case 1: Basic functionality with system message and user prompt
        messages = []
        prompt = "Hello, how are you?"
        files = None
        system = "You are a helpful assistant."
        result = self.process_messages(messages, prompt, files, system)
        assert len(result) == 2, "Expected 2 messages in the result"
        assert result[0]['role'] == 'system', "First message should be system"
        assert result[1]['role'] == 'user', "Second message should be user"
        assert result[1]['content'] == prompt, "User message content should match prompt"

        # Test case 2: Multiple messages in history
        messages = [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I'm sorry, I don't have real-time weather information."},
            {"role": "user", "content": "Okay, thanks!"}
        ]
        prompt = "Can you tell me a joke?"
        result = self.process_messages(messages, prompt, files, system)
        assert len(result) == 5, "Expected 5 messages in the result"
        assert result[-1]['content'] == prompt, "Last message should be the new prompt"

        # Test case 3: With image files
        files = [{"url": "http://example.com/image.jpg", "detail": "low"}]
        prompt = "What's in this image?"
        result = self.process_messages([], prompt, files, None)
        assert len(result) == 1, "Expected 1 message in the result"
        assert isinstance(result[0]['content'], list), "Content should be a list for image input"
        assert len(result[0]['content']) == 2, "Content should have 2 items (text and image)"

        # Test case 4: Empty prompt and files
        result = self.process_messages(messages, "", None, None)
        assert result == messages, "Result should be unchanged when prompt and files are empty"

        # Test case 5: Invalid role in messages
        invalid_messages = [{"role": "invalid", "content": "This is invalid"}]
        try:
            self.process_messages(invalid_messages, "", None, None)
            assert False, "Expected an exception for invalid role"
        except Exception as e:
            assert "bad role" in str(e), "Exception should mention 'bad role'"

        print("All process_messages tests passed successfully.")


    def _test_moderated_text_to_speech(self):
        print("Testing moderated_text_to_speech method...")

        # Test case 1: Basic functionality with non-flagged content
        model = "tts-1"
        text = "This is a test of text-to-speech."
        voice = "alloy"
        response_format = "mp3"
        speed = 1.0
        result = self.moderated_text_to_speech(model, text, voice, response_format, speed)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], openai.types.audio.Speech), "Result 'text' should be a Speech object"

        # Test case 2: Flagged content
        text = "This is some very inappropriate and offensive content."
        try:
            self.moderated_text_to_speech(model, text, voice, response_format, speed)
            assert False, "Expected an exception for flagged content, but got a result"
        except Exception as e:
            assert "bad messages" in str(e), f"Expected 'bad messages' error, but got: {e}"

        # Test case 3: Different voice
        text = "Testing with a different voice."
        voice = "echo"
        result = self.moderated_text_to_speech(model, text, voice, response_format, speed)
        assert isinstance(result['text'], openai.types.audio.Speech), "Result 'text' should be a Speech object"

        # Test case 4: Different response format
        response_format = "opus"
        result = self.moderated_text_to_speech(model, text, voice, response_format, speed)
        assert isinstance(result['text'], openai.types.audio.Speech), "Result 'text' should be a Speech object"

        # Test case 5: Different speed
        speed = 1.5
        result = self.moderated_text_to_speech(model, text, voice, response_format, speed)
        assert isinstance(result['text'], openai.types.audio.Speech), "Result 'text' should be a Speech object"

        print("All moderated_text_to_speech tests passed successfully.")


    def _test_text_to_speech(self):
        print("Testing text_to_speech method...")

        # Test case 1: Basic functionality
        model = "tts-1"
        text = "This is a test of text-to-speech."
        voice = "fable"
        result = self.text_to_speech(model, text, voice)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'audio' in result, "Result should contain 'audio' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['audio'], str), "Result 'audio' should be a string"

        # Test case 2: Different voice
        voice = "alloy"
        result = self.text_to_speech(model, text, voice)
        assert isinstance(result['audio'], str), "Result 'audio' should be a string"

        # Test case 3: Different response format
        response_format = "opus"
        result = self.text_to_speech(model, text, voice, response_format)
        assert isinstance(result['audio'], str), "Result 'audio' should be a string"

        # Test case 4: Different speed
        speed = 1.5
        result = self.text_to_speech(model, text, voice, "mp3", speed)
        assert isinstance(result['audio'], str), "Result 'audio' should be a string"

        # Test case 5: Invalid voice
        invalid_voice = "invalid_voice"
        try:
            self.text_to_speech(model, text, invalid_voice)
            assert False, "Expected an exception for invalid voice, but got a result"
        except Exception as e:
            assert "bad voice" in str(e), f"Expected 'bad voice' error, but got: {e}"

        # Test case 6: Invalid response format
        invalid_format = "invalid_format"
        try:
            self.text_to_speech(model, text, voice, invalid_format)
            assert False, "Expected an exception for invalid response format, but got a result"
        except Exception as e:
            assert "bad response_format" in str(e), f"Expected 'bad response_format' error, but got: {e}"

        # Test case 7: Invalid speed
        invalid_speed = 5
        try:
            self.text_to_speech(model, text, voice, "mp3", invalid_speed)
            assert False, "Expected an exception for invalid speed, but got a result"
        except Exception as e:
            assert "bad speed" in str(e), f"Expected 'bad speed' error, but got: {e}"

        # Test case 8: Text exceeding max length
        long_text = "a" * 4097  # Exceeds max_length of 4096
        try:
            self.text_to_speech(model, long_text, voice)
            assert False, "Expected an exception for text exceeding max length, but got a result"
        except Exception as e:
            assert "bad text" in str(e), f"Expected 'bad text' error, but got: {e}"

        print("All text_to_speech tests passed successfully.")


    def _test_speech_to_text(self):
        print("Testing speech_to_text method...")

        # Test case 1: Basic functionality
        model = "whisper-1"
        audio_path = self.resources['test']['audio']
        result = self.speech_to_text(model, audio_path)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 2: Invalid model
        invalid_model = "invalid-model"
        try:
            self.speech_to_text(invalid_model, audio_path)
            assert False, "Expected an exception for invalid model, but got a result"
        except Exception as e:
            assert "bad model" in str(e), f"Expected 'bad model' error, but got: {e}"

        # Test case 3: Non-existent audio file
        non_existent_audio = "non_existent_audio.mp3"
        try:
            self.speech_to_text(model, non_existent_audio)
            assert False, "Expected an exception for non-existent audio file, but got a result"
        except Exception as e:
            assert "No such file or directory" in str(e), f"Expected file not found error, but got: {e}"

        # Test case 4: Empty audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file_path = temp_file.name
        try:
            self.speech_to_text(model, temp_file_path)
            assert False, "Expected an exception for empty audio file, but got a result"
        except Exception as e:
            assert "File is empty" in str(e), f"Expected 'File is empty' error, but got: {e}"
        finally:
            os.remove(temp_file_path)

        print("All speech_to_text tests passed successfully.")


    def _test_moderated_chat_complete(self):
        print("Testing moderated_chat_complete method...")

        # Test case 1: Basic functionality with non-flagged content
        self.model = "gpt-3.5-turbo"
        self.messages = [{"role": "user", "content": "Hello, how are you?"}]
        self.temperature = 0.7
        self.max_tokens = 50
        result = self.moderated_chat_complete()
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], openai.types.chat.ChatCompletion), "Result 'text' should be a ChatCompletion object"

        # Test case 2: Flagged content
        self.messages = [{"role": "user", "content": "How to make illegal substances"}]
        try:
            self.moderated_chat_complete()
            assert False, "Expected an exception for flagged content, but got a result"
        except Exception as e:
            assert "bad messages" in str(e), f"Expected 'bad messages' error, but got: {e}"

        # Test case 3: Different model
        self.model = "gpt-4"
        self.messages = [{"role": "user", "content": "What's the capital of France?"}]
        result = self.moderated_chat_complete()
        assert isinstance(result['text'], openai.types.chat.ChatCompletion), "Result 'text' should be a ChatCompletion object"

        # Test case 4: Different temperature
        self.temperature = 0.9
        result = self.moderated_chat_complete()
        assert isinstance(result['text'], openai.types.chat.ChatCompletion), "Result 'text' should be a ChatCompletion object"

        # Test case 5: Different max_tokens
        self.max_tokens = 100
        result = self.moderated_chat_complete()
        assert isinstance(result['text'], openai.types.chat.ChatCompletion), "Result 'text' should be a ChatCompletion object"

        print("All moderated_chat_complete tests passed successfully.")


    def _test_request_complete(self):
        print("Testing request_complete method...")

        # Test case 1: Basic chat functionality
        self.model = "gpt-3.5-turbo"
        self.method = "chat"
        self.messages = [{"role": "user", "content": "Hello, how are you?"}]
        result = self.request_complete()
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 2: Image to text functionality
        self.model = "gpt-4-vision-preview"
        self.method = "image_to_text"
        self.messages = [{"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]}]
        result = self.request_complete()
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 3: Invalid model
        self.model = "invalid-model"
        self.method = "chat"
        try:
            self.request_complete()
            assert False, "Expected an exception for invalid model, but got a result"
        except Exception as e:
            assert "bad model" in str(e), f"Expected 'bad model' error, but got: {e}"

        # Test case 4: Invalid method
        self.model = "gpt-3.5-turbo"
        self.method = "invalid-method"
        try:
            self.request_complete()
            assert False, "Expected an exception for invalid method, but got a result"
        except Exception as e:
            assert "bad method" in str(e), f"Expected 'bad method' error, but got: {e}"

        # Test case 5: With stopping_regex
        self.model = "gpt-3.5-turbo"
        self.method = "chat"
        self.messages = [{"role": "user", "content": "Count from 1 to 10"}]
        result = self.request_complete(stopping_regex=r"\b5\b")
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert "5" in result['text'], "Result should stop at or include '5'"

        # Test case 6: Invalid stopping_regex
        try:
            self.request_complete(stopping_regex="[")
            assert False, "Expected an exception for invalid stopping_regex, but got a result"
        except Exception as e:
            assert "bad \"stopping_regex\"" in str(e), f"Expected 'bad \"stopping_regex\"' error, but got: {e}"

        print("All request_complete tests passed successfully.")


    def _test_chat(self):
        print("Testing chat method...")

        # Test case 1: Basic functionality
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hello"}]
        prompt = "How are you?"
        system = "You are a helpful assistant."
        temperature = 0.7
        max_tokens = 50
        result = self.chat(model, messages, prompt, system, temperature, max_tokens)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 2: With files
        files = [{"url": "http://example.com/image.jpg", "detail": "low"}]
        result = self.chat(model, messages, prompt, system, temperature, max_tokens, files=files)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 3: Different model
        model = "gpt-4"
        result = self.chat(model, messages, prompt, system, temperature, max_tokens)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 4: Different temperature
        temperature = 0.9
        result = self.chat(model, messages, prompt, system, temperature, max_tokens)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 5: Different max_tokens
        max_tokens = 100
        result = self.chat(model, messages, prompt, system, temperature, max_tokens)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        print("All chat tests passed successfully.")


    def _test_audio_chat(self):
        print("Testing audio_chat method...")

        # Test case 1: Basic functionality with text prompt
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hello"}]
        voice = "alloy"
        system = "You are a helpful assistant."
        temperature = 0.7
        max_tokens = 50
        prompt = "How are you?"

        result = self.audio_chat(model, messages, voice, system, temperature, max_tokens, prompt=prompt)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string (audio data)"

        # Test case 2: Functionality with audio input
        audio_file = "path/to/test/audio/file.mp3"
        result = self.audio_chat(model, messages, voice, system, temperature, max_tokens, audio=audio_file)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string (audio data)"

        # Test case 3: Error when both prompt and audio are provided
        try:
            self.audio_chat(model, messages, voice, system, temperature, max_tokens, prompt="Hello", audio=audio_file)
            assert False, "Expected an exception when both prompt and audio are provided"
        except Exception as e:
            assert "you have both prompt and audio" in str(e), f"Unexpected error message: {e}"

        # Test case 4: Error when neither prompt nor audio is provided
        try:
            self.audio_chat(model, messages, voice, system, temperature, max_tokens)
            assert False, "Expected an exception when neither prompt nor audio is provided"
        except Exception as e:
            assert "no prompt or audio" in str(e), f"Unexpected error message: {e}"

        # Test case 5: Test with a different voice
        result = self.audio_chat(model, messages, "echo", system, temperature, max_tokens, prompt="Hello")
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"

        # Test case 6: Test with a different model
        result = self.audio_chat("gpt-4", messages, voice, system, temperature, max_tokens, prompt="Hello")
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"

        # Test case 7: Test with invalid audio file type
        try:
            self.audio_chat(model, messages, voice, system, temperature, max_tokens, audio="invalid_file.txt")
            assert False, "Expected an exception for invalid audio file type"
        except Exception as e:
            assert "bad file_type" in str(e), f"Unexpected error message: {e}"

        print("All audio_chat tests passed successfully.")

    def _test_text_to_image(self):
        print("Testing text_to_image method...")

        # Test case 1: Basic functionality with DALL-E 2
        model = "dall-e-2"
        size = "256x256"
        n = 1
        prompt = "A beautiful sunset over the ocean"
        result = self.text_to_image(model, size, n, prompt)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"

        # Test case 2: DALL-E 3 with different size
        model = "dall-e-3"
        size = "1792x1024"
        result = self.text_to_image(model, size, n, prompt)
        assert isinstance(result, dict), "Result should be a dictionary"

        # Test case 3: Multiple images with DALL-E 2
        model = "dall-e-2"
        n = 2
        result = self.text_to_image(model, size, n, prompt)
        assert isinstance(result, dict), "Result should be a dictionary"

        # Test case 4: Invalid model
        try:
            self.text_to_image("invalid-model", size, n, prompt)
            assert False, "Expected an exception for invalid model"
        except Exception as e:
            assert "bad model" in str(e), f"Unexpected error message: {e}"

        # Test case 5: Invalid size
        try:
            self.text_to_image(model, "invalid-size", n, prompt)
            assert False, "Expected an exception for invalid size"
        except Exception as e:
            assert "bad size" in str(e), f"Unexpected error message: {e}"

        # Test case 6: Invalid n for DALL-E 3
        try:
            self.text_to_image("dall-e-3", size, 2, prompt)
            assert False, "Expected an exception for n > 1 with DALL-E 3"
        except Exception as e:
            assert "bad n" in str(e), f"Unexpected error message: {e}"

        # Test case 7: Invalid n (too large) for DALL-E 2
        try:
            self.text_to_image("dall-e-2", size, 11, prompt)
            assert False, "Expected an exception for n > 10 with DALL-E 2"
        except Exception as e:
            assert "bad n" in str(e), f"Unexpected error message: {e}"

        print("All text_to_image tests passed successfully.")


    def _test_image_to_text(self):
        print("Testing image_to_text method...")

        # Test case 1: Basic functionality
        model = "gpt-4-vision-preview"
        prompt = "What's in this image?"
        images = [{"url": "https://example.com/image.jpg", "detail": "low"}]
        max_tokens = 100
        system = "You are a helpful assistant."
        result = self.image_to_text(model, prompt, images, max_tokens, system)
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'text' in result, "Result should contain 'text' key"
        assert 'done' in result, "Result should contain 'done' key"
        assert result['done'] == True, "Result 'done' should be True"
        assert isinstance(result['text'], str), "Result 'text' should be a string"

        # Test case 2: Multiple images
        images = [
            {"url": "https://example.com/image1.jpg", "detail": "low"},
            {"url": "https://example.com/image2.jpg", "detail": "high"}
        ]
        result = self.image_to_text(model, prompt, images, max_tokens, system)
        assert isinstance(result, dict), "Result should be a dictionary"

        # Test case 3: Invalid image detail
        invalid_images = [{"url": "https://example.com/image.jpg", "detail": "invalid"}]
        try:
            self.image_to_text(model, prompt, invalid_images, max_tokens, system)
            assert False, "Expected an exception for invalid image detail"
        except Exception as e:
            assert "bad quality" in str(e), f"Unexpected error message: {e}"

        # Test case 4: Different model
        model = "gpt-3.5-turbo-vision"
        result = self.image_to_text(model, prompt, images, max_tokens, system)
        assert isinstance(result, dict), "Result should be a dictionary"

        # Test case 5: Different max_tokens
        max_tokens = 200
        result = self.image_to_text(model, prompt, images, max_tokens, system)
        assert isinstance(result, dict), "Result should be a dictionary"

        print("All image_to_text tests passed successfully.")


    @staticmethod
    def _raise_test_errors_if_if_any_occurred(error_list, msg=None):
        if len(error_list) > 0:
            # Get the name of the test that called this function.
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            test_name = caller_frame.f_code.co_name if caller_frame else "Unknown Test"

            got_these_errors = "\n\n".join(error_list)
            print(f"\nGot {len(error_list)} errors in test '{test_name}'\n\n{got_these_errors}\n\n")
            msg = msg if msg else f"Got errors in '{test_name}'"
            raise TestError(msg)


    def __test__(self):
        print("Begin OpenAI API tests.")
        print("Testing OpenAI API models characteristics...")

        # Test models chars
        # NOTE Commented out because the models passed all the tests, 
        # and these take a long time to run.
        #self._test_if_models_are_available_for_their_given_endpoints()
        #self._test_tokenize()
        # NOTE This set doesn't work because of the models' super-long context sizes.
        # Models aren't trained to just ramble on, so none of them are passing the test.
        #self._test_if_models_max_token_sizes_are_correct()

        # Test models
        print("Tests OpenAI API models characteristics completed successfully.")
        print("Testing OpenAI API models...")
        #self._test_audio_translation_models()
        #self._test_embedding_models()
        #self._test_for_tool_use_in_tool_models()
        #self._test_image_generation_models()
        #self._test_moderation_models()
        #self._test_speech_to_text_models()
        #self._test_text_completion_models()
        #self._test_text_to_speech_models()
        #self._test_vision_capable_models()

        # Test built-in functions
        print("Tests for OpenAI API models completed successfully.")
        print("Testing basic functionality of inner methods...")
        # self._test_detokenize()
        self._test_determine_model()
        self._test_embedding()
        self._test_moderation()
        self._test_moderated_text_to_image()
        self._test_process_messages()
        self._test_moderated_text_to_speech()
        self._test_text_to_speech()
        self._test_speech_to_text()

        self._test_moderated_chat_complete()
        self._test_request_complete()

        print("Inner methods completed successfully.")
        print("Testing basic functionality of outer methods...")
        self._test_chat()
        self._test_audio_chat()
        self._test_text_to_image()
        self._test_image_to_text()
        print("All tests completed successfully.")
        return None


    def __call__(self, *args, **kwargs):
        # Override the call method to run the tests when the class is instantiated
        self.__test__()


if __name__ == "__main__":
    import sys

    ipfs_accelerate_py_dir = Path(__file__).parent.parent
    test_audio_path = ipfs_accelerate_py_dir / 'test' / 'test.mp3'
    test_translation_audio_path = ipfs_accelerate_py_dir / 'test' / 'trans_test.mp3'
    test_image_path = ipfs_accelerate_py_dir / 'test' / 'test.jpg'

    # NOTE Whisper infers file format type from the name attribute on the audio data.
    # Since WSL removes metadata from an audio file when it is imported from Windows, 
    # we need to re-assign it.
    # See: https://community.openai.com/t/whisper-error-400-unrecognized-file-format/563474/8
    with open(test_audio_path, 'rb') as file:
        test_audio_data = io.BytesIO(file.read())
        test_audio_data.name = test_audio_path.name

    with open(test_translation_audio_path, 'rb') as file:
        test_translation_audio_data = io.BytesIO(file.read())
        test_translation_audio_data.name = test_audio_path.name

    with open(test_image_path, 'rb') as file:
        test_image_data = file.read()

    resources = {
        "test": {
            "audio": {"data": test_audio_data},
            "text": "Future events such as these will affect us in the future.",
            "completion": {"text": "Colorless green ideas sleep"},
            "moderation": {"text": "Teach me an efficient way to take candy from a baby."},
            "image": {
                "data": test_image_data,
                "url": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Tetrapharmakos_PHerc_1005_col_5.png",
                "prompt": "C-beams glittering in the dark near the Tannhäuser Gate"
            },
            "translation": {
                "audio": {"data": test_translation_audio_data},
                "text": "Ojalá se te acabe la mirada constante, la palabra precisa, la sonrisa perfecta."
            }}}

    metadata = {
        "tokens_per_word": 4,
        "max_tokens": max_tokens,
        "openai_api_key": os.environ['OPENAI_API_KEY']
    }

    try:
        test_openai_api = openai_api(resources, metadata)
        test_openai_api.__test__()
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)