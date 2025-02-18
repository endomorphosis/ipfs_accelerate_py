import time
import re
import os
import openai
# from cloudkit_worker import dispatch_result
import tiktoken
import tempfile
import base64
import requests
import tempfile
import json
import subprocess
from datetime import datetime

# Deprecations page
# https://platform.openai.com/docs/deprecations

class TestError(Exception):
    pass

assistants_models = [ # /v1/assistants
    "o1", # 100,000
    "o1-mini", # 65,536
    "o1-preview", # 32,768
    "o3-mini", # 100,000
    "gpt-4o", # 16,384
    "gpt-4o-audio-preview", # 16,384
    "gpt-4o-realtime-preview", # 4,096
    "gpt-4o-mini", # 16,384
    "gpt-4o-mini-audio-preview", # 16,384
    "gpt-4o-mini-realtime-preview", # 4,096
    "gpt-4", # 8192
    "gpt-4-0613", # 8,192
    "gpt-4-0125-preview", # 4,096
    "gpt-4-1106-preview", # 128000
    "gpt-3.5-turbo", # 4096
    "gpt-3.5-turbo-0125", # 4,096
    "gpt-3.5-turbo-1106", # 16385
    "gpt-3.5-turbo-16k", # 16385
    "gpt-3.5-turbo-instruct", # 4096
    "gpt-3.5-turbo-instruct-0914" # 4096
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

chat_completion_models = [ # /v1/chat/completions
    "o1", # 100,000
    "o1-mini", # 100,000
    "o1-preview", # 32,768
    "o3-mini",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "gpt-4o-mini", # 16,384
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-realtime-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-instruct",
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
    "tts-1-1106"
    "tts-1-hd",
    "tts-1-hd-1106"
]

translation_models = [ # /v1/audio/translations
    "whisper-1"
]

vision_models = [ # https://platform.openai.com/docs/guides/vision
    "o1",
    "gpt-4-turbo"
    "gpt-4o",
    "gpt-4o-mini",
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
    def __init__(self, resources=None, metadata=None):
        self.prompt = None
        self.messages = None
        self.instruct = None
        self.input = None
        self.method = None
        self.temperature = None
        self.api_key = None
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
        moderation = openai.moderations.create(input=text)
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
        self.method = 'embedding',
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
        return self.request_complete( **kwargs)



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

    def determine_model(self, model, messages):
        model_type = ""
        this_max_tokens = self.max_tokens

        if "gpt-4" in model:
            model_type = "gpt-4"
        elif "gpt-3" in model:
            model_type = "gpt-3"

        if "instruct" in model:
            model_type = "instruct"

        if "vision" in model:
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
            'gpt-3.5-turbo-instruct-0914': 4096,
            'gpt-4-turbo-preview': 4096,
            'chatgpt-4o-latest': 16384
        }
        stringifed_messages = ""
        stringified_messages = json.dumps(messages)
        if "image_url" in stringified_messages:
            model_type = "vision"
            pass
        message_tokens = self.tokenize(stringifed_messages, model)
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
                    if "vision" in model:
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
                pass

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


    def _test_model_endpoints(self):
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
            'vision': vision_models
        }
        for endpoint, models in endpoints.items():
            for model in models:
                try:
                    if endpoint == 'chat':
                        openai.chat.completions.create(model=model, messages=[{"role": "user", "content": "Test"}])
                    elif endpoint == 'embedding':
                        openai.embeddings.create(model=model, input="Test")
                    elif endpoint == 'image':
                        openai.images.generate(model=model, prompt="Test")
                    elif endpoint == 'moderation':
                        openai.moderations.create(model=model, input="Test")
                    elif endpoint == 'speech':
                        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
                            openai.audio.transcriptions.create(model=model, file=temp_file.name)
                    elif endpoint == 'tts':
                        openai.audio.speech.create(model=model, input="Test", voice="alloy")
                    elif endpoint == 'translation':
                        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
                            openai.audio.translations.create(model=model, file=temp_file.name)
                    elif endpoint == 'vision':
                        openai.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "user",
                                "content": [
                                    {"type": "text", "text": "What's in this image?"},
                                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                                ]}])
                    print(f"Model {model} is available for {endpoint} endpoint")
                except Exception as e:
                    raise TestError(f"Error testing model {model} for {endpoint} endpoint: {e}") from e
        return


    def _test_token_size(self):
        """Check to see if the max token return size for each model is correct"""
        print("Testing max token size for each model:")
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
            'gpt-3.5-turbo-instruct-0914': 4096,
            'gpt-4-turbo-preview': 4096,
            'chatgpt-4o-latest': 16384
        }
        for model, expected_max_tokens in max_tokens.items():
            try:
                # Generate a string of the expected length
                test_string = "test " * (expected_max_tokens // 5)  # Approximate 5 tokens per word

                # Attempt to tokenize the string
                tokens = self.tokenize(test_string, model)
                actual_max_tokens = len(tokens)

                # Check if the actual token count is within a small margin of error (e.g., 1%)
                margin = expected_max_tokens * 0.01
                if abs(actual_max_tokens - expected_max_tokens) <= margin:
                    print(f"Model {model}: Max token size correct ({actual_max_tokens})")
                else:
                    print(f"Model {model}: Max token size mismatch. Expected {expected_max_tokens}, got {actual_max_tokens}")

            except Exception as e:
                raise TestError(f"Error testing max token size for model {model}: {e}") from e
        return


    def _test_for_tool_model_tool_use(self):
        """Check to see if the tool-use models support tool use"""

        print("Testing tool use models:")
        for model in tools_models:
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
                                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"]}
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
                    print(f"Model {model} successfully supports tool use.")
                else:
                    print(f"Model {model} did not attempt to use the provided tool.")

            except Exception as e:
                raise TestError(f"Error testing tool use for mode {model}: {e}") from e
        return


    def _test_models_with_vision_capabilities(self):
        """Check to see if the listed vision models support vision capabilities"""
        test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        test_prompt = "What's in this image?"

        print("Testing vision models:")
        for model in vision_models:
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
                    print(f"Model {model} successfully processed the image and provided a response.")
                    print(f"Response: {response.choices[0].message.content[:100]}...")  # Print first 100 characters
                else:
                    raise ValueError(f"Model {model} did not provide a valid response for the image.")

            except Exception as e:
                raise TestError(f"Error testing vision capabilities for model {model}: {e}") from e
        return

    def _test_text_to_speech_models(self):
        """Check to see if the listed audio models support audio capabilities"""
        test_audio_url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
        
        # Test speech-to-text models
        print("Testing speech-to-text models:")
        for model in speech_to_text:
            try:
                with tempfile.NamedTemporaryFile(suffix=".flac") as temp_file:
                    # Download the test audio file
                    subprocess.run(["wget", "-O", temp_file.name, test_audio_url], check=True)

                    # Attempt transcription
                    transcript = openai.audio.transcriptions.create(
                        model=model,
                        file=temp_file.name
                    )
                    print(f"Model {model} successfully transcribed audio. First 50 characters: {transcript.text[:50]}...")
            except Exception as e:
                raise TestError(f"Error testing speech-to-text for model {model}: {e}") from e

    def _test_text_to_speech_models(self):
        """Test text-to-speech models."""

        print("\nTesting text-to-speech models:")
        test_text = "This is a test for text-to-speech capabilities."
        for model in text_to_speech:
            try:
                response = openai.audio.speech.create(
                    model=model,
                    voice="alloy",
                    input=test_text
                )
                print(f"Model {model} successfully generated speech audio.")
            except Exception as e:
                raise TestError(f"Error testing text-to-speech for model {model}: {e}") from e


    def _test_audio_translation_models(self):
        """Test the functionality of audio translation models."""
        # Test translation models
        print("\nTesting audio translation models:")

        test_audio_url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"

        for model in translation_models:
            try:
                with tempfile.NamedTemporaryFile(suffix=".flac") as temp_file:
                    # Download the test audio file
                    subprocess.run(["wget", "-O", temp_file.name, test_audio_url], check=True)

                    # Attempt translation
                    translation = openai.audio.translations.create(
                        model=model,
                        file=temp_file.name
                    )
                    print(f"Model {model} successfully translated audio. First 50 characters: {translation.text[:50]}...")
            except Exception as e:
                raise TestError(f"Error testing audio translation for model {model}: {e}") from e


    def _test_embedding_models(self):
        """Test the functionality of embedding models."""

        test_text = "This is a test sentence for embedding."
        for model in embedding_models:
            try:
                embedding = openai.embeddings.create(
                    model=model,
                    input=test_text
                )
                print(f"Model {model} successfully generated embeddings. Dimension: {len(embedding.data[0].embedding)}")
            except Exception as e:
                raise TestError(f"Error testing embedding model {model}: {e}") from e


    def _test_completions(self):
        """Test the functionality of completion models."""
        test_prompt = "Complete this sentence: The quick brown fox"
        for model in chat_completion_models:
            try:
                completion = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}]
                )
                print(f"Model {model} successfully generated completion: {completion.choices[0].message.content[:50]}...")
            except Exception as e:
                raise TestError(f"Error testing completion model {model}: {e}") from e


    def _test_image_models(self):
        """Test the functionality of image generation models."""
        test_prompt = "A serene landscape with mountains and a lake"
        for model in image_models:
            try:
                image = openai.images.generate(
                    model=model,
                    prompt=test_prompt,
                    n=1,
                    size="256x256"
                )
                print(f"Model {model} successfully generated an image. URL: {image.data[0].url}")
            except Exception as e:
                raise TestError(f"Error testing image model {model}: {e}") from e


    def _test_moderation_models(self):
        """Test the functionality of content moderation models."""
        test_text = "This is a test sentence for moderation."
        for model in moderation_models:
            try:
                moderation = openai.moderations.create(
                    model=model,
                    input=test_text
                )
                print(f"Model {model} successfully performed moderation. Flagged: {moderation.results[0].flagged}")
            except Exception as e:
                raise TestError(f"Error testing moderation model {model}: {e}") from e


    def _test_speech_to_text(self):
        """Test the functionality of speech-to-text models."""
        test_audio_url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
        for model in speech_to_text:
            try:
                with tempfile.NamedTemporaryFile(suffix=".flac") as temp_file:
                    subprocess.run(["wget", "-O", temp_file.name, test_audio_url], check=True)
                    transcript = openai.audio.transcriptions.create(
                        model=model,
                        file=temp_file.name
                    )
                print(f"Model {model} successfully transcribed audio: {transcript.text[:50]}...")
            except Exception as e:
                raise TestError(f"Error testing speech-to-text model {model}: {e}") from e


    def _test_text_to_speech(self):
        """Test the functionality of text-to-speech models."""
        test_text = "This is a test for text-to-speech capabilities."
        for model in text_to_speech:
            try:
                response = openai.audio.speech.create(
                    model=model,
                    voice="alloy",
                    input=test_text
                )
                print(f"Model {model} successfully generated speech audio.")
            except Exception as e:
                raise TestError(f"Error testing text-to-speech model {model}: {e}") from e


    def _test_whisper_models(self):
        """Test the functionality of Whisper models for audio transcription."""
        test_audio_url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
        for model in speech_to_text:
            if "whisper" in model:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".flac") as temp_file:
                        subprocess.run(["wget", "-O", temp_file.name, test_audio_url], check=True)
                        transcript = openai.audio.transcriptions.create(
                            model=model,
                            file=temp_file.name
                        )
                    print(f"Whisper model {model} successfully transcribed audio: {transcript.text[:50]}...")
                except Exception as e:
                    print(f"Error testing Whisper model {model}: {e}")


    def _test_translation_models(self):
        """Test the functionality of translation models."""
        test_audio_url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
        for model in translation_models:
            try:
                with tempfile.NamedTemporaryFile(suffix=".flac") as temp_file:
                    subprocess.run(["wget", "-O", temp_file.name, test_audio_url], check=True)
                    translation = openai.audio.translations.create(
                        model=model,
                        file=temp_file.name
                    )
                print(f"Model {model} successfully translated audio: {translation.text[:50]}...")
            except Exception as e:
                raise TestError(f"Error testing translation model {model}: {e}") from e


    def __test__(self):
        print("Starting OpenAI API tests...")
        self._test_model_endpoints()
        self._test_token_size()
        self._test_for_tool_model_tool_use()
        self._test_models_with_vision_capabilities()
        self._test_text_to_speech_models()
        self._test_audio_translation_models()
        self._test_embedding_models()
        self._test_completions()
        self._test_image_models()
        self._test_moderation_models()
        print("\nAll tests completed successfully.")
        return None
