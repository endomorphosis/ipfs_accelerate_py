from cloudkit_worker import dispatch_result, should_abort, TaskAbortion
import gc
import base64
import io
import os
import tempfile
import wave
import torch
import numpy as np
from typing import List
from pydantic import BaseModel
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except (ImportError, ValueError):
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable") 

class StreamingInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = "20"


class TTSInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str


class coqui_tts_kit:

    def __init__(self, resources, meta=None):
        self.tts = self.tts
        self.stream_tts = self.stream_tts
        
        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper(auto_detect_ci=True)
            except Exception:
                self._storage = None
        else:
            self._storage = None
        
        header_path = os.path.join(os.path.dirname(__file__), "test", "assets", "header.bin")
        if self._storage and self._storage.is_distributed:
            try:
                self.header = self._storage.read_file(header_path)
            except Exception:
                with open(header_path, "rb") as f:
                    self.header = f.read()
        else:
            with open(header_path, "rb") as f:
                self.header = f.read()
        self.dataPreprocess = lambda x: base64.b64decode(x.split(",")[1])
        self.embed = self.predict_speaker
        if "checkpoint" in resources.keys():
            self.custom_model_path = resources['checkpoint']
        else:
            self.custom_model_path = ''
        if os.path.exists(self.custom_model_path) and os.path.isfile(self.custom_model_path + "/config.json"):
            model_path = self.custom_model_path
            print("Loading custom model from", model_path, flush=True)
            self.config = XttsConfig()
            config_path = os.path.join(model_path, "config.json")
            if self._storage and self._storage.is_distributed:
                try:
                    cached_config = self._storage.read_file(config_path)
                    if cached_config:
                        import json
                        config_data = json.loads(cached_config.decode('utf-8'))
                        self.config.load_json(config_path)
                except Exception:
                    self.config.load_json(config_path)
            else:
                self.config.load_json(config_path)
            self.model = model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(self.config, checkpoint_dir=model_path , eval=True, use_deepspeed=True if device == "cuda" else False)
            self.model.to(device)
        else:
            load_model = None
            if self.custom_model_path != '':
                try:
                    print("Downloading XTTS Model:", self.custom_model_path, flush=True)
                    ModelManager().download_model(self.custom_model_path)
                    model_path = os.path.join(get_user_data_dir("tts"), self.custom_model_path.replace("/", "--"))
                    print("XTTS Model downloaded", flush=True)
                except Exception as e:
                    print("Custom model not found, loading default model", flush=True)
                    pass
                finally:
                    model_path = os.path.join(get_user_data_dir("tts"), self.custom_model_path.replace("/", "--"))
                    load_model = True
                    pass
                pass
            if load_model is None:
                self.custom_model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
                print("Downloading XTTS Model:", self.custom_model_path, flush=True)
                ModelManager().download_model(self.custom_model_path)
                model_path = os.path.join(get_user_data_dir("tts"), self.custom_model_path.replace("/", "--"))
                pass
            self.config = XttsConfig()
            config_path = os.path.join(model_path, "config.json")
            if self._storage and self._storage.is_distributed:
                try:
                    cached_config = self._storage.read_file(config_path)
                    if cached_config:
                        import json
                        config_data = json.loads(cached_config.decode('utf-8'))
                        self.config.load_json(config_path)
                except Exception:
                    self.config.load_json(config_path)
            else:
                self.config.load_json(config_path)
            self.model = model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(self.config, checkpoint_dir=model_path , eval=True, use_deepspeed=True if device == "cuda" else False)
            self.model.to(device)

    def __call__(self, method, **kwargs):
        if method == 'stream_tts':
            return self.stream_tts(**kwargs)
        if method == 'tts':
            return self.tts(**kwargs)
        if method == 'embed':
            return self.predict_speaker(**kwargs)
        if method == 'speakers':
            return self.get_speakers()
        if method == 'languages':
            return self.get_languages()
        else:
            raise Exception('unknown method: %s' % method)
        
    def tts(self, input, **kwargs):
        if "language" in kwargs.keys():
            language = kwargs["language"]
        else:
            language = 'en'

        if "speaker" in kwargs.keys():
            speaker = kwargs["speaker"]
        else:
            speaker = None

        if speaker is None:
            speaker = "Daisy Studious"

        speakers = self.get_speakers()
        if speaker not in speakers.keys():
            raise Exception('unknown speaker: %s' % speaker)
        
        speaker = speakers[speaker]
        speaker["text"] = input
        speaker["language"] = language
        parsed_input = TTSInputs(**speaker)
        results = self.predict_speech(parsed_input)

        return {
            'files': {
                'result': results
            }
        }

    def postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = wav[None, : int(wav.shape[0])]
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def predict_speaker(self, wav_file):
        """Compute conditioning inputs from reference audio file."""
        audio_bytes = wav_file
        with tempfile.NamedTemporaryFile(suffix=".ogg", dir="/tmp") as temp_audio:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm").export(temp_audio.name, format="ogg")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                temp_audio.name
            )
            return {
                "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
                "done": True,
            }

    def get_speakers(self):
        if hasattr(self.model, "speaker_manager") and hasattr(self.model.speaker_manager, "speakers"):
            return {
                speaker: {
                    "speaker_embedding": self.model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                    "gpt_cond_latent": self.model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
                }
                for speaker in self.model.speaker_manager.speakers.keys()
            }
        else:
            return {}

    def predict_streaming_generator(self, parsed_input):
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
        text = parsed_input.text
        language = parsed_input.language

        stream_chunk_size = int(parsed_input.stream_chunk_size)
        add_wav_header = parsed_input.add_wav_header


        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=stream_chunk_size,
            enable_text_splitting=True
        )

        for i, chunk in enumerate(chunks):
            chunk = self.postprocess(chunk)
            if i == 0 and add_wav_header:
                yield self.encode_audio_common(b"", encode_base64=False)
                yield chunk.tobytes()
            else:
                yield chunk.tobytes()

    def predict_speech(self, parsed_input: TTSInputs):
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
        text = parsed_input.text
        language = parsed_input.language

        out = self.model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
        )

        wav = self.postprocess(torch.tensor(out["wav"]))
        wav_output = wav.tobytes()
        return self.encode_audio_common(wav_output)

    def encode_audio_common(
        self, frame_input, encode_base64=False, sample_rate=24000, sample_width=2, channels=1
    ):
        """Return base64 encoded audio"""
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as vfout:
            vfout.setnchannels(channels)
            vfout.setsampwidth(sample_width)
            vfout.setframerate(sample_rate)
            vfout.writeframes(frame_input)

        wav_buf.seek(0)
        if encode_base64:
            b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
            return b64_encoded
        else:
            return wav_buf.read()

    def get_languages(self):
        return self.config.languages

    def stream_tts(self, input, **kwargs):
        chunk_size = None
        speaker = None
        language = None
        if "language" in kwargs.keys():
            language = kwargs["language"]
        else:
            language = 'en'
        
        if language is None:
            language = 'en'

        if "chunk_size" in kwargs.keys():
            chunk_size = kwargs["chunk_size"]
        else:
            chunk_size = 20

        if chunk_size is None:
            chunk_size = 20

        if "speaker" in kwargs.keys():
            speaker = kwargs["speaker"]
        else:
            speaker = None

        if speaker is None:
            speaker = "Daisy Studious"

        speakers = self.get_speakers()
        if speaker not in speakers.keys():
            raise Exception('unknown speaker: %s' % speaker)
        
        if len(input) < chunk_size:
            ##pad input to left with spaces
            input = input.rjust(chunk_size, ' ')
    
        speaker = speakers[speaker]
        speaker["text"] = input
        speaker["language"] = language
        speaker['add_wav_header'] = True
        speaker["stream_chunk_size"] = str(chunk_size)
        parsed_input = StreamingInputs(**speaker)
        results = self.predict_streaming_generator(parsed_input)
        stream = True
        final_result = bytes()
        for chunk in results:
            if should_abort():
                raise TaskAbortion

            audio = chunk

            if audio == '':
                continue
							
            final_result = final_result + audio

            if stream:
                yield {
					'audio': audio,
					'done': False
				}

        yield {
			'audio': '' if stream else final_result,
			'done': True
		}
    
    def test_tts(self):
        ## iterate through the generator to get the audio
        input = "hello world"
        results = self.tts(input)
        print(results)

    def test_predict_speaker(self):
        this_dir = os.path.dirname(__file__)
        audio_path = os.path.join(this_dir,"test", "assets", "base64ogg.txt")
        header_path = os.path.join(this_dir,"test", "assets", "header.bin")
        audio_data = None
        header_data = None
        with open(audio_path, "r") as f:
            audio_data = f.read()
        audio_data = self.dataPreprocess(audio_data)
        with open(header_path, "rb") as f:
            header_data = f.read()
        request = self.predict_speaker(header_data + audio_data)
        #print(request)
        return request

    def unload(self):
        del self.model
        if hasattr(self, 'pipe'):
            del self.pipe
        gc.collect()


#if __name__ == "__main__":
#    test_tts = coqui_tts_kit({"checkpoint":"/storage/cloudkit-models/tts_models--multilingual--multi-dataset--xtts_v2@hf/"}, {})
#    test_tts.test_predict_speaker()
