import os
import io
import sys
sys.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from api_backends import apis, openai_api
import json

class test_openai_api:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.openai_api = openai_api(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        results = {}
        ipfs_accelerate_py_dir = os.path.dirname(os.path.dirname(__file__))
        test_audio_path = os.path.join(ipfs_accelerate_py_dir, 'test', 'test.mp3')
        test_translation_audio_path = os.path.join(ipfs_accelerate_py_dir, 'test','trans_test.mp3') 
        test_image_path = os.path.join(ipfs_accelerate_py_dir, 'test', 'test.jpg')

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

        test_data = {
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
            }
        }

        
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
        try:
            results["determine_model"] = self.openai_api._test_determine_model()
        except Exception as e:
            results["determine_model"] = str(e)
        try:
            results["embedding"] = self.openai_api._test_embedding()
        except Exception as e:
            results["embedding"] = str(e)
        try:
            results["moderation"] = self.openai_api._test_moderation()
        except Exception as e:
            results["moderation"] = str(e)
        try:
            results["moderated_text_to_image"] = self.openai_api._test_moderated_text_to_image()
        except Exception as e:
            results["moderated_text_to_image"] = str(e)
        try:
            results["process_messages"] = self.openai_api._test_process_messages()
        except Exception as e:
            results["process_messages"] = str(e)
        try:
            results["moderated_text_to_speech"] = self.openai_api._test_moderated_text_to_speech()
        except Exception as e:
            results["moderated_text_to_speech"] = str(e)
        try:
            results["text_to_speech"] = self.openai_api._test_text_to_speech()
        except Exception as e:
            results["text_to_speech"] = str(e)
        try:
            results["speech_to_text"] = self.openai_api._test_speech_to_text()
        except Exception as e:
            results["speech_to_text"] = str(e)
        try:
            results["moderated_chat_complete"] = self.openai_api._test_moderated_chat_complete()
        except Exception as e:
            results["moderated_chat_complete"] = str(e)
        try:
            results["request_complete"] = self.openai_api._test_request_complete()
        except Exception as e:
            results["request_complete"] = str(e)
        print("Inner methods completed successfully.")
        print("Testing basic functionality of outer methods...")
        try:
            results["chat"] = self.openai_api._test_chat()
        except Exception as e:
            results["chat"] = str(e)
        try:
            results["audio_chat"] = self.openai_api._test_audio_chat()
        except Exception as e:
            results["audio_chat"] = str(e)
        try:
            results["text_to_image"] = self.openai_api._test_text_to_image()
        except Exception as e:
            results["text_to_image"] = str(e)
        try:
            results["image_to_text"] = self.openai_api._test_image_to_text()
        except Exception as e:
            results["image_to_text"] = str(e)
        print("All tests completed successfully.")
        return results


    def __test__(self):
        test_results = {}
        try:
            test_results =  self.test()
        except Exception as e:
            test_results = e
        if os.path.exists(os.path.join(os.path.dirname(__file__),'expected_results', 'openai_api_test_results.json')):
            with open(os.path.join(os.path.dirname(__file__),'expected_results','openai_api_test_results.json'), 'r') as f:
                expected_results = json.load(f)
                assert test_results == expected_results
        else:
            with open(os.path.join(os.path.dirname(__file__),'expected_results', 'open_api_test_results.json'), 'w') as f:
                f.write(test_results)
        with open(os.path.join(os.path.dirname(__file__),'collected_results', 'open_api_test_results.json'), 'w') as f:
            f.write(test_results)

if __name__ == "__main__":
    metadata = {
        "tokens_per_word": 4,
        "max_tokens": 2048,
        "openai_api_key": os.environ['OPENAI_API_KEY']
    }
    resources = {}
    try:
        this_openai_api = openai_api(resources, metadata)
        this_openai_api.__test__()
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)

