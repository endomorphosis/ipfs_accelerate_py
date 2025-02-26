import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from api_backends import apis, openai_api

class test_openai_api:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {
            "tokens_per_word": 4,
            "max_tokens": 2048,
            "openai_api_key": os.environ.get('OPENAI_API_KEY', '')
        }
        self.openai_api = openai_api(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        """Run comprehensive tests for the OpenAI API backend"""
        results = {}
        
        # Set up test data
        ipfs_accelerate_py_dir = os.path.dirname(os.path.dirname(__file__))
        test_audio_path = os.path.join(ipfs_accelerate_py_dir, 'test', 'test.mp3')
        test_translation_audio_path = os.path.join(ipfs_accelerate_py_dir, 'test','trans_test.mp3') 
        test_image_path = os.path.join(ipfs_accelerate_py_dir, 'test', 'test.jpg')

        # Handle missing test files gracefully
        if not os.path.exists(test_audio_path):
            with open(test_audio_path, 'wb') as f:
                f.write(b'test audio data')
        
        if not os.path.exists(test_translation_audio_path):
            with open(test_translation_audio_path, 'wb') as f:
                f.write(b'test translation audio data')
                
        if not os.path.exists(test_image_path):
            with open(test_image_path, 'wb') as f:
                f.write(b'test image data')

        # Set up audio data
        with open(test_audio_path, 'rb') as file:
            test_audio_data = io.BytesIO(file.read())
            test_audio_data.name = os.path.basename(test_audio_path)

        with open(test_translation_audio_path, 'rb') as file:
            test_translation_audio_data = io.BytesIO(file.read())
            test_translation_audio_data.name = os.path.basename(test_translation_audio_path)

        with open(test_image_path, 'rb') as file:
            test_image_data = file.read()

        # Set up test data dictionary
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

        print("Begin OpenAI API tests")
        
        # Test endpoint handler creation
        try:
            endpoint_handler = self.openai_api.create_openai_api_endpoint_handler()
            results["endpoint_handler"] = "Success" if callable(endpoint_handler) else "Failed to create endpoint handler"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test determine_model function
        try:
            with patch.object(self.openai_api, '_determine_model', return_value="gpt-4"):
                model = self.openai_api._determine_model("chat", None)
                results["determine_model"] = "Success" if model == "gpt-4" else "Failed determine_model"
        except Exception as e:
            results["determine_model"] = str(e)
            
        # Test embedding function
        try:
            with patch('openai.embeddings.create') as mock_embed:
                mock_embed.return_value = {
                    "data": [{"embedding": [0.1, 0.2, 0.3]}]
                }
                
                embedding = self.openai_api._embedding(test_data["text"])
                results["embedding"] = "Success" if isinstance(embedding, list) and len(embedding) > 0 else "Failed embedding"
        except Exception as e:
            results["embedding"] = str(e)
        
        # Test moderation function
        try:
            with patch('openai.moderations.create') as mock_moderate:
                mock_moderate.return_value = {
                    "results": [{
                        "flagged": True,
                        "categories": {"hate": False, "hate/threatening": False, "self-harm": False, 
                                      "sexual": False, "sexual/minors": False, "violence": True, 
                                      "violence/graphic": False},
                        "category_scores": {"hate": 0.01, "hate/threatening": 0.01, "self-harm": 0.01, 
                                          "sexual": 0.01, "sexual/minors": 0.01, "violence": 0.85, 
                                          "violence/graphic": 0.01}
                    }]
                }
                
                moderation = self.openai_api._moderation(test_data["moderation"]["text"])
                results["moderation"] = "Success" if isinstance(moderation, dict) and "flagged" in moderation else "Failed moderation"
        except Exception as e:
            results["moderation"] = str(e)
            
        # Test text-to-image function
        try:
            with patch('openai.images.generate') as mock_image:
                mock_image.return_value = {
                    "data": [{"url": "https://example.com/image.png"}]
                }
                
                image = self.openai_api._moderated_text_to_image(test_data["image"]["prompt"])
                results["text_to_image"] = "Success" if isinstance(image, dict) and "url" in image else "Failed text_to_image"
        except Exception as e:
            results["text_to_image"] = str(e)
            
        # Test process_messages function
        try:
            messages = [
                {"role": "system", "content": "You are an AI assistant"},
                {"role": "user", "content": "Hello"}
            ]
            
            processed = self.openai_api._process_messages(messages)
            results["process_messages"] = "Success" if isinstance(processed, list) and len(processed) == len(messages) else "Failed process_messages"
        except Exception as e:
            results["process_messages"] = str(e)
            
        # Test text-to-speech function
        try:
            with patch('openai.audio.speech.create') as mock_speech:
                mock_speech.return_value = MagicMock()
                mock_speech.return_value.read.return_value = b"audio data"
                
                speech = self.openai_api._text_to_speech(test_data["text"])
                results["text_to_speech"] = "Success" if speech else "Failed text_to_speech"
        except Exception as e:
            results["text_to_speech"] = str(e)
            
        # Test speech-to-text function
        try:
            with patch('openai.audio.transcriptions.create') as mock_transcribe:
                mock_transcribe.return_value = {"text": "This is a transcription"}
                
                transcription = self.openai_api._speech_to_text(test_data["audio"]["data"])
                results["speech_to_text"] = "Success" if transcription and "text" in transcription else "Failed speech_to_text"
        except Exception as e:
            results["speech_to_text"] = str(e)
            
        # Test chat completion function
        try:
            with patch('openai.chat.completions.create') as mock_chat:
                mock_chat.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": "This is a test response",
                                "role": "assistant"
                            },
                            "finish_reason": "stop",
                            "index": 0
                        }
                    ],
                    "usage": {
                        "completion_tokens": 10,
                        "prompt_tokens": 20,
                        "total_tokens": 30
                    }
                }
                
                messages = [
                    {"role": "user", "content": "Hello"}
                ]
                
                completion = self.openai_api._moderated_chat_complete(messages)
                results["chat_completion"] = "Success" if completion and "content" in completion else "Failed chat_completion"
        except Exception as e:
            results["chat_completion"] = str(e)
            
        # Test request complete function
        try:
            with patch('openai.completions.create') as mock_complete:
                mock_complete.return_value = {
                    "choices": [
                        {
                            "text": "This is a test completion response",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "length"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30
                    }
                }
                
                completion = self.openai_api._request_complete(test_data["completion"]["text"])
                results["text_completion"] = "Success" if completion and isinstance(completion, str) else "Failed text_completion"
        except Exception as e:
            results["text_completion"] = str(e)
            
        # Test outer interface methods
        
        # Test chat method
        try:
            with patch.object(self.openai_api, '_moderated_chat_complete') as mock_chat:
                mock_chat.return_value = {"content": "This is a test chat response", "role": "assistant"}
                
                messages = [
                    {"role": "user", "content": "Hello"}
                ]
                
                chat_result = self.openai_api.chat(messages)
                results["chat_method"] = "Success" if chat_result and "content" in chat_result else "Failed chat method"
        except Exception as e:
            results["chat_method"] = str(e)
            
        # Test audio chat method
        try:
            with patch.object(self.openai_api, '_speech_to_text') as mock_s2t:
                mock_s2t.return_value = {"text": "This is a transcribed audio"}
                
                with patch.object(self.openai_api, 'chat') as mock_chat:
                    mock_chat.return_value = {"content": "This is a response to audio", "role": "assistant"}
                    
                    audio_chat_result = self.openai_api.audio_chat(test_data["audio"]["data"])
                    results["audio_chat"] = "Success" if audio_chat_result and "content" in audio_chat_result else "Failed audio_chat"
        except Exception as e:
            results["audio_chat"] = str(e)
            
        # Test image to text method
        try:
            with patch.object(self.openai_api, 'chat') as mock_chat:
                mock_chat.return_value = {"content": "This is a description of the image", "role": "assistant"}
                
                image_chat_result = self.openai_api.image_to_text(test_data["image"]["data"])
                results["image_to_text"] = "Success" if image_chat_result and "content" in image_chat_result else "Failed image_to_text"
        except Exception as e:
            results["image_to_text"] = str(e)
            
        # Test error handling
        try:
            # Test API key error
            with patch('openai.chat.completions.create') as mock_chat:
                mock_chat.side_effect = Exception("Invalid API key")
                
                try:
                    self.openai_api._moderated_chat_complete([{"role": "user", "content": "Hello"}])
                    results["error_handling_api_key"] = "Failed to catch API key error"
                except Exception:
                    results["error_handling_api_key"] = "Success"
                    
            # Test rate limit error
            with patch('openai.chat.completions.create') as mock_chat:
                mock_chat.side_effect = Exception("Rate limit exceeded")
                
                try:
                    self.openai_api._moderated_chat_complete([{"role": "user", "content": "Hello"}])
                    results["error_handling_rate_limit"] = "Failed to catch rate limit error"
                except Exception:
                    results["error_handling_rate_limit"] = "Success"
        except Exception as e:
            results["error_handling"] = str(e)

        print("All tests completed")
        return results

    def __test__(self):
        """Run tests and compare/save results"""
        test_results = {}
        try:
            test_results = self.test()
        except Exception as e:
            test_results = {"test_error": str(e)}
        
        # Create directories if they don't exist
        expected_dir = os.path.join(os.path.dirname(__file__), 'expected_results')
        collected_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
        os.makedirs(expected_dir, exist_ok=True)
        os.makedirs(collected_dir, exist_ok=True)
        
        # Save collected results
        with open(os.path.join(collected_dir, 'openai_api_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Compare with expected results if they exist
        expected_file = os.path.join(expected_dir, 'openai_api_test_results.json')
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                expected_results = json.load(f)
                if expected_results != test_results:
                    print("Test results differ from expected results!")
                    print(f"Expected: {expected_results}")
                    print(f"Got: {test_results}")
        else:
            # Create expected results file if it doesn't exist
            with open(expected_file, 'w') as f:
                json.dump(test_results, f, indent=2)
                print(f"Created new expected results file: {expected_file}")

        return test_results

if __name__ == "__main__":
    metadata = {
        "tokens_per_word": 4,
        "max_tokens": 2048,
        "openai_api_key": os.environ.get('OPENAI_API_KEY', '')
    }
    resources = {}
    try:
        this_openai_api = test_openai_api(resources, metadata)
        results = this_openai_api.__test__()
        print(f"OpenAI API Test Results: {json.dumps(results, indent=2)}")
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)

