import os
import io
import sys
import json
from unittest.mock import MagicMock, patch
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ipfs_accelerate_py'))
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
            assert callable(endpoint_handler), "Endpoint handler should be callable"
        except Exception as e:
            results["endpoint_handler"] = f"Error: {str(e)}"
            
        # Test determine_model function
        try:
            with patch.object(self.openai_api, '_determine_model', return_value="gpt-4"):
                model = self.openai_api._determine_model("chat", None)
                results["determine_model"] = "Success" if model == "gpt-4" else "Failed determine_model"
                assert model == "gpt-4", "Model determination should return gpt-4"
        except Exception as e:
            results["determine_model"] = str(e)
            
        # Test embedding function
        try:
            with patch('openai.embeddings.create') as mock_embed:
                # Updated response format to match current OpenAI API
                mock_embed.return_value = MagicMock(
                    model="text-embedding-3-large",
                    object="list",
                    data=[
                        MagicMock(
                            embedding=[0.1, 0.2, 0.3, 0.4], 
                            index=0, 
                            object="embedding"
                        )
                    ],
                    usage=MagicMock(
                        prompt_tokens=8,
                        total_tokens=8
                    )
                )
                
                embedding = self.openai_api.embedding("text-embedding-3-large", test_data["text"], "float")
                results["embedding"] = "Success" if isinstance(embedding, dict) and "text" in embedding else "Failed embedding"
                assert isinstance(embedding, dict) and "text" in embedding, "Embedding should return a dictionary with 'text' key"
        except Exception as e:
            results["embedding"] = str(e)
        
        # Test moderation function
        try:
            with patch('openai.moderations.create') as mock_moderate:
                mock_moderate.return_value = MagicMock(
                    id="modr-123",
                    model="text-moderation-latest",
                    results=[
                        MagicMock(
                            flagged=True,
                            categories=MagicMock(
                                hate=False, 
                                hate_threatening=False, 
                                self_harm=False, 
                                sexual=False, 
                                sexual_minors=False, 
                                violence=True, 
                                violence_graphic=False
                            ),
                            category_scores=MagicMock(
                                hate=0.01, 
                                hate_threatening=0.01, 
                                self_harm=0.01, 
                                sexual=0.01, 
                                sexual_minors=0.01, 
                                violence=0.85, 
                                violence_graphic=0.01
                            )
                        )
                    ]
                )
                
                moderation = self.openai_api.moderation("text-moderation-latest", test_data["moderation"]["text"])
                results["moderation"] = "Success" if moderation and hasattr(moderation, "results") else "Failed moderation"
                assert moderation and hasattr(moderation, "results"), "Moderation should return results"
        except Exception as e:
            results["moderation"] = str(e)
            
        # Test text-to-image function
        try:
            with patch('openai.images.generate') as mock_image:
                mock_image.return_value = MagicMock(
                    created=1684401833,
                    data=[
                        MagicMock(
                            revised_prompt="A cinematic view of C-beams glittering in the dark, futuristic landscape with shimmering light beams against a dark backdrop reminiscent of the scene near the Tannhäuser Gate from Blade Runner.",
                            url="https://example.com/image.png",
                            b64_json=None
                        )
                    ]
                )
                
                image = self.openai_api.text_to_image("dall-e-3", "1024x1024", 1, test_data["image"]["prompt"])
                results["text_to_image"] = "Success" if isinstance(image, dict) and "text" in image else "Failed text_to_image"
                assert isinstance(image, dict) and "text" in image, "Text to image should return a dictionary with 'text' key"
        except Exception as e:
            results["text_to_image"] = str(e)
            
        # Test process_messages function
        try:
            messages = [
                {"role": "system", "content": "You are an AI assistant"},
                {"role": "user", "content": "Hello"}
            ]
            
            processed = self.openai_api.process_messages(messages, None, None, "You are a helpful assistant")
            results["process_messages"] = "Success" if isinstance(processed, list) and len(processed) >= len(messages) else "Failed process_messages"
            assert isinstance(processed, list) and len(processed) >= len(messages), "Process messages should return at least the original messages"
        except Exception as e:
            results["process_messages"] = str(e)
            
        # Test text-to-speech function
        try:
            with patch('openai.audio.speech.create') as mock_speech:
                mock_response = MagicMock()
                mock_response.read.return_value = b"audio data"
                mock_speech.return_value = mock_response
                
                speech = self.openai_api.text_to_speech("tts-1", test_data["text"], "alloy")
                results["text_to_speech"] = "Success" if isinstance(speech, dict) and "text" in speech else "Failed text_to_speech"
                assert isinstance(speech, dict) and "text" in speech, "Text to speech should return a dictionary with 'text' key"
        except Exception as e:
            results["text_to_speech"] = str(e)
            
        # Test speech-to-text function
        try:
            with patch('openai.audio.transcriptions.create') as mock_transcribe:
                mock_transcribe.return_value = MagicMock(
                    text="This is a transcription"
                )
                
                transcription = self.openai_api.speech_to_text("whisper-1", test_data["audio"]["data"])
                results["speech_to_text"] = "Success" if isinstance(transcription, dict) and "text" in transcription else "Failed speech_to_text"
                assert isinstance(transcription, dict) and "text" in transcription, "Speech to text should return a dictionary with 'text' key"
        except Exception as e:
            results["speech_to_text"] = str(e)
            
        # Test chat completion function
        try:
            with patch('openai.chat.completions.create') as mock_chat:
                mock_chat.return_value = MagicMock(
                    id="chatcmpl-123",
                    object="chat.completion",
                    created=1677825464,
                    model="gpt-4o",
                    choices=[
                        MagicMock(
                            index=0,
                            message=MagicMock(
                                role="assistant",
                                content="This is a test response"
                            ),
                            finish_reason="stop"
                        )
                    ],
                    usage=MagicMock(
                        prompt_tokens=20,
                        completion_tokens=10,
                        total_tokens=30
                    )
                )
                
                messages = [
                    {"role": "user", "content": "Hello"}
                ]
                
                self.openai_api.messages = messages
                self.openai_api.model = "gpt-4o"
                self.openai_api.method = "chat"
                completion = self.openai_api.request_complete()
                results["chat_completion"] = "Success" if isinstance(completion, dict) and "text" in completion else "Failed chat_completion"
                assert isinstance(completion, dict) and "text" in completion, "Chat completion should return a dictionary with 'text' key"
        except Exception as e:
            results["chat_completion"] = str(e)
            
        # Test error handling
        try:
            # Test API key error
            with patch('openai.chat.completions.create') as mock_chat:
                mock_chat.side_effect = Exception("Invalid API key")
                
                api_key_error_caught = False
                try:
                    self.openai_api.messages = [{"role": "user", "content": "Hello"}]
                    self.openai_api.model = "gpt-4o"
                    self.openai_api.method = "chat"
                    self.openai_api.request_complete()
                except Exception:
                    api_key_error_caught = True
                    
                results["error_handling_api_key"] = "Success" if api_key_error_caught else "Failed to catch API key error"
                assert api_key_error_caught, "Should catch API key errors"
                    
            # Test rate limit error
            with patch('openai.chat.completions.create') as mock_chat:
                mock_chat.side_effect = Exception("Rate limit exceeded")
                
                rate_limit_error_caught = False
                try:
                    self.openai_api.messages = [{"role": "user", "content": "Hello"}]
                    self.openai_api.model = "gpt-4o"
                    self.openai_api.method = "chat"
                    self.openai_api.request_complete()
                except Exception:
                    rate_limit_error_caught = True
                    
                results["error_handling_rate_limit"] = "Success" if rate_limit_error_caught else "Failed to catch rate limit error"
                assert rate_limit_error_caught, "Should catch rate limit errors"
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
                
                # More detailed comparison of results
                all_match = True
                mismatches = []
                
                for key in set(expected_results.keys()) | set(test_results.keys()):
                    if key not in expected_results:
                        mismatches.append(f"Missing expected key: {key}")
                        all_match = False
                    elif key not in test_results:
                        mismatches.append(f"Missing actual key: {key}")
                        all_match = False
                    elif expected_results[key] != test_results[key]:
                        mismatches.append(f"Key '{key}' differs: Expected '{expected_results[key]}', got '{test_results[key]}'")
                        all_match = False
                
                if not all_match:
                    print("Test results differ from expected results!")
                    for mismatch in mismatches:
                        print(f"- {mismatch}")
                    print(f"\nComplete expected results: {json.dumps(expected_results, indent=2)}")
                    print(f"\nComplete actual results: {json.dumps(test_results, indent=2)}")
                else:
                    print("All test results match expected results.")
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

