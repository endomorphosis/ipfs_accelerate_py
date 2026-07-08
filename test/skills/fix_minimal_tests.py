#\!/usr/bin/env python3
"""
Fix indentation issues in minimal test files.
"""
import os
import sys
import re

def fix_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix HAS_X logger warnings
    content = content.replace('    HAS_TORCH = False\n        logger.warning', '    HAS_TORCH = False\n    logger.warning')
    content = content.replace('    HAS_TRANSFORMERS = False\n        logger.warning', '    HAS_TRANSFORMERS = False\n    logger.warning')
    content = content.replace('    HAS_TOKENIZERS = False\n        logger.warning', '    HAS_TOKENIZERS = False\n    logger.warning')
    content = content.replace('    HAS_SENTENCEPIECE = False\n        logger.warning', '    HAS_SENTENCEPIECE = False\n    logger.warning')
    content = content.replace('    HAS_PIL = False\n        logger.warning', '    HAS_PIL = False\n    logger.warning')
    
    # Fix method boundaries
    content = content.replace('self.performance_stats = {}    def', 'self.performance_stats = {}\n\n    def')
    content = content.replace('self.examples = []    self.performance_stats', 'self.examples = []\n        self.performance_stats')
    content = content.replace('return results    def', 'return results\n\n    def')
    
    # Fix indentation in try/except blocks
    content = content.replace('        try:\n        import openvino', '    try:\n        import openvino')
    
    # Fix mock class assignments
    content = content.replace('        tokenizers.Tokenizer = MockTokenizer', '    tokenizers.Tokenizer = MockTokenizer')
    content = content.replace('        sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor', '    sentencepiece.SentencePieceProcessor = MockSentencePieceProcessor')
    content = content.replace('        Image.open = MockImage.open', '    Image.open = MockImage.open')
    content = content.replace('        requests.get = MockRequests.get', '    requests.get = MockRequests.get')

    # Fix if statement spacing
    content = content.replace('logger.warning("sentencepiece not available, using mock")        if not HAS_TOKENIZERS:', 'logger.warning("sentencepiece not available, using mock")\n\nif not HAS_TOKENIZERS:')
    content = content.replace('logger.warning("tokenizers not available, using mock")        if not HAS_SENTENCEPIECE:', 'logger.warning("tokenizers not available, using mock")\n\nif not HAS_SENTENCEPIECE:')

    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    base_dir = './minimal_tests'
    for model in ['bert', 'gpt2', 't5', 'vit']:
        file_path = os.path.join(base_dir, f'test_hf_{model}.py')
        if os.path.exists(file_path):
            fix_file(file_path)
