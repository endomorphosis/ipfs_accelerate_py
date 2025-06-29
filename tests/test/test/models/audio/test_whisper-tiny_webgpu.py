#!/usr/bin/env python3
"""
Test for whisper-tiny on webgpu hardware.
"""

import unittest
import numpy as np
from whisper_tiny_webgpu_skill import whisper_tinySkill

class TestWhisper_Tiny:
    """Test suite for whisper-tiny on webgpu."""
    
    def test_setup(self):
        """Test model setup."""
        skill = whisper_tinySkill()
        success = skill.setup()
        assert success, "Model setup should succeed"
    
    def test_run(self):
        """Test model inference."""
        skill = whisper_tinySkill()
        skill.setup()
        result = skill.run("Test input")
        assert "outputs" in result, "Result should contain outputs"
