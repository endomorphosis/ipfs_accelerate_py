#!/usr/bin/env python3
"""
Test for gpt2 on webgpu hardware.
"""

import unittest
import numpy as np
from gpt2_webgpu_skill import gpt2Skill

class TestGpt2:
    """Test suite for gpt2 on webgpu."""
    
    def test_setup(self):
        """Test model setup."""
        skill = gpt2Skill()
        success = skill.setup()
        assert success, "Model setup should succeed"
    
    def test_run(self):
        """Test model inference."""
        skill = gpt2Skill()
        skill.setup()
        result = skill.run("Test input")
        assert "outputs" in result, "Result should contain outputs"
