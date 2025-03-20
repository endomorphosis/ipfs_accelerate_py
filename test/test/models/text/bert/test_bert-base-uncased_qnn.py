#!/usr/bin/env python3
"""
Test for bert-base-uncased on qnn hardware.
"""

import unittest
import numpy as np
from bert_base_uncased_qnn_skill import bert_base_uncasedSkill

class TestBert_Base_Uncased:
    """Test suite for bert-base-uncased on qnn."""
    
    def test_setup(self):
        """Test model setup."""
        skill = bert_base_uncasedSkill()
        success = skill.setup()
        assert success, "Model setup should succeed"
    
    def test_run(self):
        """Test model inference."""
        skill = bert_base_uncasedSkill()
        skill.setup()
        result = skill.run("Test input")
        assert "outputs" in result, "Result should contain outputs"
