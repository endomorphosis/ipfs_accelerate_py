import os
import io
import sys
sys.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from api_backends import apis, hf_tei
import json

class test_hf_tei:
    def __init__(self, resources=None, metadata=None):
        self.resources = resources
        self.metadata = metadata
        self.hf_tei = hf_tei(resources=self.resources, metadata=self.metadata)
        return None
    
    def test(self):
        try:
            return self.hf_tei.__test__()
        except Exception as e:
            return e
        
    def __test__(self):
        test_results = {}
        try:
            test_results =  self.test()
        except Exception as e:
            test_results = e
        if os.path.exists(os.path.join(os.path.dirname(__file__),'expected_results', 'hf_tei_test_results.json')):
            with open(os.path.join(os.path.dirname(__file__),'expected_results','hf_tei_test_results.json'), 'r') as f:
                expected_results = json.load(f)
                assert test_results == expected_results
        else:
            with open(os.path.join(os.path.dirname(__file__),'expected_results', 'hf_tei_test_results.json'), 'w') as f:
                f.write(test_results)
        with open(os.path.join(os.path.dirname(__file__),'collected_results', 'hf_tei_test_results.json'), 'w') as f:
            f.write(test_results)
    
if __name__ == "__main__":
    metadata = {}
    resources = {}
    try:
        this_hf_tei = test_hf_tei(resources, metadata)
        this_hf_tei.__test__()
    except KeyboardInterrupt:
        print("Tests stopped by user.")
        sys.exit(1)