
import os
import json
import tempfile
import random
import string

def get_test_data_path(filename):
    """Get path to a test data file."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    return os.path.join(test_data_dir, filename)

def create_temp_file(content, suffix=".txt"):
    """Create a temporary file with the given content."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    except:
        os.unlink(path)
        raise

def random_string(length=10):
    """Generate a random string of the given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def load_json_data(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_data(data, filepath):
    """Save JSON data to a file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
