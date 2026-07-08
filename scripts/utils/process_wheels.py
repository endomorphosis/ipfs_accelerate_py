import os
import glob
import sys
import subprocess

def find_wheel_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wheels = []
    
    for dir_name in ['wheels', 'new-wheels']:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            wheels.extend(glob.glob(os.path.join(dir_path, '*.whl')))
    
    return sorted(wheels)

def install_wheel(wheel_path):
    print(f"\nInstalling {os.path.basename(wheel_path)}...")
    try:
        # Install wheel directly with pip
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-v", wheel_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Command output:\n{result.stdout}")
        if result.stderr:
            print(f"Errors:\n{result.stderr}")
        return result.returncode == 0
            
    except Exception as e:
        print(f"Error installing {wheel_path}: {str(e)}")
        if hasattr(e, 'output'):
            print(f"Command output: {e.output}")
        return False

if __name__ == '__main__':
    wheels = find_wheel_files()
    if not wheels:
        print("No wheel files found!")
        sys.exit(1)
        
    print(f"Found {len(wheels)} wheels to process")
    processed = set()
    success_count = 0
    
    for wheel in wheels:
        wheel_name = os.path.basename(wheel)
        if wheel_name in processed:
            print(f"Skipping duplicate wheel: {wheel_name}")
            continue
            
        if install_wheel(wheel):
            success_count += 1
        processed.add(wheel_name)
    
    print(f"\nInstalled {success_count} out of {len(wheels)} wheels successfully")