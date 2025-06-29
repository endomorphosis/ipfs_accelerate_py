#!/usr/bin/env python3
"""
Test script to verify the audio notification sound files.

This script checks that all required sound files exist, have appropriate file sizes,
and performs a simple validation of the MP3 files to ensure they are properly formatted.
"""

import os
import sys
import subprocess

def check_file_existence(sound_files):
    """Check that all required sound files exist and have appropriate file sizes."""
    all_files_exist = True
    print("Checking sound files existence and sizes:")
    
    for file_name in sound_files:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 1000:  # Basic size check (should be at least 1KB)
                print(f"✅ {file_name}: Found ({file_size} bytes)")
            else:
                print(f"❌ {file_name}: Found but suspiciously small ({file_size} bytes)")
                all_files_exist = False
        else:
            print(f"❌ {file_name}: Not found")
            all_files_exist = False
    
    return all_files_exist

def validate_mp3_files(sound_files):
    """Validate that the MP3 files are properly formatted."""
    all_files_valid = True
    print("\nValidating MP3 file format:")
    
    for file_name in sound_files:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        if not os.path.exists(file_path):
            continue
            
        try:
            # Try to get MP3 metadata using ffmpeg
            result = subprocess.run(
                ["ffmpeg", "-i", file_path, "-f", "null", "-"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=False
            )
            
            # Check ffmpeg output for MP3 information
            output = result.stderr.decode()
            if "Audio: mp3" in output and result.returncode == 0:
                # Extract duration if available
                duration = "unknown"
                for line in output.split('\n'):
                    if "Duration:" in line:
                        duration = line.split("Duration:")[1].split(",")[0].strip()
                        break
                print(f"✅ {file_name}: Valid MP3 (Duration: {duration})")
            else:
                print(f"❌ {file_name}: Invalid MP3 format or corrupted file")
                all_files_valid = False
                
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"❌ {file_name}: Validation failed: {str(e)}")
            all_files_valid = False
    
    return all_files_valid

def main():
    """Run the sound file validation tests."""
    print("Error Notification Sound System Test\n")
    
    sound_files = [
        "error-system-critical.mp3",
        "error-critical.mp3",
        "error-warning.mp3",
        "error-info.mp3",
        "error-notification.mp3"
    ]
    
    # Check that all files exist
    files_exist = check_file_existence(sound_files)
    
    # Validate MP3 format
    files_valid = validate_mp3_files(sound_files)
    
    # Summary
    print("\nTest Summary:")
    if files_exist and files_valid:
        print("✅ All sound files are present and valid!")
        return 0
    else:
        print("❌ Some sound files are missing or invalid")
        return 1

if __name__ == "__main__":
    sys.exit(main())