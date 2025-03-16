#!/usr/bin/env python3
"""
Generate sound files for error notifications.

This script generates four different sound files for different error severities:
1. error-system-critical.mp3: Highest priority alert sound for system-level critical errors
2. error-critical.mp3: High priority alert sound for critical errors
3. error-warning.mp3: Medium priority alert sound for warning level errors
4. error-info.mp3: Low priority notification sound for informational errors

It also creates a general error-notification.mp3 file that is referenced in the JavaScript code.
"""

import numpy as np
from scipy.io import wavfile

def generate_system_critical_sound(filename, duration=1.0):
    """Generate the highest priority alert sound for system-level critical errors."""
    # Use a more complex sound pattern with increasing urgency
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Use multiple frequencies with a rising pattern
    frequency1 = 880  # A5
    frequency2 = 1046.5  # C6
    frequency3 = 1318.5  # E6
    
    # Create rising tones with amplitude modulation
    segment_duration = duration / 3
    segment1 = t < segment_duration
    segment2 = (t >= segment_duration) & (t < 2 * segment_duration)
    segment3 = t >= 2 * segment_duration
    
    # Create three tones that build in intensity
    tone1 = np.sin(2 * np.pi * frequency1 * t) * segment1
    tone2 = np.sin(2 * np.pi * frequency2 * t) * segment2
    tone3 = np.sin(2 * np.pi * frequency3 * t) * segment3
    
    # Combine tones with crossfade
    signal = tone1 + tone2 + tone3
    
    # Add an urgent pulsing effect that speeds up
    pulse_rate = 4 + 12 * t/duration  # Pulse rate increases from 4Hz to 16Hz
    pulse = 0.7 + 0.3 * np.sin(2 * np.pi * pulse_rate * t)
    signal = signal * pulse
    
    # Add a subtle harmonic for richness
    harmonic = 0.2 * np.sin(2 * np.pi * 2 * frequency1 * t) * np.exp(-1 * t/duration)
    signal = signal + harmonic
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filename, sample_rate, signal)
    print(f"Generated {filename}")

def generate_critical_sound(filename, duration=0.7):
    """Generate a high priority alert sound for critical errors."""
    # Use a higher frequency with amplitude modulation for urgency
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Start with a higher frequency and then drop
    frequency1 = 880  # A5
    frequency2 = 440  # A4
    
    # Create two tones with amplitude modulation
    tone1 = np.sin(2 * np.pi * frequency1 * t) * np.exp(-3 * t)
    tone2 = np.sin(2 * np.pi * frequency2 * t) * (1 - np.exp(-5 * t))
    
    # Combine tones
    signal = 0.5 * tone1 + 0.5 * tone2
    
    # Add a pulsing effect
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * t)
    signal = signal * pulse
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filename, sample_rate, signal)
    print(f"Generated {filename}")

def generate_warning_sound(filename, duration=0.5):
    """Generate a medium priority alert sound for warning level errors."""
    # Use a medium frequency with a simple decay
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a medium frequency tone
    frequency = 660  # E5
    
    # Create tone with decay
    signal = np.sin(2 * np.pi * frequency * t) * np.exp(-4 * t)
    
    # Add a second tone for depth
    frequency2 = 330  # E4
    signal2 = np.sin(2 * np.pi * frequency2 * t) * np.exp(-2 * t)
    
    # Combine signals
    signal = 0.6 * signal + 0.4 * signal2
    
    # Normalize
    signal = 0.9 * signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filename, sample_rate, signal)
    print(f"Generated {filename}")

def generate_info_sound(filename, duration=0.3):
    """Generate a low priority notification sound for informational errors."""
    # Use a lower frequency with quick decay for non-intrusive notification
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a medium frequency tone
    frequency = 523.25  # C5
    
    # Create tone with quick decay
    signal = np.sin(2 * np.pi * frequency * t) * np.exp(-10 * t)
    
    # Normalize
    signal = 0.8 * signal / np.max(np.abs(signal))
    
    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)
    
    # Save to file
    wavfile.write(filename, sample_rate, signal)
    print(f"Generated {filename}")

def convert_wav_to_mp3(wav_filename, mp3_filename):
    """Convert WAV file to MP3 using ffmpeg."""
    import subprocess
    
    try:
        subprocess.run(
            ["ffmpeg", "-i", wav_filename, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_filename],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Converted {wav_filename} to {mp3_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {wav_filename} to {mp3_filename}: {e}")
        print(f"STDERR: {e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Using WAV files instead.")
        # Copy the WAV file to the MP3 filename as a fallback
        import shutil
        shutil.copy(wav_filename, mp3_filename)
        print(f"Copied {wav_filename} to {mp3_filename} as fallback")

def main():
    """Generate all sound files."""
    # Create output directory
    import os
    os.makedirs(".", exist_ok=True)
    
    # Generate WAV files
    generate_system_critical_sound("error-system-critical.wav")
    generate_critical_sound("error-critical.wav")
    generate_warning_sound("error-warning.wav")
    generate_info_sound("error-info.wav")
    
    # Convert WAV to MP3
    convert_wav_to_mp3("error-system-critical.wav", "error-system-critical.mp3")
    convert_wav_to_mp3("error-critical.wav", "error-critical.mp3")
    convert_wav_to_mp3("error-warning.wav", "error-warning.mp3")
    convert_wav_to_mp3("error-info.wav", "error-info.mp3")
    
    # Create a copy of the critical sound as the default notification sound
    import shutil
    try:
        shutil.copy("error-critical.mp3", "error-notification.mp3")
        print("Created error-notification.mp3 (copy of error-critical.mp3)")
    except Exception as e:
        print(f"Error creating error-notification.mp3: {e}")
    
    # Cleanup WAV files
    try:
        os.remove("error-system-critical.wav")
        os.remove("error-critical.wav")
        os.remove("error-warning.wav")
        os.remove("error-info.wav")
        print("Cleaned up WAV files")
    except Exception as e:
        print(f"Error cleaning up WAV files: {e}")

if __name__ == "__main__":
    main()