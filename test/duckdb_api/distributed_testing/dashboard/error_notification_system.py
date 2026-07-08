#!/usr/bin/env python3
"""
Sound notification system for error visualization dashboard.

This module provides a sound notification system that plays different sounds
based on error severity, helping users quickly identify critical issues.
"""

import os
import logging
import threading
from typing import Dict, Optional, Any, Union

# Third-party imports (conditionally imported)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    
logger = logging.getLogger("error_notification")

class SoundNotificationSystem:
    """
    Sound notification system for error visualization dashboard.
    
    This class handles playing error notification sounds based on severity level,
    with volume control and user preferences.
    """
    
    def __init__(
        self, 
        sound_directory: Optional[str] = None,
        default_volume: float = 0.7,
        enable_sounds: bool = True
    ):
        """
        Initialize the sound notification system.
        
        Args:
            sound_directory: Directory containing sound files (defaults to module directory)
            default_volume: Default volume level (0.0 to 1.0)
            enable_sounds: Whether sounds are enabled by default
        """
        # Set up sound directory
        if sound_directory is None:
            sound_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "static/sounds"
            )
        self.sound_directory = sound_directory
        
        # Map severity levels to sound files
        self.sounds: Dict[str, str] = {
            'system_critical': 'error-system-critical.mp3',
            'critical': 'error-critical.mp3',
            'warning': 'error-warning.mp3',
            'info': 'error-info.mp3',
            'default': 'error-notification.mp3'
        }
        
        # User preferences
        self.volume = max(0.0, min(1.0, default_volume))
        self.muted = not enable_sounds
        self._sound_lock = threading.Lock()
        self._current_sound = None
        
        # Initialize pygame for sound playback if available
        self._pygame_initialized = False
        if PYGAME_AVAILABLE and not self.muted:
            try:
                pygame.mixer.init()
                self._pygame_initialized = True
                logger.info("Pygame audio system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize pygame audio: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize the sound system if not already initialized.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._pygame_initialized:
            return True
            
        if not PYGAME_AVAILABLE:
            logger.warning("Pygame not available. Sound notifications disabled.")
            return False
            
        try:
            pygame.mixer.init()
            self._pygame_initialized = True
            logger.info("Pygame audio system initialized")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize pygame audio: {e}")
            return False
    
    def set_volume(self, volume: float) -> None:
        """
        Set the notification volume level.
        
        Args:
            volume: Volume level from 0.0 (muted) to 1.0 (maximum)
        """
        self.volume = max(0.0, min(1.0, volume))
        logger.debug(f"Notification volume set to {self.volume:.1f}")
        
        # Update pygame volume if initialized
        if self._pygame_initialized:
            pygame.mixer.music.set_volume(self.volume)
    
    def set_muted(self, muted: bool) -> None:
        """
        Set whether notifications are muted.
        
        Args:
            muted: True to mute notifications, False to enable
        """
        self.muted = bool(muted)
        logger.debug(f"Sound notifications {'muted' if self.muted else 'enabled'}")
    
    def validate_sound_files(self) -> Dict[str, bool]:
        """
        Validate that all sound files exist and are accessible.
        
        Returns:
            Dict mapping severity levels to file existence (True/False)
        """
        results = {}
        for severity, sound_file in self.sounds.items():
            sound_path = os.path.join(self.sound_directory, sound_file)
            results[severity] = os.path.exists(sound_path)
            
            if not results[severity]:
                logger.warning(f"Sound file for {severity} not found: {sound_path}")
        
        return results
    
    def notify(self, severity: str, error_data: Dict[str, Any]) -> bool:
        """
        Play a notification sound for the given error severity.
        
        Args:
            severity: Error severity level (system_critical, critical, warning, info)
            error_data: Error data dictionary
            
        Returns:
            bool: True if sound played successfully, False otherwise
        """
        # Check if sounds are disabled
        if self.muted:
            logger.debug(f"Sound notification for {severity} suppressed (muted)")
            return False
            
        # Get the sound file for this severity
        sound_file = self.sounds.get(severity, self.sounds.get('default'))
        if not sound_file:
            logger.warning(f"No sound file found for severity: {severity}")
            return False
            
        sound_path = os.path.join(self.sound_directory, sound_file)
        if not os.path.exists(sound_path):
            logger.warning(f"Sound file not found: {sound_path}")
            return False
        
        # Play the sound in a separate thread to avoid blocking
        threading.Thread(
            target=self._play_sound,
            args=(sound_path,),
            daemon=True
        ).start()
        
        logger.debug(f"Playing notification sound for {severity} error")
        return True
    
    def _play_sound(self, sound_path: str) -> None:
        """
        Play a sound file using pygame.
        
        Args:
            sound_path: Path to the sound file to play
        """
        if not PYGAME_AVAILABLE:
            return
            
        # Initialize pygame if needed
        if not self._pygame_initialized:
            if not self.initialize():
                return
        
        # Acquire lock to ensure only one sound plays at a time
        with self._sound_lock:
            try:
                # Stop any currently playing sound
                pygame.mixer.music.stop()
                
                # Load and play the new sound
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play()
                
                # Store current sound for debugging
                self._current_sound = sound_path
            except Exception as e:
                logger.error(f"Error playing sound {sound_path}: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources used by the sound system."""
        if PYGAME_AVAILABLE and self._pygame_initialized:
            try:
                pygame.mixer.quit()
                logger.debug("Pygame audio system cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up pygame audio: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create sound notification system
    sound_system = SoundNotificationSystem()
    
    # Validate sound files
    validation_results = sound_system.validate_sound_files()
    all_valid = all(validation_results.values())
    
    if not all_valid:
        print("Some sound files are missing:")
        for severity, exists in validation_results.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {severity}: {sound_system.sounds.get(severity)}")
        sys.exit(1)
    
    # Test all sounds
    print("Testing all notification sounds...")
    print("System Critical:", sound_system.notify("system_critical", {"message": "Test"}))
    time.sleep(1.5)
    print("Critical:", sound_system.notify("critical", {"message": "Test"}))
    time.sleep(1.5)
    print("Warning:", sound_system.notify("warning", {"message": "Test"}))
    time.sleep(1.5)
    print("Info:", sound_system.notify("info", {"message": "Test"}))
    
    # Test volume control
    print("\nTesting volume control...")
    print("Setting volume to 30%")
    sound_system.set_volume(0.3)
    print("Critical:", sound_system.notify("critical", {"message": "Test"}))
    time.sleep(1.5)
    
    print("Setting volume to 100%")
    sound_system.set_volume(1.0)
    print("Warning:", sound_system.notify("warning", {"message": "Test"}))
    time.sleep(1.5)
    
    # Test mute control
    print("\nTesting mute control...")
    print("Muting sounds")
    sound_system.set_muted(True)
    print("Critical:", sound_system.notify("critical", {"message": "Test"}))
    time.sleep(1.5)
    
    print("Unmuting sounds")
    sound_system.set_muted(False)
    print("Info:", sound_system.notify("info", {"message": "Test"}))
    
    # Clean up
    time.sleep(1.5)
    sound_system.cleanup()
    print("\nSound notification test complete")