import queue
import time
import threading
from typing import Optional

import sounddevice as sd
import numpy as np

# Audio settings (should match recorder)
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class AudioPlayer:
    """
    Plays audio chunks.
    
    Can be used in two modes:
    1. Queue-based: Consumes audio chunks from tts_audio_q (legacy mode)
    2. Direct: Use play_audio() for blocking playback
    """
    def __init__(self, tts_audio_q: Optional[queue.Queue] = None):
        self.tts_audio_q = tts_audio_q
        self.is_running = False
        self._player_thread: Optional[threading.Thread] = None
        self._stop_playing = False
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Blocking method to play audio directly.
        
        Args:
            audio_data: Audio data as np.ndarray (int16)
            sample_rate: Sample rate of the audio
        """
        # Play audio and wait for completion
        sd.play(audio_data, sample_rate)
        sd.wait()
        # add a bit of a pause
        time.sleep(0.2)

    def start(self):
        """Start playing audio in a background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_playing = False
        
        def _play_loop():
            while self.is_running:
                try:
                    # Get audio chunk from queue
                    audio_data, sample_rate = self.tts_audio_q.get(timeout=0.1)
                    
                    if not self._stop_playing:
                        sd.play(audio_data, sample_rate)
                        time.sleep(0.2)
                        if not self._stop_playing:
                            sd.wait()
                
                except queue.Empty:
                    # Timeout - continue loop to check is_running
                    continue
                except Exception as e:
                    print(f"Error in audio player: {e}", flush=True)
        
        self._player_thread = threading.Thread(target=_play_loop, daemon=True)
        self._player_thread.start()
    
    def stop(self):
        """Stop playing audio (interrupts current playback)."""
        self._stop_playing = True
        sd.stop()
    
    def shutdown(self):
        """Shutdown the player thread."""
        self.is_running = False
        self.stop()
        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=2.0)
