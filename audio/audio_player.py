import queue
import time
import threading
from typing import Optional, Callable

import sounddevice as sd
import numpy as np

from audio.types import AudioFrame, StopToken

# Audio settings (should match recorder)
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


class AudioPlayer:
    """
    Plays audio chunks.
    
    Can be used in two modes:
    1. Queue-based: Consumes audio frames from queue (primary mode for new architecture)
    2. Direct: Use play_audio() for blocking playback (legacy mode)
    """
    def __init__(
        self,
        tts_audio_q: Optional[queue.Queue] = None,
        frame_queue: Optional[queue.Queue] = None,
        on_frame_played: Optional[Callable[[AudioFrame], None]] = None
    ):
        """
        Initialize audio player.
        
        Args:
            tts_audio_q: Legacy queue for (audio_data, sample_rate) tuples
            frame_queue: New queue for AudioFrame objects (or StopToken)
            on_frame_played: Callback when a frame is actually played (for reference publishing)
        """
        self.tts_audio_q = tts_audio_q
        self.frame_queue = frame_queue
        self.on_frame_played = on_frame_played
        self.is_running = False
        self._player_thread: Optional[threading.Thread] = None
        self._stop_playing = False
    
    def enqueue_frame(self, frame: AudioFrame):
        """
        Enqueue an audio frame for playback.
        
        Args:
            frame: AudioFrame to play
        """
        if self.frame_queue is None:
            raise RuntimeError("frame_queue not initialized")
        self.frame_queue.put(frame)
    
    def enqueue_stop(self):
        """Enqueue stop token to immediately stop and flush playback."""
        if self.frame_queue is None:
            raise RuntimeError("frame_queue not initialized")
        self.frame_queue.put(StopToken())
    
    def flush(self):
        """Flush all queued audio frames."""
        if self.frame_queue is None:
            return
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Blocking method to play audio directly (legacy mode).
        
        Args:
            audio_data: Audio data as np.ndarray (int16)
            sample_rate: Sample rate of the audio
        """
        # Play audio and wait for completion
        sd.play(audio_data, sample_rate)
        sd.wait()
        # Note: Removed 0.2s pause for better barge-in responsiveness

    def start(self):
        """Start playing audio in a background thread (legacy queue mode)."""
        if self.is_running:
            return
        
        if self.frame_queue is not None:
            # New mode: frame queue is handled by FullDuplexIO
            return
        
        if self.tts_audio_q is None:
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
        self.flush()
    
    def shutdown(self):
        """Shutdown the player thread."""
        self.is_running = False
        self.stop()
        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=2.0)
