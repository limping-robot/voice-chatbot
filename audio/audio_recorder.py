import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd

# Audio recording settings
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100  # Process audio in 100ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Silence detection settings
SILENCE_THRESHOLD = 0.005  # RMS threshold for silence detection

class AudioRecorder:
    """
    Records audio from microphone.
    
    Can be used in two modes:
    1. Queue-based: Produces audio chunks to mic_audio_q (legacy mode)
    2. Direct: Use record_until_silence() for blocking recording
    """
    def __init__(
        self, 
        mic_audio_q: Optional[queue.Queue] = None
    ):
        self.mic_audio_q = mic_audio_q
        self.is_recording = False
        self.stream: Optional[sd.InputStream] = None
        self._recording_thread: Optional[threading.Thread] = None
        
        # Direct recording mode
        self._direct_recording_buffer: list[np.ndarray] = []
        self._direct_recording_active = False
        self._direct_recording_lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):  # pyright: ignore[reportUnusedParameter]
        """Callback function for sounddevice to capture audio chunks."""
        if status:
            print(f"Audio status: {status}", flush=True)
        
        # Convert to float32
        audio_chunk = indata[:, 0].astype(np.float32)
        
        # Direct recording mode
        with self._direct_recording_lock:
            if self._direct_recording_active:
                self._direct_recording_buffer.append(audio_chunk.copy())
        
        # Queue-based mode (legacy)
        if self.mic_audio_q is not None:
            # Calculate RMS and send to queue
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            self.mic_audio_q.put((audio_chunk, rms))
    
    def record_until_silence(self, silence_duration_sec: float = 1.5) -> list[np.ndarray]:
        """
        Blocking method to record audio until silence is detected.
        
        Returns:
            List of audio chunks (np.ndarray) recorded until silence.
        """
        with self._direct_recording_lock:
            self._direct_recording_active = True
            self._direct_recording_buffer = []
        
        # Start recording if not already started
        if not self.is_recording:
            self.start()
        
        # Wait for recording to start
        while self.stream is None:
            threading.Event().wait(0.1)
        
        audio_buffer = []
        initial_silence_start_time: Optional[float] = None  # When silence first detected
        silence_start_time: Optional[float] = None  # When we've confirmed 100ms of silence
        SILENCE_CONFIRMATION_MS = 0.1  # 100ms
        
        while True:
            with self._direct_recording_lock:
                if len(self._direct_recording_buffer) > 0:
                    chunk = self._direct_recording_buffer.pop(0)
                    rms = np.sqrt(np.mean(chunk ** 2))
                    
                    current_time = time.time()
                    
                    if rms > SILENCE_THRESHOLD:
                        # audio activity detected - reset all silence tracking
                        audio_buffer.append(chunk)
                        initial_silence_start_time = None
                        silence_start_time = None
                    else:
                        # Silence detected
                        if initial_silence_start_time is None:
                            # First time we detect silence
                            initial_silence_start_time = current_time
                        
                        # Check if we've had continuous silence for at least 100ms
                        if initial_silence_start_time is not None:
                            initial_silence_duration = current_time - initial_silence_start_time
                            
                            if initial_silence_duration >= SILENCE_CONFIRMATION_MS:
                                # We've confirmed at least 100ms of silence
                                if silence_start_time is None:
                                    # Start the actual silence timer now (from when we confirmed)
                                    silence_start_time = current_time
                                
                                # Check if we've had enough silence after confirmation
                                if silence_start_time is not None:
                                    silence_duration = current_time - silence_start_time
                                    if silence_duration >= silence_duration_sec and len(audio_buffer) > 0:
                                        # Silence long enough, return recorded audio
                                        break
                        
                        # Still add silence chunks (in case user continues speaking)
                        audio_buffer.append(chunk)
            
            threading.Event().wait(0.05)  # Small sleep to avoid busy-waiting
        
        with self._direct_recording_lock:
            self._direct_recording_active = False
            self._direct_recording_buffer = []
        
        return audio_buffer

    def start(self):
        """Start recording in a background thread."""
        if self.is_recording:
            return
        
        self.is_recording = True
        
        def _record_loop():
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    blocksize=CHUNK_SIZE,
                    callback=self._audio_callback,
                    dtype=np.float32
                ) as stream:
                    self.stream = stream
                    while self.is_recording:
                        threading.Event().wait(0.1)
            except Exception as e:
                print(f"Error in recorder: {e}", flush=True)
            finally:
                self.stream = None
        
        self._recording_thread = threading.Thread(target=_record_loop, daemon=True)
        self._recording_thread.start()
    
    def shutdown(self):
        """Shutdown recording thread."""
        self.is_recording = False
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
