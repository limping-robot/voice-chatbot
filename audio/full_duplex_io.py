"""
Full-duplex audio I/O for coordinated microphone input and speaker output.

This module provides a single full-duplex stream that handles both input and output
simultaneously, ensuring proper timing alignment for AEC.
"""

import queue
import threading
import time
from typing import Optional, Callable
from collections import deque

import numpy as np
import sounddevice as sd

from audio.types import AudioFrame, StopToken
from audio.aec import (
    AEC_SAMPLES_PER_FRAME, 
    SAMPLE_RATE, 
    DEVICE_BLOCK_MS,
    DEVICE_SAMPLES_PER_BLOCK
)


class FullDuplexIO:
    """
    Full-duplex audio I/O with coordinated input/output.
    
    Provides:
    - Continuous microphone frame stream (with timestamps)
    - Speaker output queue
    - Speaker reference ring buffer for AEC
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 1,
        frame_size: int = AEC_SAMPLES_PER_FRAME,
        speaker_ref_callback: Optional[Callable[[AudioFrame], None]] = None
    ):
        """
        Initialize full-duplex I/O.
        
        Args:
            sample_rate: Audio sample rate (default 16000)
            channels: Number of channels (default 1 for mono)
            frame_size: Frame size in samples (default 160 for 10ms at 16kHz)
            speaker_ref_callback: Optional callback for each speaker reference frame
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        
        # Queues
        self.mic_frame_queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=100)
        self.speaker_frame_queue: queue.Queue[AudioFrame | StopToken] = queue.Queue(maxsize=200)
        
        # Speaker reference ring buffer (for AEC alignment)
        # Store last ~500ms of speaker output (50 frames at 10ms each)
        self.speaker_ref_buffer_size = 50
        self.speaker_ref_buffer: deque[AudioFrame] = deque(maxlen=self.speaker_ref_buffer_size)
        self.speaker_ref_lock = threading.Lock()
        self.speaker_ref_callback = speaker_ref_callback
        
        # Stream state
        self.stream: Optional[sd.Stream] = None
        self.is_running = False
        self._stop_requested = False
        
        # Silence frame for when queue is empty
        self._silence_frame = np.zeros(frame_size, dtype=np.int16)
    
    def get_mic_frame(self, timeout: Optional[float] = None) -> Optional[AudioFrame]:
        """
        Get next microphone frame (blocking).
        
        Args:
            timeout: Optional timeout in seconds
        
        Returns:
            AudioFrame or None if timeout
        """
        try:
            return self.mic_frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def enqueue_speaker_frame(self, frame: AudioFrame):
        """
        Enqueue a frame for speaker output.
        
        Args:
            frame: AudioFrame to play
        """
        self.speaker_frame_queue.put(frame)
    
    def enqueue_stop(self):
        """Enqueue stop token to immediately stop and flush playback."""
        self.speaker_frame_queue.put(StopToken())
    
    def flush_speaker_queue(self):
        """Flush all queued speaker frames."""
        while not self.speaker_frame_queue.empty():
            try:
                self.speaker_frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_speaker_reference(self, delay_frames: int = 0) -> Optional[AudioFrame]:
        """
        Get speaker reference frame from ring buffer.
        
        Args:
            delay_frames: Number of frames to look back (for AEC alignment)
        
        Returns:
            AudioFrame or None if not available
        """
        with self.speaker_ref_lock:
            if len(self.speaker_ref_buffer) > delay_frames:
                # Return frame from delay_frames ago
                idx = len(self.speaker_ref_buffer) - 1 - delay_frames
                return self.speaker_ref_buffer[idx]
        return None
    
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Full-duplex audio callback."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        current_time = time.time()
        
        # Process input (microphone)
        # With int16 dtype, sounddevice provides int16 directly
        mic_block = indata[:, 0].astype(np.int16)
        
        # Handle variable frame sizes
        if len(mic_block) < DEVICE_SAMPLES_PER_BLOCK:
            mic_block = np.pad(mic_block, (0, DEVICE_SAMPLES_PER_BLOCK - len(mic_block)), mode='constant')
        elif len(mic_block) > DEVICE_SAMPLES_PER_BLOCK:
            mic_block = mic_block[:DEVICE_SAMPLES_PER_BLOCK]
        
        # Split into 10ms frames and push to mic queue
        frames_pushed = 0
        for i in range(0, len(mic_block), self.frame_size):
            frame_data = mic_block[i:i + self.frame_size]
            if len(frame_data) == self.frame_size:
                frame = AudioFrame(
                    pcm=frame_data.copy(),
                    sample_rate=self.sample_rate,
                    timestamp=current_time
                )
                try:
                    self.mic_frame_queue.put_nowait(frame)
                    frames_pushed += 1
                except queue.Full:
                    # Drop oldest frame if queue is full
                    try:
                        self.mic_frame_queue.get_nowait()
                        self.mic_frame_queue.put_nowait(frame)
                        frames_pushed += 1
                    except queue.Empty:
                        pass
        
        # Process output (speaker)
        out_block = np.zeros(DEVICE_SAMPLES_PER_BLOCK, dtype=np.int16)
        
        # Check for stop token first
        stop_requested = False
        try:
            item = self.speaker_frame_queue.get_nowait()
            if isinstance(item, StopToken):
                stop_requested = True
                self.flush_speaker_queue()
            else:
                # Put it back and process normally
                self.speaker_frame_queue.put(item)
        except queue.Empty:
            pass
        
        if not stop_requested:
            # Fill output block with frames from queue
            frames_needed = DEVICE_SAMPLES_PER_BLOCK // self.frame_size
            for i in range(frames_needed):
                try:
                    item = self.speaker_frame_queue.get_nowait()
                    if isinstance(item, StopToken):
                        stop_requested = True
                        break
                    
                    frame = item
                    start_idx = i * self.frame_size
                    end_idx = start_idx + self.frame_size
                    
                    # Handle frame size mismatch
                    if len(frame.pcm) == self.frame_size:
                        # Perfect match
                        out_block[start_idx:end_idx] = frame.pcm
                    elif len(frame.pcm) < self.frame_size:
                        # Frame too small - pad with zeros
                        padded = np.pad(frame.pcm, (0, self.frame_size - len(frame.pcm)), mode='constant')
                        out_block[start_idx:end_idx] = padded
                        logger.debug(f"Padded frame from {len(frame.pcm)} to {self.frame_size} samples")
                    else:
                        # Frame too large - take first part
                        out_block[start_idx:end_idx] = frame.pcm[:self.frame_size]
                        logger.debug(f"Truncated frame from {len(frame.pcm)} to {self.frame_size} samples")
                    
                    # Store in speaker reference buffer (always use exact frame size)
                    ref_frame = AudioFrame(
                        pcm=out_block[start_idx:end_idx].copy(),
                        sample_rate=self.sample_rate,  # Use stream sample rate, not frame sample rate
                        timestamp=current_time
                    )
                    with self.speaker_ref_lock:
                        self.speaker_ref_buffer.append(ref_frame)
                    
                    # Call callback if provided
                    if self.speaker_ref_callback:
                        self.speaker_ref_callback(ref_frame)
                except queue.Empty:
                    # Fill with silence if queue is empty
                    start_idx = i * self.frame_size
                    end_idx = start_idx + self.frame_size
                    out_block[start_idx:end_idx] = self._silence_frame
        
        # Write to output
        # With int16 dtype, sounddevice expects int16 directly (matches aec_test.py)
        outdata[:, 0] = out_block
    
    def start(self):
        """Start full-duplex audio stream."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_requested = False
        
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',  # Match aec_test.py - use int16 for direct assignment
                blocksize=DEVICE_SAMPLES_PER_BLOCK,
                latency='high',  # Improves stability
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info(f"Full-duplex stream started: {self.sample_rate}Hz, {self.channels}ch, frame_size={self.frame_size}")
        except Exception as e:
            logger.error(f"Error starting full-duplex stream: {e}", exc_info=True)
            self.is_running = False
            raise
    
    def stop(self):
        """Stop full-duplex audio stream."""
        self._stop_requested = True
        self.flush_speaker_queue()
        self.enqueue_stop()
        
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            self.stream = None
        
        self.is_running = False
    
    def shutdown(self):
        """Shutdown and cleanup."""
        self.stop()
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
