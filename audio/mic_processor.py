"""
Microphone processor with AEC, VAD, and barge-in detection.

Processes microphone input continuously, applies AEC, runs VAD,
and emits events for the orchestrator.
"""

import queue
import threading
import time
import logging
from typing import Optional, Callable

import numpy as np

from audio.types import AudioFrame, MicEvent, SPEECH_START, SPEECH_FRAME, SPEECH_END, BARGE_IN
from audio.aec import AECProcessor, AEC_SAMPLES_PER_FRAME, SAMPLE_RATE

logger = logging.getLogger(__name__)

try:
    import webrtcvad
except ImportError:
    webrtcvad = None
    logger.warning("webrtcvad not available. VAD will use simple RMS threshold.")


class MicProcessor:
    """
    Processes microphone input with AEC, VAD, and barge-in detection.
    
    Emits events:
    - speech_start: VAD detected speech beginning
    - speech_frame: Frame containing speech
    - speech_end: VAD detected speech ending
    - barge_in: Speech detected during assistant playback
    """
    
    def __init__(
        self,
        mic_frame_queue: queue.Queue[AudioFrame],
        event_queue: queue.Queue[MicEvent],
        speaker_ref_getter: Callable[[], Optional[AudioFrame]],
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = AEC_SAMPLES_PER_FRAME,
        enable_aec: bool = True,
        barge_in_threshold_ms: int = 150,  # Speech must persist for this long
        speech_end_silence_ms: int = 500,  # Silence duration to trigger speech_end
    ):
        """
        Initialize microphone processor.
        
        Args:
            mic_frame_queue: Queue of microphone frames from FullDuplexIO
            event_queue: Queue to emit events to orchestrator
            speaker_ref_getter: Function to get aligned speaker reference frame
            sample_rate: Audio sample rate
            frame_size: Frame size in samples
            enable_aec: Enable acoustic echo cancellation
            barge_in_threshold_ms: Minimum speech duration to trigger barge-in
            speech_end_silence_ms: Silence duration to trigger speech_end
        """
        self.mic_frame_queue = mic_frame_queue
        self.event_queue = event_queue
        self.speaker_ref_getter = speaker_ref_getter
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.barge_in_threshold_ms = barge_in_threshold_ms
        self.speech_end_silence_ms = speech_end_silence_ms
        
        # AEC
        self.aec = AECProcessor(enable_aec=enable_aec, sample_rate=sample_rate) if enable_aec else None
        
        # VAD
        if webrtcvad is not None:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3, 2 is moderate
                logger.info("Using WebRTC VAD")
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC VAD: {e}, using RMS fallback")
                self.vad = None
                self.vad_threshold = 0.003
        else:
            self.vad = None
            # Fallback: RMS-based VAD
            # Increased threshold to reduce false positives from background noise
            self.vad_threshold = 0.01  # Increased from 0.003 to be less sensitive
            logger.info(f"Using RMS-based VAD (threshold={self.vad_threshold})")
        
        # State
        self.is_running = False
        self._processor_thread: Optional[threading.Thread] = None
        
        # VAD state
        self._in_speech = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        
        # Barge-in detection
        self._barge_in_speech_start: Optional[float] = None
        
        # Utterance buffer (for STT)
        self._utterance_frames: list[AudioFrame] = []
        self._utterance_lock = threading.Lock()
    
    def _is_speech(self, frame: np.ndarray) -> bool:
        """
        Determine if frame contains speech using VAD.
        
        Args:
            frame: Audio frame (int16 for WebRTC, float32 for RMS)
        
        Returns:
            True if speech detected
        """
        if self.vad is not None:
            # WebRTC VAD expects 10, 20, or 30ms frames at 8/16/32/48kHz
            # We use 10ms at 16kHz which is supported
            if frame.dtype != np.int16:
                frame = frame.astype(np.int16)
            frame_bytes = frame.tobytes()
            try:
                return self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception as e:
                # Fallback to RMS if VAD fails
                logger.debug(f"VAD error: {e}, falling back to RMS")
                pass
        
        # RMS-based fallback
        # Convert to float32 if needed
        if frame.dtype == np.int16:
            frame_float = frame.astype(np.float32) / 32768.0
        else:
            frame_float = frame.astype(np.float32)
        
        rms = np.sqrt(np.mean(frame_float ** 2))
        is_speech = rms > self.vad_threshold
        
        # Debug logging periodically
        if not hasattr(self, '_rms_log_count'):
            self._rms_log_count = 0
        if self._rms_log_count < 50 or (self._rms_log_count % 500 == 0):
            logger.info(f"VAD: RMS={rms:.6f}, threshold={self.vad_threshold:.6f}, is_speech={is_speech}")
            self._rms_log_count += 1
        else:
            self._rms_log_count += 1
        
        return is_speech
    
    def _process_frame(self, mic_frame: AudioFrame, orchestrator_state: str):
        """
        Process a single microphone frame.
        
        Args:
            mic_frame: Raw microphone frame
            orchestrator_state: Current orchestrator state ("LISTENING", "THINKING", "SPEAKING")
        """
        # Get speaker reference for AEC
        speaker_ref = self.speaker_ref_getter()
        
        # Apply AEC
        if self.aec is not None and self.aec.enabled and speaker_ref is not None:
            try:
                clean_frame = self.aec.process(mic_frame.pcm, speaker_ref.pcm)
            except Exception as e:
                logger.debug(f"AEC processing error: {e}, using raw mic frame")
                clean_frame = mic_frame.pcm
        else:
            # AEC disabled or no reference - use raw mic frame
            clean_frame = mic_frame.pcm
        
        # Convert int16 to float32 for VAD (if needed)
        # VAD expects int16, but RMS calculation works better with float32
        if clean_frame.dtype == np.int16:
            clean_frame_float = clean_frame.astype(np.float32) / 32768.0
        else:
            clean_frame_float = clean_frame
        
        # Create cleaned frame (ensure int16 for storage)
        if clean_frame.dtype != np.int16:
            clean_frame_int16 = (clean_frame * 32767.0).astype(np.int16)
        else:
            clean_frame_int16 = clean_frame
        
        clean_audio_frame = AudioFrame(
            pcm=clean_frame_int16,
            sample_rate=mic_frame.sample_rate,
            timestamp=mic_frame.timestamp
        )
        
        # Run VAD (convert to int16 for WebRTC VAD if needed, or use float32 for RMS)
        if self.vad is not None:
            # WebRTC VAD needs int16
            if clean_frame.dtype != np.int16:
                vad_frame = (clean_frame_float * 32767.0).astype(np.int16)
            else:
                vad_frame = clean_frame
            is_speech_now = self._is_speech(vad_frame)
        else:
            # RMS-based VAD uses float32
            is_speech_now = self._is_speech(clean_frame_float)
        current_time = time.time()
        
        # Handle speech state transitions
        if is_speech_now:
            if not self._in_speech:
                # Speech start
                self._in_speech = True
                self._speech_start_time = current_time
                self._silence_start_time = None
                
                # Clear utterance buffer
                with self._utterance_lock:
                    self._utterance_frames = []
                
                # Emit speech_start event
                self.event_queue.put(MicEvent(
                    event_type=SPEECH_START,
                    timestamp=current_time
                ))
            
            # Create cleaned frame for storage
            clean_audio_frame = AudioFrame(
                pcm=clean_frame.copy(),
                sample_rate=mic_frame.sample_rate,
                timestamp=mic_frame.timestamp
            )
            
            # Add to utterance buffer
            with self._utterance_lock:
                self._utterance_frames.append(clean_audio_frame)
            
            # Emit speech_frame event
            self.event_queue.put(MicEvent(
                event_type=SPEECH_FRAME,
                frame=clean_audio_frame,
                timestamp=current_time
            ))
            
            # Check for barge-in (if in SPEAKING state)
            if orchestrator_state == "SPEAKING":
                if self._barge_in_speech_start is None:
                    self._barge_in_speech_start = current_time
                else:
                    # Check if speech has persisted long enough
                    speech_duration = (current_time - self._barge_in_speech_start) * 1000
                    if speech_duration >= self.barge_in_threshold_ms:
                        # Emit barge-in event
                        self.event_queue.put(MicEvent(
                            event_type=BARGE_IN,
                            timestamp=current_time
                        ))
                        # Reset barge-in tracking
                        self._barge_in_speech_start = None
        else:
            # No speech
            if self._in_speech:
                # Create cleaned frame for storage
                clean_audio_frame = AudioFrame(
                    pcm=clean_frame.copy(),
                    sample_rate=mic_frame.sample_rate,
                    timestamp=mic_frame.timestamp
                )
                
                # Still add silence frames to utterance buffer (in case user continues)
                with self._utterance_lock:
                    self._utterance_frames.append(clean_audio_frame)
                
                # Check if silence has persisted long enough
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                
                silence_duration = (current_time - self._silence_start_time) * 1000
                if silence_duration >= self.speech_end_silence_ms:
                    # Speech end
                    self._in_speech = False
                    self._speech_start_time = None
                    self._silence_start_time = None
                    self._barge_in_speech_start = None
                    
                    # Emit speech_end event
                    logger.info(f"Speech end detected after {silence_duration:.0f}ms of silence")
                    self.event_queue.put(MicEvent(
                        event_type=SPEECH_END,
                        timestamp=current_time
                    ))
            else:
                # Reset barge-in tracking if not in speech
                self._barge_in_speech_start = None
    
    def get_utterance_frames(self) -> list[AudioFrame]:
        """
        Get and clear the current utterance buffer.
        
        Returns:
            List of AudioFrame objects for the current utterance
        """
        with self._utterance_lock:
            frames = self._utterance_frames.copy()
            self._utterance_frames = []
        return frames
    
    def start(self, orchestrator_state_getter: Callable[[], str]):
        """
        Start processing microphone frames.
        
        Args:
            orchestrator_state_getter: Function to get current orchestrator state
        """
        if self.is_running:
            return
        
        self.is_running = True
        
        frames_processed = 0
        
        def _process_loop():
            nonlocal frames_processed
            logger.info("Mic processor loop started")
            while self.is_running:
                try:
                    # Get microphone frame
                    mic_frame = self.mic_frame_queue.get(timeout=0.1)
                    frames_processed += 1
                    
                    # Log every 100 frames (roughly every second at 10ms frames)
                    if frames_processed % 100 == 0:
                        logger.info(f"Mic processor: processed {frames_processed} frames, queue size: {self.mic_frame_queue.qsize()}, in_speech={self._in_speech}")
                    
                    # Get current orchestrator state
                    state = orchestrator_state_getter()
                    
                    # Process frame
                    self._process_frame(mic_frame, state)
                
                except queue.Empty:
                    # No frames available - this is normal, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error in mic processor: {e}", exc_info=True)
        
        self._processor_thread = threading.Thread(target=_process_loop, daemon=True)
        self._processor_thread.start()
        logger.info("Mic processor started")
    
    def stop(self):
        """Stop processing."""
        self.is_running = False
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=2.0)
    
    def shutdown(self):
        """Shutdown and cleanup."""
        self.stop()
