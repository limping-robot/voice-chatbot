"""
Acoustic Echo Cancellation (AEC) using WebRTC AudioProcessing.

Provides a reusable AEC processor that can be integrated into the audio pipeline.
Matches the implementation from aec_test.py exactly.
"""

import logging
import numpy as np

try:
    from aec_audio_processing import AudioProcessor
except ImportError:
    AudioProcessor = None
    print("Warning: aec_audio_processing not available. AEC functionality will be disabled.")

logger = logging.getLogger(__name__)

# ====== Audio format ======
# Matches aec_test.py exactly
SR = 16000
CH = 1
SAMPLE_WIDTH_BYTES = 2  # int16

# WebRTC AEC likes 10 ms frames
AEC_FRAME_MS = 10
AEC_SAMPLES_PER_FRAME = SR * AEC_FRAME_MS // 1000  # 160
AEC_FRAME_BYTES = AEC_SAMPLES_PER_FRAME * CH * SAMPLE_WIDTH_BYTES  # 320 bytes

# Use a larger device blocksize for stability (reduce xruns/crackling)
# Note: Following aec_test.py exactly - DEVICE_BLOCK_MS=20 gives 320 samples (20ms at 16kHz)
# The comment in aec_test.py says "# 640" but the calculation gives 320 for 20ms
DEVICE_BLOCK_MS = 20
DEVICE_SAMPLES_PER_BLOCK = SR * DEVICE_BLOCK_MS // 1000  # 320 samples (20ms at 16kHz)
DEVICE_BLOCK_BYTES = DEVICE_SAMPLES_PER_BLOCK * CH * SAMPLE_WIDTH_BYTES  # 640 bytes
FRAMES_PER_DEVICE_BLOCK = DEVICE_SAMPLES_PER_BLOCK // AEC_SAMPLES_PER_FRAME  # 2 frames of 10ms = 20ms

# Export constants for use in other modules (using SR/CH for consistency with aec_test.py)
SAMPLE_RATE = SR
CHANNELS = CH


class AECProcessor:
    """
    Acoustic Echo Cancellation processor.
    
    Processes microphone input with speaker reference to remove echo.
    Matches the FullDuplexAEC implementation from aec_test.py.
    """
    
    def __init__(
        self,
        enable_aec: bool = True,
        enable_ns: bool = True,
        enable_agc: bool = True,
        sample_rate: int = SR,
        channels: int = CH
    ):
        """
        Initialize AEC processor.
        
        Args:
            enable_aec: Enable acoustic echo cancellation
            enable_ns: Enable noise suppression
            enable_agc: Enable automatic gain control
            sample_rate: Audio sample rate (default 16000)
            channels: Number of audio channels (default 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = AEC_SAMPLES_PER_FRAME
        
        if AudioProcessor is None:
            self.ap = None
            self.enabled = False
            logger.warning("AEC disabled - aec_audio_processing not available")
            return
            
        self.enabled = True
        self.ap = AudioProcessor(enable_aec=enable_aec, enable_ns=enable_ns, enable_agc=enable_agc)
        self.ap.set_stream_format(sample_rate, channels)
        self.ap.set_reverse_stream_format(sample_rate, channels)
        
        self._delay_set = False
        self._silence_frame = b"\x00" * AEC_FRAME_BYTES
    
    def set_stream_delay(self, delay_ms: int):
        """
        Set the stream delay for AEC alignment.
        
        Args:
            delay_ms: Delay in milliseconds (typically 10-200ms)
        """
        if self.ap is not None:
            self.ap.set_stream_delay(delay_ms)
            self._delay_set = True
    
    def auto_set_delay(self, input_latency_s: float, output_latency_s: float):
        """
        Automatically set delay based on stream latencies.
        Matches _maybe_set_delay_from_stream from aec_test.py.
        
        Args:
            input_latency_s: Input latency in seconds
            output_latency_s: Output latency in seconds
        """
        # sounddevice gives (input_latency, output_latency) in seconds
        delay_ms = int((input_latency_s + output_latency_s) * 1000)
        
        # Add one AEC frame to account for callback scheduling
        delay_ms += AEC_FRAME_MS
        
        # Clamp to a sane range
        delay_ms = max(10, min(delay_ms, 200))
        
        self.set_stream_delay(delay_ms)
        logger.info(f"[AEC] Using stream_delay_ms={delay_ms} (in_lat={input_latency_s:.3f}s out_lat={output_latency_s:.3f}s)")
    
    def _post_gain_with_limiter(
        self,
        frame_i16: np.ndarray,
        target_peak: float = 1.0,  # 0..1 (0.5 ~= -6 dBFS)
        max_gain: float = 8.0,
    ) -> np.ndarray:
        """
        Post-process frame to limit peak to target_peak and apply gain.
        Matches _post_gain_with_limiter from aec_test.py.
        
        Args:
            frame_i16: Audio frame (int16)
            target_peak: Target peak level (0-1)
            max_gain: Maximum gain multiplier
        
        Returns:
            Processed frame (int16)
        """
        x = frame_i16.astype(np.float32) / 32768.0
        peak = np.max(np.abs(x)) + 1e-9
        gain = min(max_gain, target_peak / peak)
        y = np.clip(x * gain, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16)
    
    def process(
        self,
        mic_frame: np.ndarray,
        speaker_ref_frame: np.ndarray
    ) -> np.ndarray:
        """
        Process a microphone frame with speaker reference to remove echo.
        Matches the processing logic from FullDuplexAEC.callback in aec_test.py.
        
        Args:
            mic_frame: Microphone input frame (int16, shape [samples])
            speaker_ref_frame: Speaker reference frame (int16, shape [samples])
        
        Returns:
            Cleaned audio frame (int16, shape [samples])
        """
        if not self.enabled or self.ap is None:
            # Passthrough if AEC not available
            return mic_frame
        
        # Ensure frames are int16
        if mic_frame.dtype != np.int16:
            mic_frame = mic_frame.astype(np.int16)
        if speaker_ref_frame.dtype != np.int16:
            speaker_ref_frame = speaker_ref_frame.astype(np.int16)
        
        # Ensure frames are exactly AEC_FRAME_SIZE (10ms)
        if len(mic_frame) != self.frame_size:
            if len(mic_frame) < self.frame_size:
                mic_frame = np.pad(mic_frame, (0, self.frame_size - len(mic_frame)), mode='constant')
            else:
                mic_frame = mic_frame[:self.frame_size]
        
        if len(speaker_ref_frame) != self.frame_size:
            if len(speaker_ref_frame) < self.frame_size:
                speaker_ref_frame = np.pad(speaker_ref_frame, (0, self.frame_size - len(speaker_ref_frame)), mode='constant')
            else:
                speaker_ref_frame = speaker_ref_frame[:self.frame_size]
        
        # Convert to bytes (exactly as in aec_test.py)
        play_bytes = speaker_ref_frame.tobytes()
        mic_bytes = mic_frame.tobytes()
        
        # Feed reverse stream (exact samples played) - matches aec_test.py
        self.ap.process_reverse_stream(play_bytes)
        
        # AEC process - matches aec_test.py
        clean_bytes = self.ap.process_stream(mic_bytes)
        
        # Convert back to numpy array
        clean_i16 = np.frombuffer(clean_bytes, dtype=np.int16)
        
        # Apply post-gain with limiter (matches aec_test.py)
        clean_i16 = self._post_gain_with_limiter(clean_i16)
        
        return clean_i16
