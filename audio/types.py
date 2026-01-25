"""
Shared audio data structures and types.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import time


@dataclass
class AudioFrame:
    """Represents a single audio frame with metadata."""
    pcm: np.ndarray  # int16 PCM data
    sample_rate: int
    timestamp: float  # Time when frame was captured/generated
    
    def to_bytes(self) -> bytes:
        """Convert PCM array to bytes."""
        return self.pcm.astype(np.int16).tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes, sample_rate: int, timestamp: Optional[float] = None) -> 'AudioFrame':
        """Create AudioFrame from bytes."""
        if timestamp is None:
            timestamp = time.time()
        pcm = np.frombuffer(data, dtype=np.int16)
        return cls(pcm=pcm, sample_rate=sample_rate, timestamp=timestamp)


class StopToken:
    """Token to signal immediate stop and flush of audio playback."""
    pass


# Event types for mic pipeline
SPEECH_START = "speech_start"
SPEECH_FRAME = "speech_frame"
SPEECH_END = "speech_end"
BARGE_IN = "barge_in"


@dataclass
class MicEvent:
    """Event from microphone pipeline."""
    event_type: str  # SPEECH_START, SPEECH_FRAME, SPEECH_END, BARGE_IN
    frame: Optional[AudioFrame] = None  # For SPEECH_FRAME events
    timestamp: Optional[float] = None
