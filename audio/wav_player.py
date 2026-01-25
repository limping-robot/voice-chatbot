"""
WAV file playback utilities.

Supports both legacy blocking playback and new frame-based enqueueing.
"""

import logging
from pathlib import Path
from typing import Iterator
import numpy as np

try:
    import soundfile as sf
except ImportError:
    # Fallback to scipy.io.wavfile if soundfile not available
    from scipy.io import wavfile
    sf = None

from audio.types import AudioFrame
from audio.aec import SAMPLE_RATE, AEC_SAMPLES_PER_FRAME

logger = logging.getLogger(__name__)

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available, WAV resampling will be limited")


def play_wav_file(audio_player, filename: str):
    """
    Play a WAV file. This function blocks until playback completes.
    
    Args:
        audio_player: AudioPlayer instance to use for playback
        filename: Name of the WAV file in the resources directory
    """
    # Get the directory of this file (audio/) and construct path to resources
    audio_dir = Path(__file__).parent
    wav_path = audio_dir / "resources" / filename
    
    if not wav_path.exists():
        logger.warning(f"Sound file not found: {wav_path}")
        return
    
    try:
        if sf is not None:
            # Use soundfile (preferred)
            audio_data, sample_rate = sf.read(str(wav_path), dtype='int16')
        else:
            # Fallback to scipy.io.wavfile
            sample_rate, audio_data = wavfile.read(str(wav_path))
            # Ensure int16 format
            if audio_data.dtype != np.int16:
                # Normalize and convert to int16
                audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
        
        # Play the audio
        audio_player.play_audio(audio_data, sample_rate)
    except Exception as e:
        logger.error(f"Error playing sound file {filename}: {e}", exc_info=True)


def play_listening_sound(audio_player):
    """
    Play the WAV file that indicates listening has started.
    """
    play_wav_file(audio_player, "mixkit-cool-interface-click-tone-2568.wav")

def play_transcribed_sound(audio_player):
    """
    Play the WAV file that indicates transcription has finished.
    """
    play_wav_file(audio_player, "mixkit-modern-technology-select-3124.wav")

def play_prompting_sound(audio_player):
    """
    Play the WAV file that indicates prompting has started.
    """
    play_wav_file(audio_player, "mixkit-select-click-1109.wav")


def load_wav_frames(filename: str, frame_size: int = AEC_SAMPLES_PER_FRAME) -> Iterator[AudioFrame]:
    """
    Load WAV file and yield AudioFrame objects.
    
    Resamples to target sample rate (16kHz) if needed and ensures correct frame size.
    
    Args:
        filename: Name of the WAV file in the resources directory
        frame_size: Frame size in samples (default 160 for 10ms at 16kHz)
    
    Yields:
        AudioFrame objects with sample_rate=16000 and correct frame size
    """
    import time
    
    # Get the directory of this file (audio/) and construct path to resources
    audio_dir = Path(__file__).parent
    wav_path = audio_dir / "resources" / filename
    
    if not wav_path.exists():
        logger.warning(f"Sound file not found: {wav_path}")
        return
    
    try:
        if sf is not None:
            # Use soundfile (preferred)
            audio_data, source_sample_rate = sf.read(str(wav_path), dtype='int16')
        else:
            # Fallback to scipy.io.wavfile
            source_sample_rate, audio_data = wavfile.read(str(wav_path))
            # Ensure int16 format
            if audio_data.dtype != np.int16:
                # Normalize and convert to int16
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Resample to target sample rate (16kHz) if needed
        target_sample_rate = SAMPLE_RATE
        if source_sample_rate != target_sample_rate:
            if HAS_SCIPY:
                # Calculate number of samples after resampling
                num_samples = int(len(audio_data) * target_sample_rate / source_sample_rate)
                # Resample using scipy
                audio_data_float = audio_data.astype(np.float32) / 32768.0
                audio_data_resampled = signal.resample(audio_data_float, num_samples)
                # Convert back to int16
                audio_data = np.clip(audio_data_resampled * 32767.0, -32768, 32767).astype(np.int16)
                logger.info(f"Resampled WAV from {source_sample_rate}Hz to {target_sample_rate}Hz ({len(audio_data)} -> {num_samples} samples)")
            else:
                logger.warning(f"WAV file has sample rate {source_sample_rate}Hz but target is {target_sample_rate}Hz. "
                             f"scipy not available for resampling. Audio may sound incorrect.")
        else:
            logger.debug(f"WAV file already at target sample rate {target_sample_rate}Hz")
        
        # Yield frames with target sample rate
        timestamp = time.time()
        for i in range(0, len(audio_data), frame_size):
            frame_data = audio_data[i:i + frame_size]
            if len(frame_data) == frame_size:
                yield AudioFrame(
                    pcm=frame_data.copy(),
                    sample_rate=target_sample_rate,
                    timestamp=timestamp + (i / target_sample_rate)
                )
            elif len(frame_data) > 0:
                # Pad last frame if needed
                padded = np.pad(frame_data, (0, frame_size - len(frame_data)), mode='constant')
                yield AudioFrame(
                    pcm=padded,
                    sample_rate=target_sample_rate,
                    timestamp=timestamp + (i / target_sample_rate)
                )
    
    except Exception as e:
        logger.error(f"Error loading sound file {filename}: {e}", exc_info=True)
