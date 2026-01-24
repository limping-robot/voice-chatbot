"""
WAV file playback utilities.
"""

import logging
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
except ImportError:
    # Fallback to scipy.io.wavfile if soundfile not available
    from scipy.io import wavfile
    sf = None

logger = logging.getLogger(__name__)


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
