import time
import threading
import queue
from typing import Optional, Iterator
from os.path import dirname, join
from piper import PiperVoice, SynthesisConfig

import numpy as np
from audio.types import AudioFrame
from audio.aec import SAMPLE_RATE, AEC_SAMPLES_PER_FRAME

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TtsSynthesizer:
    """
    Synthesizes text to speech.
    
    Can be used in two modes:
    1. Queue-based: Consumes text events from text_in_q and produces audio chunks to tts_audio_q (legacy)
    2. Direct: Use synthesize_and_play() for blocking synthesis and playback
    """
    def __init__(self, text_in_q=None, audio_out_q=None):
        self.text_in_q = text_in_q
        self.tts_audio_q = audio_out_q
        self.voice = PiperVoice.load(
            join(
                dirname(__file__), 
                join("piper_voices", "en_US-kristin-medium.onnx")
            ),
            use_cuda=True
        )
        self.syn_config = SynthesisConfig(
            volume=1.0,
            length_scale=1.0,
            noise_scale=0.7,  # audio variation
            noise_w_scale=0.8,  # speaking variation
            normalize_audio=False,  # use raw audio from voice
        )
        self.is_running = False
        self._synthesizer_thread: Optional[threading.Thread] = None
    
    def synthesize_and_play(self, text: str, audio_player):
        """
        Blocking method to synthesize text and play it.
        
        Args:
            text: Text to synthesize and play
            audio_player: AudioPlayer instance to use for playback
        """
        text = text.strip()
        if not text:
            return
        
        # Synthesize text into audio chunks
        chunks = self.voice.synthesize(text, self.syn_config)
        for chunk in chunks:
            # Convert int16 array to numpy array
            audio_data = np.array(chunk.audio_int16_array, dtype=np.int16)
            # Play audio (blocks until playback completes)
            audio_player.play_audio(audio_data, chunk.sample_rate)
    
    def synthesize_stream(self, text: str) -> Iterator[AudioFrame]:
        """
        Synthesize text to speech and yield audio frames (non-blocking, no playback).
        
        Splits chunks into 10ms frames (160 samples at 16kHz) and resamples if needed.
        
        Args:
            text: Text to synthesize
        
        Yields:
            AudioFrame objects with sample_rate=16000 and frame_size=160
        """
        text = text.strip()
        if not text:
            return
        
        target_sample_rate = SAMPLE_RATE  # 16kHz
        frame_size = AEC_SAMPLES_PER_FRAME  # 160 samples (10ms at 16kHz)
        
        # Synthesize text into audio chunks
        chunks = self.voice.synthesize(text, self.syn_config)
        timestamp = time.time()
        
        for chunk in chunks:
            # Convert int16 array to numpy array
            audio_data = np.array(chunk.audio_int16_array, dtype=np.int16)
            source_sample_rate = chunk.sample_rate
            
            # Resample to target sample rate (16kHz) if needed
            if source_sample_rate != target_sample_rate:
                if HAS_SCIPY:
                    # Calculate number of samples after resampling
                    num_samples = int(len(audio_data) * target_sample_rate / source_sample_rate)
                    # Resample using scipy
                    audio_data_float = audio_data.astype(np.float32) / 32768.0
                    audio_data_resampled = signal.resample(audio_data_float, num_samples)
                    # Convert back to int16
                    audio_data = np.clip(audio_data_resampled * 32767.0, -32768, 32767).astype(np.int16)
                else:
                    # Simple linear interpolation fallback (not ideal but better than nothing)
                    indices = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * target_sample_rate / source_sample_rate))
                    audio_data_float = audio_data.astype(np.float32)
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data_float).astype(np.int16)
            
            # Split into 10ms frames (160 samples at 16kHz)
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
            
            # Update timestamp for next chunk
            timestamp += len(audio_data) / target_sample_rate

    def start(self):
        """Start synthesizing in a background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        
        def _synthesize_loop():
            while self.is_running:
                try:
                    # Get text event from queue
                    text_event = self.text_in_q.get(timeout=0.1)
                    text = text_event.get("text", "").strip()
                    
                    if not text:
                        continue
                    
                    # Synthesize text into audio chunks
                    chunks = self.voice.synthesize(text, self.syn_config)
                    for chunk in chunks:
                        if not self.is_running:
                            break
                        # Convert int16 array to numpy array and put in queue
                        audio_data = np.array(chunk.audio_int16_array, dtype=np.int16)
                        self.tts_audio_q.put((audio_data, chunk.sample_rate))
                
                except queue.Empty:
                    # Timeout - continue loop to check is_running
                    continue
                except Exception as e:
                    print(f"Error in TTS synthesizer: {e}", flush=True)
        
        self._synthesizer_thread = threading.Thread(target=_synthesize_loop, daemon=True)
        self._synthesizer_thread.start()
    
    def stop(self):
        """Stop synthesizing."""
        self.is_running = False
        if self._synthesizer_thread and self._synthesizer_thread.is_alive():
            self._synthesizer_thread.join(timeout=2.0)
