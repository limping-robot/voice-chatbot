import time
import threading
import queue
from typing import Optional
from os.path import dirname, join
from piper import PiperVoice, SynthesisConfig

import numpy as np


class TextToSpeech:
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
