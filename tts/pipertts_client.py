import time
from os.path import dirname, join
from piper import PiperVoice, SynthesisConfig

import sounddevice as sd

class TtsClient:
    def __init__(self):
        self.voice = PiperVoice.load(
            join(
                dirname(__file__), 
                join("piper_voices", "en_US-kristin-medium.onnx")
            ),
            use_cuda=True
        )

        self.syn_config = SynthesisConfig(
            volume=1.0,
            length_scale=0.9,
            noise_scale=1.0,  # more audio variation
            noise_w_scale=1.0,  # more speaking variation
            normalize_audio=False,  # use raw audio from voice
        )

    # Speak the text using the voice and synthesis config.
    # The text is synthesized into chunks, which are played sequentially.
    # The method blocks until the text is spoken.
    def speak(self, text):
        # strip whitespaces from text
        text = text.strip()
        if not text:
            return
        chunks = self.voice.synthesize(text, self.syn_config)
        for chunk in chunks:
            sd.play(chunk.audio_int16_array, chunk.sample_rate)
            time.sleep(0.2)
            sd.wait()

if __name__ == "__main__":
    tts = TtsClient()
    tts.speak("Hello World! This is a test of the TTS engine.")