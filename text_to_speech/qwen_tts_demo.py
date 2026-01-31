"""
Demonstration of Qwen3-TTS using the smallest model (Qwen3-TTS-12Hz-0.6B-Base).

This model supports voice cloning from a reference audio clip.
"""

import torch
import soundfile as sf
import sounddevice as sd
from qwen_tts import Qwen3TTSModel


def play_audio(wavs, sr):
    """
    Play audio waveforms using sounddevice.
    
    Args:
        wavs: Audio waveform(s) to play (can be a single array or list of arrays)
        sr: Sample rate of the audio
    """
    for wav in wavs:
        sd.play(wav, sr)
        sd.wait()


def main():
    print("Loading Qwen3-TTS-12Hz-0.6B-CustomVoice model...")
    print("(This may take a moment on first run as the model downloads)")
    
    # Load the smallest Qwen3-TTS model
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="auto",  # Automatically uses GPU if available, otherwise CPU
        dtype=torch.bfloat16,  # Use bfloat16 for efficiency (or torch.float16)
        attn_implementation="sdpa",  # Uncomment if you have flash-attn installed
    )
    
    print("Model loaded successfully!")

    print("torch.cuda.is_available:", torch.cuda.is_available())
    
    print("Supported speakers:")

    for speaker in model.get_supported_speakers():
        text = "Hello! This is Qwen3-TTS speaking with a preset voice. Tell me, tell me! Do you like it? Do you like my fucking voice?"
        language = "English"          # pick from get_supported_languages()
        speaker = speaker             # pick from get_supported_speakers()
        # instruct = "Relaxed, clear, slow pace, friendly and engaging."  # optional (style control)

        print(f"- {speaker}")

        wavs, sr = model.generate_custom_voice(
            text,
            language=language,
            speaker=speaker,
            #instruct=instruct,
        )

        play_audio(wavs, sr)

if __name__ == "__main__":
    main()
