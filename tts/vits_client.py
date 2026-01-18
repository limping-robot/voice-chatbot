import torch
import sounddevice as sd
from transformers import VitsModel, AutoTokenizer

# You can find other language models on the Hugging Face Hub: facebook/mms-tts
MODEL_NAME = "facebook/mms-tts-eng"


class TtsClient:
    def __init__(self):
        # Load the model and tokenizer
        self.model = VitsModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Speak the text using the VITS model.
    # The text is synthesized into a waveform, which is then played.
    # The method blocks until the text is spoken.
    def speak(self, text):
        # Tokenize the text input
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Generate the speech waveform
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        # Convert the waveform to a format playable by sounddevice
        # The output is a float tensor, which needs to be converted to an integer format for typical playback.
        audio_data = output.numpy().squeeze()
        sampling_rate = self.model.config.sampling_rate
        
        # Convert to 16-bit PCM for broader compatibility
        audio_data_int16 = (audio_data * 32767).astype("int16")
        
        # Play the audio
        sd.play(audio_data_int16, sampling_rate)
        sd.wait()


if __name__ == "__main__":
    tts = TtsClient()
    tts.speak("Hello World! This is a test of the VITS TTS engine.")