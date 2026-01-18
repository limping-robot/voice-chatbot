import pyttsx3

class TtsClient:
    def __init__(self):
        print("Initializing TTS engine...")
        self.engine = pyttsx3.init() # object creation

        # RATE
        print("Setting speaking rate...")
        self.engine.setProperty('rate', 160)     # setting up new voice rate

        # VOLUME
        print("Setting volume...")
        self.engine.setProperty('volume',1.0)        # setting up volume level  between 0 and 1

        # VOICE
        print("Setting voice...")
        voices = self.engine.getProperty('voices')       # getting details of current voice
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} - {voice.id}")
        self.engine.setProperty('voice', voices[2].id)

    def speak(self, text):
        print("Speaking...")
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()

if __name__ == "__main__":
    tts = TtsClient()
    tts.speak("Hello World! This is a test of the TTS engine.")