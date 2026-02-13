# Voice Chatbot

A local voice-based conversational assistant. Speak into your microphone; the app transcribes your speech, sends it to a language model running in [LM Studio](https://lmstudio.ai/), and speaks the reply back using text-to-speech.

## How it works

1. **Listen** — Records from the microphone until a pause (configurable silence).
2. **Transcribe** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) turns speech into text.
3. **Respond** — The transcript is sent to an LLM via the OpenAI-compatible API (default: LM Studio at `http://localhost:1234/v1`).
4. **Speak** — The streamed reply is split into sentences and spoken with [Piper](https://github.com/rhasspy/piper) TTS.

Low-confidence transcriptions trigger short prompts like “Pardon?” or “Could you please repeat?” instead of calling the model.

## Project layout

| Package / area | Role |
|----------------|------|
| `speech_to_text` | Whisper-based transcription (`SpeechToText`) |
| `text_to_speech` | Piper TTS (`TextToSpeech`) and a Qwen3-TTS demo |
| `language_model` | LM Studio client (`LmStudioClient`, `LlmChat`) |
| `audio` | Recording, playback, and UI sounds |

## Requirements

- Python 3.x
- [LM Studio](https://lmstudio.ai/) running with a model loaded and the local server started (default port 1234)
- Microphone and speakers/headphones
- GPU recommended for Whisper and Piper (CUDA)

See `requirements.txt` for Python dependencies. Piper voices are separate; for example:

```bash
python -m piper.download_voices en_US-kristin-medium
```

## Run

From the project root:

```bash
python main.py
```

Then speak when the app is listening; use Ctrl+C to exit.

## Work in progress

- **Barge-in** — Planned support for interrupting the assistant while it is speaking (e.g. stop TTS and re-listen when the user starts talking).
- **AEC (acoustic echo cancellation)** — Experiments in `audio/aec_test.py` using an AEC pipeline (e.g. WebRTC-style processing) to reduce speaker bleed-through into the microphone when the assistant is playing audio.
- **Speaker recognition** — Experiments in `speakers/speaker_recognition.py` using [SpeechBrain](https://speechbrain.github.io/) (SepFormer for source separation, ECAPA-TDNN for speaker embeddings) for multi-speaker separation and speaker identification.

## License

See repository for license information.
