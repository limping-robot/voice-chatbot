import os
from typing import Any
import queue
import time

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# ---------------- Config ----------------
CUDA_AVAILABLE = torch.cuda.is_available()

TRANSCRIPTION_MODEL_NAME = os.getenv("TRANSCRIPTION", "base")
TRANSCRIPTION_DEVICE = os.getenv("TRANSCRIPTION_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")
TRANSCRIPTION_DATATYPE = "float32" if CUDA_AVAILABLE else "torch.int8"

# Audio recording settings
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100  # Process audio in 100ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
SILENCE_THRESHOLD = 0.002  # RMS threshold for silence detection
SILENCE_DURATION_SEC = 2.0  # Pause duration to trigger transcription


class SttClient:
    def __init__(self) -> None:
        self.asr = WhisperModel(
            TRANSCRIPTION_MODEL_NAME,
            device=TRANSCRIPTION_DEVICE,
            compute_type=TRANSCRIPTION_DATATYPE
        )
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_buffer = []

    def _audio_callback(self, indata, frames, time_info, status):  # pyright: ignore[reportUnusedParameter]
        """Callback function for sounddevice to capture audio chunks."""
        if status:
            print(f"Audio status: {status}", flush=True)
        # Convert to float32 and calculate RMS for silence detection
        audio_chunk = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        self.audio_queue.put((audio_chunk, rms))

    def _clamp(self, x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def _utterance_confidence(self, segments: list[Any]) -> dict[str, Any]:
        """
        Build a 0..1 confidence score using Whisper-style signals:
        - avg_logprob (higher is better; closer to 0)
        - no_speech_prob (lower is better)
        - compression_ratio (lower is better; high can indicate hallucination/repetition)

        Returns dict with score + diagnostics you can log/tune.
        """
        segs = list(segments)
        if not segs:
            return {"score": 0.0, "reason": "no_segments"}

        # Duration-weighted aggregation (more robust than simple mean)
        total_dur = 0.0
        w_logprob = 0.0
        w_nospeech = 0.0
        w_comp = 0.0

        for s in segs:
            dur = float(getattr(s, "end", 0.0) - getattr(s, "start", 0.0))
            if dur <= 0:
                dur = 0.2  # fallback weight
            total_dur += dur
            w_logprob += dur * float(getattr(s, "avg_logprob", -10.0))
            w_nospeech += dur * float(getattr(s, "no_speech_prob", 1.0))
            w_comp += dur * float(getattr(s, "compression_ratio", 999.0))

        avg_logprob = w_logprob / total_dur
        no_speech_prob = w_nospeech / total_dur
        compression_ratio = w_comp / total_dur

        # --- Map avg_logprob to 0..1 ---
        # Typical values: around -0.2 (good) to -2.0 (bad), but tune for your audio/model.
        # This linear mapping is easy to tune:
        #   -1.6 => 0
        #   -0.4 => 1
        logprob_score = self._clamp((avg_logprob + 1.6) / 1.2)

        # --- Penalize no_speech_prob ---
        # If Whisper thinks it's not speech, confidence should drop sharply.
        # Keep it gentle at low values, harsh after ~0.4.
        nospeech_penalty = 1.0
        if no_speech_prob > 0.15:
            # Smooth-ish curve: 0.15->~1.0, 0.6->~0.2
            nospeech_penalty = self._clamp(1.0 - (no_speech_prob - 0.15) / 0.55) ** 1.5

        # --- Penalize compression_ratio (hallucination/repetition) ---
        # Whisper often flags hallucinations with high compression ratios.
        # Common "okay" range is often ~1.0–2.0; suspicious above ~2.4–2.6.
        comp_penalty = 1.0
        if compression_ratio > 2.2:
            comp_penalty = self._clamp(1.0 - (compression_ratio - 2.2) / 1.0) ** 1.5  # 3.2->~0

        # Combine
        score = logprob_score * nospeech_penalty * comp_penalty
        score = self._clamp(score)

        reason = "ok"
        if score < 0.35:
            reason = "very_low_confidence"
        elif score < 0.6:
            reason = "low_confidence"

        return {
            "score": score,
            "reason": reason,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio,
            "logprob_score": logprob_score,
            "nospeech_penalty": nospeech_penalty,
            "comp_penalty": comp_penalty,
        }


    def listen(self, silence_duration_sec: float = SILENCE_DURATION_SEC) -> tuple[str, dict[str, Any]]:
        """
        Listen for speech from the microphone and transcribe when there's a pause
        of more than 'silence_duration_sec' seconds. Returns the transcribed text.
        """
        
        self.is_recording = True
        self.audio_buffer = []
        silence_start_time = None
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                callback=self._audio_callback,
                dtype=np.float32
            ):
                while self.is_recording:
                    try:
                        # Get audio chunk from queue (with timeout to allow checking is_recording)
                        audio_chunk, rms = self.audio_queue.get(timeout=0.1)
                        
                        current_time = time.time()
                        
                        # Check if audio level is above silence threshold
                        if rms > SILENCE_THRESHOLD:
                            # Speech detected
                            self.audio_buffer.append(audio_chunk)
                            silence_start_time = None
                        else:
                            # Silence detected
                            if silence_start_time is None:
                                silence_start_time = current_time
                            
                            # Check if silence has lasted long enough
                            silence_duration = current_time - silence_start_time
                            if silence_duration >= silence_duration_sec and len(self.audio_buffer) > 0:
                                # Pause detected, transcribe the buffered audio
                                self.is_recording = False
                                break
                            
                            # Still add silence chunks to buffer (in case user continues speaking)
                            self.audio_buffer.append(audio_chunk)
                    
                    except queue.Empty:
                        # Timeout - continue loop to check is_recording
                        continue
        
        except KeyboardInterrupt:
            print("\nRecording interrupted", flush=True)
            self.is_recording = False
        
        # Transcribe the buffered audio
        if len(self.audio_buffer) > 0:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Transcribe using faster-whisper
            segments, _ = self.asr.transcribe(
                audio_data,
                language="en",
                beam_size=5
            )
            
            # Collect all transcribed text

            segments = list(segments)
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text

        confidence_info = self._utterance_confidence(segments)

        return [transcribed_text.strip(), confidence_info]
