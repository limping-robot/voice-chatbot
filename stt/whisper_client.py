import os
from typing import Any, Optional
import queue
import time
import threading

import numpy as np
import torch
from faster_whisper import WhisperModel

# ---------------- Config ----------------
CUDA_AVAILABLE = torch.cuda.is_available()

TRANSCRIPTION_MODEL_NAME = os.getenv("TRANSCRIPTION", "base")
TRANSCRIPTION_DEVICE = os.getenv("TRANSCRIPTION_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")
TRANSCRIPTION_DATATYPE = "float32" if CUDA_AVAILABLE else "torch.int8"

# Silence detection settings
SILENCE_THRESHOLD = 0.003  # RMS threshold for silence detection
SILENCE_DURATION_SEC = 1.5  # Pause duration to trigger transcription


class SttTranscriber:
    """
    Transcribes audio to text.
    
    Can be used in two modes:
    1. Queue-based: Consumes audio chunks from mic_audio_q and produces text events to text_in_q (legacy)
    2. Direct: Use transcribe() for blocking transcription
    """
    def __init__(
        self, 
        mic_audio_q: Optional[queue.Queue] = None, 
        text_in_q: Optional[queue.Queue] = None, 
        silence_duration_sec: float = SILENCE_DURATION_SEC
    ):
        self.mic_audio_q = mic_audio_q
        self.text_in_q = text_in_q
        self.asr = WhisperModel(
            TRANSCRIPTION_MODEL_NAME,
            device=TRANSCRIPTION_DEVICE,
            compute_type=TRANSCRIPTION_DATATYPE
        )
        self.is_running = False
        self.audio_buffer = []
        self.silence_start_time: Optional[float] = None
        self.silence_duration_sec = silence_duration_sec
        self._transcriber_thread: Optional[threading.Thread] = None
        self._audio_recorder: Optional[Any] = None
    
    def transcribe(self, audio_recorder) -> tuple[str, dict]:
        """
        Blocking method to listen and transcribe audio until silence.
        
        Args:
            audio_recorder: AudioRecorder instance to use for recording
        
        Returns:
            Tuple of (transcribed_text: str, confidence_info: dict)
        """
        # Record audio until silence
        audio_chunks = audio_recorder.record_until_silence(self.silence_duration_sec)
        
        if not audio_chunks:
            return "", {"score": 0.0, "reason": "no_audio"}
        
        # Concatenate audio chunks
        audio_data = np.concatenate(audio_chunks)
        
        # Transcribe using faster-whisper
        segments, _ = self.asr.transcribe(
            audio_data,
            language="en",
            beam_size=5
        )
        
        segments = list(segments)
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text
        
        confidence_info = self._utterance_confidence(segments)
        transcribed_text = transcribed_text.strip()
        
        return transcribed_text, confidence_info

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

    def start(self):
        """Start transcribing in a background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.audio_buffer = []
        self.silence_start_time = None
        
        def _listen_to_audio_loop():
            while self.is_running:
                try:
                    # Get audio chunk from queue (with timeout to allow checking is_running)
                    audio_chunk, rms = self.mic_audio_q.get(timeout=0.1)
                    
                    current_time = time.time()
                    
                    # Check if audio level is above silence threshold
                    if rms > SILENCE_THRESHOLD:
                        # Speech detected
                        self.audio_buffer.append(audio_chunk)
                        self.silence_start_time = None
                    else:
                        # Silence detected
                        if self.silence_start_time is None:
                            self.silence_start_time = current_time
                        
                        # Check if silence has lasted long enough
                        if self.silence_start_time is not None:
                            silence_duration = current_time - self.silence_start_time
                            if silence_duration >= self.silence_duration_sec and len(self.audio_buffer) > 0:
                                # Pause detected, transcribe the buffered audio
                                audio_data = np.concatenate(self.audio_buffer)
                                
                                # Transcribe using faster-whisper
                                segments, _ = self.asr.transcribe(
                                    audio_data,
                                    language="en",
                                    beam_size=5
                                )
                                
                                segments = list(segments)
                                transcribed_text = ""
                                for segment in segments:
                                    transcribed_text += segment.text
                                
                                confidence_info = self._utterance_confidence(segments)
                                transcribed_text = transcribed_text.strip()
                                
                                # Only produce text event if we have valid text
                                if transcribed_text and any(c.isalpha() for c in transcribed_text):
                                    self.text_in_q.put({
                                        "text": transcribed_text,
                                        "confidence_info": confidence_info,
                                        "source": "transcriber"
                                    })
                                
                                # Reset buffer for next utterance
                                self.audio_buffer = []
                                self.silence_start_time = None
                        
                        # Still add silence chunks to buffer (in case user continues speaking)
                        self.audio_buffer.append(audio_chunk)
                
                except queue.Empty:
                    # Timeout - continue loop to check is_running
                    continue
                except Exception as e:
                    print(f"Error in transcriber: {e}", flush=True)
        
        self._transcriber_thread = threading.Thread(target=_listen_to_audio_loop, daemon=True)
        self._transcriber_thread.start()
    
    def stop(self):
        """Stop transcribing."""
        self.is_running = False
        if self._transcriber_thread and self._transcriber_thread.is_alive():
            self._transcriber_thread.join(timeout=2.0)