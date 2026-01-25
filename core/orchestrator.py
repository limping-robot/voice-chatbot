"""
Orchestrator state machine for voice chatbot.

Manages the flow between LISTENING, THINKING, and SPEAKING states,
handling barge-in detection and coordinating all components.
"""

import queue
import logging
import re
import time
from enum import Enum

import numpy as np

from audio.types import MicEvent, SPEECH_END, BARGE_IN
from audio.wav_player import load_wav_frames


logger = logging.getLogger(__name__)

# Precompiled regex pattern for sentence endings
SENTENCE_ENDING_PATTERN = re.compile(r'[.!?\n](?!\S)')


class OrchestratorState(Enum):
    """Orchestrator states."""
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class Orchestrator:
    """
    Orchestrator for voice chatbot pipeline.
    
    Manages state transitions and coordinates:
    - MicProcessor (AEC + VAD + events)
    - SttTranscriber (transcription)
    - LlmClient (LLM interaction)
    - TtsSynthesizer (synthesis)
    - AudioPlayer (playback)
    """
    
    def __init__(
        self,
        mic_processor,
        transcriber,
        llm_chat,
        tts_synthesizer,
        audio_player,
        full_duplex_io,
        min_confidence_score: float = 0.5,
        min_logprob_score: float = 0.7
    ):
        """
        Initialize orchestrator.
        
        Args:
            mic_processor: MicProcessor instance
            transcriber: SttTranscriber instance
            llm_chat: LlmChat instance
            tts_synthesizer: TtsSynthesizer instance
            audio_player: AudioPlayer instance (for enqueue_frame)
            full_duplex_io: FullDuplexIO instance
            min_confidence_score: Minimum confidence score for transcription
            min_logprob_score: Minimum logprob score for transcription
        """
        self.mic_processor = mic_processor
        self.transcriber = transcriber
        self.llm_chat = llm_chat
        self.tts_synthesizer = tts_synthesizer
        self.audio_player = audio_player
        self.full_duplex_io = full_duplex_io
        self.min_confidence_score = min_confidence_score
        self.min_logprob_score = min_logprob_score
        
        # State
        self.state = OrchestratorState.LISTENING
        self.event_queue: queue.Queue[MicEvent] = queue.Queue()  # Will be set by main
        
        # LLM response streaming
        self._llm_response_buffer = ""
        self._llm_response_offset = 0
        self._synthesizing = False
        
        # Startup grace period: ignore speech events for 2 seconds after startup
        # to prevent transcribing the startup sound cue
        self._startup_time = None
        self._startup_grace_period_sec = 2.0
    
    def get_state(self) -> str:
        """Get current state as string."""
        return self.state.value
    
    def _find_first_sentence_ending(self, buffer: str, offset: int) -> int:
        """
        Find the index of the first sentence ending in the buffer after the given offset.
        
        Args:
            buffer: The text buffer to search in
            offset: The position to start searching from
        
        Returns:
            The index of the last character of the first sentence ending found, or -1 if none found
        """
        match = SENTENCE_ENDING_PATTERN.search(buffer[offset:])
        if match:
            return offset + match.end() - 1
        return -1
    
    def _enqueue_wav_sound(self, filename: str):
        """Enqueue a WAV sound through the speaker pipeline."""
        for frame in load_wav_frames(filename):
            self.audio_player.enqueue_frame(frame)
    
    def _handle_listening_state(self, event: MicEvent):
        """Handle events in LISTENING state."""
        logger.debug(f"LISTENING state: received event {event.event_type}")
        if event.event_type == SPEECH_END:
            # Ignore speech_end events during startup grace period
            if self._startup_time is not None:
                elapsed = time.time() - self._startup_time
                if elapsed < self._startup_grace_period_sec:
                    logger.info(f"Ignoring speech_end event during startup grace period ({elapsed:.2f}s < {self._startup_grace_period_sec}s)")
                    # Clear the utterance buffer to prevent transcription
                    self.mic_processor.get_utterance_frames()
                    return
            # Speech ended, get utterance and transcribe
            utterance_frames = self.mic_processor.get_utterance_frames()
            
            if not utterance_frames:
                logger.warning("Speech end but no utterance frames")
                return
            
            # Convert frames to numpy arrays for transcription
            audio_chunks = [frame.pcm.astype(np.float32) / 32768.0 for frame in utterance_frames]
            
            # Transcribe
            logger.info("Transcribing utterance...")
            transcribed_text, confidence_info = self.transcriber.transcribe_frames(audio_chunks)
            
            logger.info(f"Transcribed: text='{transcribed_text}', confidence_info={confidence_info}")
            
            # Validate transcription
            if not transcribed_text or not any(c.isalnum() for c in transcribed_text):
                logger.info("No valid text in transcription")
                return
            
            if confidence_info.get("no_speech_prob", 0) > 0.5:
                logger.info("Low speech probability, asking for repeat")
                self._enqueue_wav_sound("mixkit-modern-technology-select-3124.wav")
                # Enqueue "Pardon?" through speaker pipeline
                for frame in self.tts_synthesizer.synthesize_stream("Pardon?"):
                    self.audio_player.enqueue_frame(frame)
                return
            
            if (confidence_info.get("score", 0) < self.min_confidence_score and 
                confidence_info.get("logprob_score", 0) < self.min_logprob_score):
                logger.info("Low confidence, asking for repeat")
                self._enqueue_wav_sound("mixkit-modern-technology-select-3124.wav")
                for frame in self.tts_synthesizer.synthesize_stream("Could you please repeat?"):
                    self.audio_player.enqueue_frame(frame)
                return
            
            # Valid transcription, transition to THINKING
            self._enqueue_wav_sound("mixkit-select-click-1109.wav")
            self.state = OrchestratorState.THINKING
            self._process_llm_request(transcribed_text)
    
    def _process_llm_request(self, prompt: str):
        """Process LLM request and transition to SPEAKING."""
        logger.info(f"LLM interaction - Sending prompt: '{prompt}'")
        
        try:
            chat_response = self.llm_chat.prompt(prompt)
            
            if not chat_response:
                logger.warning("LLM interaction - No response received")
                self.state = OrchestratorState.LISTENING
                return
            
            logger.info("LLM interaction - Response stream started")
            self.state = OrchestratorState.SPEAKING
            self._synthesizing = True
            self._llm_response_buffer = ""
            self._llm_response_offset = 0
            
            # Process response stream
            for fragment in chat_response:
                if fragment:
                    self._llm_response_buffer += fragment
                    self._process_sentence_chunk()
            
            # Send any remaining text
            if self._llm_response_offset < len(self._llm_response_buffer):
                remaining = self._llm_response_buffer[self._llm_response_offset:]
                if remaining.strip():
                    for frame in self.tts_synthesizer.synthesize_stream(remaining):
                        if not self._synthesizing:
                            break
                        self.audio_player.enqueue_frame(frame)
            
            self._synthesizing = False
            logger.info("LLM interaction - Response stream completed")
            
            # Transition back to LISTENING
            self.state = OrchestratorState.LISTENING
            
        except Exception as e:
            logger.error(f"Error processing LLM request: {e}", exc_info=True)
            self.state = OrchestratorState.LISTENING
            self._synthesizing = False
    
    def _process_sentence_chunk(self):
        """Process complete sentences from LLM response buffer."""
        while True:
            first_sentence_end_idx = self._find_first_sentence_ending(
                self._llm_response_buffer,
                self._llm_response_offset
            )
            
            if first_sentence_end_idx >= self._llm_response_offset:
                # Extract sentence
                sentence = self._llm_response_buffer[self._llm_response_offset:first_sentence_end_idx + 1]
                
                # Synthesize and enqueue
                for frame in self.tts_synthesizer.synthesize_stream(sentence):
                    if not self._synthesizing:
                        break
                    self.audio_player.enqueue_frame(frame)
                
                # Update offset
                self._llm_response_offset = first_sentence_end_idx + 1
            else:
                # No more complete sentences
                break
    
    def _handle_speaking_state(self, event: MicEvent):
        """Handle events in SPEAKING state."""
        if event.event_type == BARGE_IN:
            logger.info("Barge-in detected! Stopping playback and switching to LISTENING")
            
            # Stop synthesis
            self._synthesizing = False
            
            # Flush speaker queue and stop
            self.audio_player.enqueue_stop()
            self.audio_player.flush()
            
            # Say "Yes?" and transition to LISTENING
            for frame in self.tts_synthesizer.synthesize_stream("Yes?"):
                self.audio_player.enqueue_frame(frame)
            
            self.state = OrchestratorState.LISTENING
    
    def run(self):
        """Main orchestrator loop."""
        logger.info("Starting orchestrator...")
        
        # Record startup time for grace period
        self._startup_time = time.time()
        
        # Play listening sound through speaker pipeline (non-blocking)
        self._enqueue_wav_sound("mixkit-cool-interface-click-tone-2568.wav")
        logger.info("Orchestrator ready, waiting for speech events...")
        
        while True:
            try:
                # Get event from mic processor
                event = self.event_queue.get(timeout=0.1)
                
                # Handle event based on current state
                if self.state == OrchestratorState.LISTENING:
                    self._handle_listening_state(event)
                elif self.state == OrchestratorState.THINKING:
                    # THINKING state: just wait for LLM processing to complete
                    # (handled in _process_llm_request)
                    pass
                elif self.state == OrchestratorState.SPEAKING:
                    self._handle_speaking_state(event)
            
            except queue.Empty:
                # No events, continue
                continue
            except KeyboardInterrupt:
                logger.info("Orchestrator interrupted")
                break
            except Exception as e:
                logger.error(f"Error in orchestrator: {e}", exc_info=True)
