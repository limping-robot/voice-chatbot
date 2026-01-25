"""
Main entry point for voice chatbot with barge-in support.

Uses orchestrator-based architecture with:
- Full-duplex audio I/O
- Continuous microphone processing (AEC + VAD)
- Interruptible TTS playback
- Barge-in detection
"""

import logging
import sys

from stt.whisper_client import SttTranscriber
from tts.pipertts_client import TtsSynthesizer
from audio.audio_player import AudioPlayer
from audio.full_duplex_io import FullDuplexIO
from audio.mic_processor import MicProcessor
from llm.client import LlmClient
from core.orchestrator import Orchestrator

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    import queue
    
    logger.info("Starting voice chatbot with barge-in support...")
    
    # Create full-duplex audio I/O
    full_duplex_io = FullDuplexIO(
        sample_rate=16000,
        channels=1,
        frame_size=160  # 10ms at 16kHz
    )
    
    # Create audio player (uses frame queue from full_duplex_io)
    audio_player = AudioPlayer(
        frame_queue=full_duplex_io.speaker_frame_queue
    )
    
    # Create microphone processor
    event_queue = queue.Queue()
    mic_processor = MicProcessor(
        mic_frame_queue=full_duplex_io.mic_frame_queue,
        event_queue=event_queue,
        speaker_ref_getter=lambda: full_duplex_io.get_speaker_reference(delay_frames=0),
        enable_aec=True,  # AEC will be disabled automatically if aec_audio_processing not available
        barge_in_threshold_ms=300,
        speech_end_silence_ms=1500  # 1.5 seconds to match user expectation
    )
    
    # Create STT transcriber
    transcriber = SttTranscriber(silence_duration_sec=1.5)
    
    # Create TTS synthesizer
    tts_synthesizer = TtsSynthesizer()
    
    # Create LLM client
    llm_client = LlmClient()
    llm_chat = llm_client.new_chat()
    
    # Create orchestrator
    orchestrator = Orchestrator(
        mic_processor=mic_processor,
        transcriber=transcriber,
        llm_chat=llm_chat,
        tts_synthesizer=tts_synthesizer,
        audio_player=audio_player,
        full_duplex_io=full_duplex_io,
        min_confidence_score=0.5,
        min_logprob_score=0.7
    )
    
    # Set event queue on orchestrator
    orchestrator.event_queue = event_queue
    
    try:
        # Start full-duplex audio stream
        logger.info("Starting full-duplex audio stream...")
        full_duplex_io.start()
        
        # Start microphone processor
        logger.info("Starting microphone processor...")
        mic_processor.start(orchestrator_state_getter=orchestrator.get_state)
        
        # Run orchestrator (blocks)
        logger.info("Starting orchestrator...")
        orchestrator.run()
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        mic_processor.shutdown()
        full_duplex_io.shutdown()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
