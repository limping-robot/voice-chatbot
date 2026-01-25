import re
import logging
import sys
from stt.whisper_client import SttTranscriber
from tts.pipertts_client import TtsSynthesizer
from audio.audio_recorder import AudioRecorder
from audio.audio_player import AudioPlayer
from llm.client import LlmClient

from audio import wav_player

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create pipeline components
audio_recorder = AudioRecorder()
audio_player = AudioPlayer()
transcriber = SttTranscriber(silence_duration_sec=1.5)
tts_synthesizer = TtsSynthesizer()

# LLM client
llm_client = LlmClient()
llm_chat = llm_client.new_chat()

# Precompiled regex pattern for sentence endings
# Matches: period, exclamation, question mark, or newline followed by whitespace or end of string
SENTENCE_ENDING_PATTERN = re.compile(r'[.!?\n](?!\S)')

def find_first_sentence_ending(buffer: str, offset: int) -> int:
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
        # match.end() gives the position after the match
        # We need the position of the last character of the match
        return offset + match.end() - 1
    
    return -1


def process_llm_response_sentences(chat_response):
    """
    Generator that processes LLM response stream and yields complete sentences.
    
    Buffers fragments from the response stream and extracts complete sentences
    (ending with ".", "!", "?", or "\n") as they become available.
    
    Args:
        chat_response: Iterator of text fragments from LLM
    
    Yields:
        Complete sentences as strings (including the ending punctuation)
    """
    buffer = ""
    offset = 0
    
    for fragment in chat_response:
        if fragment:
            # Add the new fragment to the buffer
            buffer += fragment
            
            # Process sentences one by one from the buffer
            while True:
                # Check if there are complete sentences (ending with ".", "!", "?") 
                # or lines (ending with "\n") after the offset
                first_sentence_end_idx = find_first_sentence_ending(buffer, offset)
                
                # If we found a sentence/line ending, extract and yield it
                if first_sentence_end_idx >= offset:
                    # Extract sentence/line from offset to first_sentence_end_idx (inclusive)
                    sentence = buffer[offset:first_sentence_end_idx + 1]
                    yield sentence
                    
                    # Update offset to after the first sentence/line ending
                    offset = first_sentence_end_idx + 1
                else:
                    # No more complete sentences in buffer, break inner loop to get more fragments
                    break
    
    # Send any remaining text in the buffer (if response ended without sentence ending)
    if offset < len(buffer):
        remaining = buffer[offset:]
        if remaining.strip():  # Only yield if there's non-whitespace content
            yield remaining

def main_loop():
    """Main loop: listen, transcribe, send to LLM, and play response."""
    
    while True:
        try:
            # Play listening start sound
            wav_player.play_listening_sound(audio_player)
            
            # Listen and transcribe (blocking until silence)
            logger.info("Listening for speech...")

            transcribed_text, confidence_info = transcriber.transcribe(audio_recorder)
            
            # Log transcription result
            logger.info(f"Transcribed: text='{transcribed_text}', confidence_info={confidence_info}")
            
            # Continue if no speech detected with at least one letter
            if not transcribed_text or not any(c.isalnum() for c in transcribed_text):
                continue

            # Did we even hear speech?
            if confidence_info.get("no_speech_prob", 0) > 0.5:
                wav_player.play_transcribed_sound(audio_player)
                tts_synthesizer.synthesize_and_play("Pardon?", audio_player)
                continue

            # Are we confident about the transcription?
            if confidence_info.get("score", 0) < 0.5 and confidence_info.get("logprob_score", 0) < 0.7:
                # Output "Pardon?" to let user confirm their request
                wav_player.play_transcribed_sound(audio_player)
                tts_synthesizer.synthesize_and_play("Could you please repeat?", audio_player)
                continue

            # Play transcription completion sound
            wav_player.play_prompting_sound(audio_player)
            
            # Send prompt to LLM
            logger.info(f"LLM interaction - Sending prompt: '{transcribed_text}'")
            chat_response = llm_chat.prompt(transcribed_text)

            # Log LLM interaction - response received
            if chat_response:
                logger.info("LLM interaction - Response stream started")
            else:
                logger.warning("LLM interaction - No response received")
                continue

            # Process and play response stream sentence by sentence
            for sentence in process_llm_response_sentences(chat_response):
                # Synthesize and play (blocking until playback completes)
                tts_synthesizer.synthesize_and_play(sentence, audio_player)
            
            # Log completion of LLM response
            logger.info("LLM interaction - Response stream completed")
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)

try:
    logger.info("Starting pipeline...")
    logger.info("Listening ...")
    
    # Start audio recorder (needed for direct recording)
    audio_recorder.start()
    
    # Run main loop
    main_loop()

except KeyboardInterrupt:
    logger.info("Exiting...")
    audio_recorder.shutdown()
