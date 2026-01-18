from stt.whisper_client import SttClient
from tts.pipertts_client import TtsClient

from lmstudio.client import LlmClient

stt_client = SttClient()
llm_client = LlmClient()
tts_client = TtsClient()

llm_chat = llm_client.new_chat()

try:
    
    while True:
        # Listen for user's spoken prompt
        print("Listening ...", flush=True)

        while True:
            [prompt, confidence_info] = stt_client.listen(1.0)
            # Cntinue if no speech detected with at least one letter
            if not prompt or not any(c.isalpha() for c in prompt):
                continue
            print(f"Heard: {prompt}\n(confidence: {confidence_info}", flush=True)
            # Not confident enough?
            if confidence_info["score"] < 0.6:
                tts_client.speak("Pardon?")
                continue
            break

        # Send prompt to LLM
        chat_response = "Mkay" #llm_chat.prompt(prompt)

        # Process and speak the response stream
        if chat_response:
            buffer = ""
            offset = 0

            for fragment in chat_response:
                if fragment:
                    # Add the new fragment to the buffer
                    buffer += fragment
                    
                    # Check if there are complete sentences (ending with ".", "!", "?") or lines (ending with "\n") after the offset
                    # Find the last sentence/line ending character in the buffer after the offset
                    sentence_endings = [".", "!", "?", "\n"]
                    last_sentence_end_idx = -1
                    for i in range(offset, len(buffer)):
                        if buffer[i] in sentence_endings:
                            last_sentence_end_idx = i
                    
                    # If we found a sentence/line ending, extract and print the sentence/line
                    if last_sentence_end_idx >= offset:
                        # Extract sentence/line from offset to last_sentence_end_idx (inclusive)
                        sentence_to_print = buffer[offset:last_sentence_end_idx + 1]
                        print(sentence_to_print, end="", flush=True)
                        tts_client.speak(sentence_to_print)
                        # Update offset to after the last sentence/line ending
                        offset = last_sentence_end_idx + 1

                        if (sentence_to_print.lower().endswith("goodbye.")):
                            exit()

            # Print any remaining content in the buffer after the stream ends
            if offset < len(buffer):
                remaining = buffer[offset:]
                print(remaining, end="", flush=True)
                tts_client.speak(remaining)

            print("\n")  # Final newline and spacing

except KeyboardInterrupt:
    print("\n\nExiting...", flush=True)