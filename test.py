from stt.whisper_client import SttClient
from tts.pipertts_client import TtsClient

from llm.client import LlmClient

stt_client = SttClient()
llm_client = LlmClient()
tts_client = TtsClient()

response = llm_chat = llm_client.prompt(
    [
        { "role": "system", "content": 
        "You are a helpful assistant. " + 
        "Keep your answers very short and concise, ideally not more that one sentence. " +
        "Before going into details, ask the user if they want to know more. " + 
        "Also avoid any separators, like --- etc."
        "Do not use any emojis. " + 
        "Skip any introductions or explanations."
        },
        { "role": "user", "content": "What is Python?" },
        { "role": "assistant", "content": "Python is an high-level" },
        { "role": "user", "content": "Wait a moment" },
        { "role": "assistant", "content": "Sure thing! How can I assist you further with this topic in mind?" },
        { "role": "user", "content": "Please continue, but keep it short" },
        { "role": "assistant", "content": "Certainly. Python supports multiple programming paradigms including procedural, object-oriented and functional styles. It's widely used for web development, data analysis, artificial intelligence, scientific computing etc. " },
        { "role": "user", "content": "Give me a sec" },
        { "role": "assistant", "content": "Of course! Feel free to ask any questions or request more information on the topic whenever you're ready. I'm here to help!" },
        { "role": "user", "content": "Thanks. Please continue" },
    ]
)

for chunk in response:
    if (chunk.choices[0].delta):
        if (chunk.choices[0].delta.content):
            print(chunk.choices[0].delta.content, end="", flush=True)