from openai import OpenAI, ChatCompletion
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_IDENTIFIER = "phi-4-mini-instruct"

class LlmChat:
    def __init__(self):
        self.llm_client = LlmClient()
        self.context = [
            { "role": "system", "content": 
                "You are a helpful assistant. " + 
                "Keep your answers very short and concise, ideally not more that one sentence. " +
                "Before going into details, ask the user if they want to know more. " + 
                "Also avoid any separators, like --- etc."
                "Do not use any emojis. " + 
                "Skip any introductions or explanations."
            }
        ]

    def prompt(self, prompt: list[dict[str, str]]) -> ChatCompletion:
        self.context.append({ "role": "user", "content": prompt })
        
        stream = self.llm_client.prompt(messages=self.context)

        fragments: list[str] = []
        
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            # 1) store fragment
            fragments.append(delta)

            # 2) forward fragment to caller
            yield delta

            # finalize full assistant message

        reply = "".join(fragments)
        self.context.append({"role": "assistant", "content": reply})

        return reply

class LlmClient:
    def __init__(self):
        self.client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    def new_chat(self) -> LlmChat:
        return LlmChat()

    def prompt(self, messages: list[dict[str, str]]) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=MODEL_IDENTIFIER,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
