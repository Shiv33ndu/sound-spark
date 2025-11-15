import os, time, json
class LLMClient:
    def __init__(self, provider="mock"):
        self.provider = provider
        self.api_key = os.getenv("LLM_API_KEY")

    def call(self, prompt, max_tokens=512, temperature=0.4):
        time.sleep(0.2)
        return {"text": '{"mock": "response"}', "raw": "mock"}
