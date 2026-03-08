import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.ollama_client import OllamaClient

prompt = "Hello BMO, are you alive?"
client = OllamaClient()

print(f"User: {prompt}")
response = client.generate(prompt)
print(f"BMO: {response}")
