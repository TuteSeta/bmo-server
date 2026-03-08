"""
Text-mode test for the BMO agent.

Runs the full AgentLoop (LLM + tools + memory) without microphone or
speakers, so the agent can be tested quickly on any machine.

Usage (from project root):
    python scripts/test_agent.py

Type your message and press Enter. Type 'exit' or 'quit' to stop.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent_loop import AgentLoop
from core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from llm.ollama_client import OllamaClient
from memory.conversation_memory import ConversationMemory


def main() -> None:
    print("=" * 40)
    print("  BMO — Text Test Mode")
    print("  Type 'exit' to quit")
    print("=" * 40)

    llm = OllamaClient(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    memory = ConversationMemory()
    agent = AgentLoop(llm=llm, memory=memory)

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user:
            continue

        if user.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        response = agent.run(user)
        print(f"BMO: {response}\n")


if __name__ == "__main__":
    main()
