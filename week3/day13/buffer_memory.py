import os
import time
from collections import deque
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"  # swap for whichever free-tier model you're using

# How many messages to keep in the buffer.
# The PDF says 6-8 messages -> we'll keep it configurable.
BUFFER_SIZE = 8


class BufferMemoryChatbot:
    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.buffer_size = buffer_size
        # deque with maxlen automatically drops the oldest item once full
        self.history = deque(maxlen=buffer_size)
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely."
        }

    def _build_messages(self):
        # system prompt + whatever is currently in the buffer
        return [self.system_prompt] + list(self.history)

    def _estimate_tokens(self, messages):
        # rough estimate: word count * 1.3, same trick as Day 11
        total_words = sum(len(m["content"].split()) for m in messages)
        return int(total_words * 1.3)

    def ask(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        messages = self._build_messages()
        token_estimate = self._estimate_tokens(messages)

        start = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        elapsed = time.time() - start

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})

        print(f"[buffer size in use: {len(self.history)}/{self.buffer_size} messages | "
              f"~{token_estimate} tokens sent | {elapsed:.2f}s]")

        return reply


def run_cli():
    bot = BufferMemoryChatbot()
    print("Day 13 Buffer Memory Chatbot. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply = bot.ask(user_input)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    run_cli()