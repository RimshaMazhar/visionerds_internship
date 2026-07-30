"""
Day 13: Compare Naive Memory (Day 11) vs Buffer Memory (Day 13)
------------------------------------------------------------------
Runs the SAME long conversation through both memory strategies and
prints a side-by-side comparison of:
  - tokens sent per turn (this is what actually grows)
  - response time per turn
  - whether each bot still remembers something from early in the chat

Run this AFTER buffer_memory.py works, to see the difference for yourself
instead of just being told about it.
"""

import os
import time
from collections import deque
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"
BUFFER_SIZE = 8


def estimate_tokens(messages):
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)


class NaiveMemoryBot:
    """Day 11 style: append everything, send the whole history every time."""

    def __init__(self):
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful assistant. Keep every answer to 1-2 short sentences, no lists."
        }
        self.history = []

    def ask(self, user_message: str):
        self.history.append({"role": "user", "content": user_message})
        messages = [self.system_prompt] + self.history
        tokens = estimate_tokens(messages)

        start = time.time()
        response = client.chat.completions.create(model=MODEL, messages=messages)
        elapsed = time.time() - start

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply, tokens, elapsed


class BufferMemoryBot:
    """Day 13 style: only keep the last N messages."""

    def __init__(self, buffer_size=BUFFER_SIZE):
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful assistant. Keep every answer to 1-2 short sentences, no lists."
        }
        self.history = deque(maxlen=buffer_size)

    def ask(self, user_message: str):
        self.history.append({"role": "user", "content": user_message})
        messages = [self.system_prompt] + list(self.history)
        tokens = estimate_tokens(messages)

        start = time.time()
        response = client.chat.completions.create(model=MODEL, messages=messages)
        elapsed = time.time() - start

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply, tokens, elapsed


def run_comparison():
    naive_bot = NaiveMemoryBot()
    buffer_bot = BufferMemoryBot()

    print("Day 13: Naive vs Buffer — LIVE comparison")
    print("Type your own messages. Each one goes to BOTH bots at the same time.")
    print("Type 'quit' to stop and see the summary.\n")

    print(f"{'Turn':<5}{'Naive tokens':<15}{'Buffer tokens':<15}{'Naive time':<12}{'Buffer time':<12}")
    print("-" * 60)

    turn = 0
    naive_reply = buffer_reply = ""

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        turn += 1
        naive_reply, naive_tokens, naive_time = naive_bot.ask(user_input)
        buffer_reply, buffer_tokens, buffer_time = buffer_bot.ask(user_input)

        print(f"{turn:<5}{naive_tokens:<15}{buffer_tokens:<15}{naive_time:<12.2f}{buffer_time:<12.2f}")
        print(f"  [Naive]  -> {naive_reply}")
        print(f"  [Buffer] -> {buffer_reply}\n")

    print("\n--- Last answers from each bot ---\n")
    print(f"Naive bot (full history) last said:\n{naive_reply}\n")
    print(f"Buffer bot (last {BUFFER_SIZE} messages) last said:\n{buffer_reply}\n")

    print("Compare: did the buffer bot forget something from early in the chat")
    print("that the naive bot still remembered? That's the trade-off in action.")


if __name__ == "__main__":
    run_comparison()