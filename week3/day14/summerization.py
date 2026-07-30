"""
Day 14: Summarization Memory
--------------------------------
Buffer memory (Day 13) just drops old messages once the buffer is full —
they're gone completely. Summarization memory is smarter: once the
history gets too long, it makes an LLM call to compress the OLDER turns
into a short summary, keeps that summary + the recent raw messages, and
carries on. This way the bot keeps "the gist" of a long conversation
without paying the full token cost of the entire history.

Flow:
  1. Keep adding messages normally.
  2. Once the raw message count crosses a threshold (e.g. 6 messages),
     summarize everything so far into a few sentences.
  3. Replace the old raw messages with ONE summary message.
  4. Keep going — new messages get added after the summary, and the
     cycle repeats once things get long again.
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

# once we have this many raw (non-summary) messages, we summarize them
SUMMARIZE_THRESHOLD = 6


def estimate_tokens(messages):
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)


class SummarizationMemoryBot:
    def __init__(self, summarize_threshold: int = SUMMARIZE_THRESHOLD):
        self.summarize_threshold = summarize_threshold
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful assistant. Keep every answer to 1-2 short sentences, no lists."
        }
        # summary of everything older than what's in self.recent_history
        self.summary = ""
        # raw recent messages (not yet folded into the summary)
        self.recent_history = []

    def _summarize(self):
        """Compress self.recent_history into a short summary, combined with
        any existing summary, and store it in self.summary."""
        convo_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.recent_history
        )

        prompt = (
            "Write a 2-3 sentence summary of the conversation below. "
            "Include any important facts (names, numbers, decisions, preferences). "
            "Output ONLY the summary text itself — no preamble, no meta-comments "
            "like 'here is a summary' or 'there is no previous summary', just the "
            "plain summary sentences.\n\n"
        )
        if self.summary:
            prompt += f"Existing summary so far: {self.summary}\n\n"
        prompt += f"Conversation to add:\n{convo_text}"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        self.summary = response.choices[0].message.content
        self.recent_history = []

    def _build_messages(self):
        messages = [self.system_prompt]
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation: {self.summary}"
            })
        messages += self.recent_history
        return messages

    def ask(self, user_message: str):
        self.recent_history.append({"role": "user", "content": user_message})

        messages = self._build_messages()
        tokens = estimate_tokens(messages)

        start = time.time()
        response = client.chat.completions.create(model=MODEL, messages=messages)
        elapsed = time.time() - start

        reply = response.choices[0].message.content
        self.recent_history.append({"role": "assistant", "content": reply})

        # if recent_history has grown past the threshold, fold it into the summary
        if len(self.recent_history) >= self.summarize_threshold:
            self._summarize()

        return reply, tokens, elapsed


def run_cli():
    bot = SummarizationMemoryBot()
    print("Day 14 Summarization Memory Chatbot. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply, tokens, elapsed = bot.ask(user_input)
        print(f"Bot: {reply}")
        print(f"[~{tokens} tokens sent | {elapsed:.2f}s | current summary: "
              f"{'(none yet)' if not bot.summary else bot.summary[:80] + '...'}]\n")


if __name__ == "__main__":
    run_cli()