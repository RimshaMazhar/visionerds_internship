import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Yahan poori conversation history store hogi
chat_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]


def estimate_tokens(text):
    # Rough estimate: word count * 1.3
    return int(len(text.split()) * 1.3)


total_tokens = 0

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot closing. Goodbye!")
        break

    # User ka message history mein daalo
    chat_history.append({"role": "user", "content": user_input})

    # Poori history bhejo model ko
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=chat_history
    )

    reply = response.choices[0].message.content
    print(f"\nAssistant: {reply}")

    # Assistant ka jawab bhi history mein daalo
    chat_history.append({"role": "assistant", "content": reply})

    # Token count track karo
    turn_tokens = estimate_tokens(user_input) + estimate_tokens(reply)
    total_tokens += turn_tokens
    print(f"[Approx tokens this turn: {turn_tokens} | Total so far: {total_tokens}]")