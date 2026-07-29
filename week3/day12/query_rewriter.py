import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def rewrite_query(chat_history, new_question):
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history

    history_text = ""
    for msg in recent_history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    system_prompt = """You are a query rewriting engine. Decide if the new question is STANDALONE or a FOLLOW-UP.

STANDALONE: complete question, understandable with no missing subject.
FOLLOW-UP: incomplete on its own, relies on the previous turn (bare names, "what about X", "why though").

RULES:
- If STANDALONE: return it exactly unchanged.
- If FOLLOW-UP: use the structure of the MOST RECENT question only (the very last one asked, ignore older ones), and substitute in the new detail.
- Output ONLY the final question, nothing else.

EXAMPLES:
History: "who is Naila's daughter?"
New: "what about maha"
Output: who is Maha's daughter?

History: "who is Naila's daughter?"
New: "what is the capital of Japan?"
Output: what is the capital of Japan?

History: "what is the price of cars?" then "who is the pm of pakistan?"
New: "what about canada?"
Output: who is the pm of canada?"""

    user_prompt = f"""Conversation history:
{history_text}

New question: {new_question}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten


chat_history = []

print("Query Rewriter Test - ask your question!")
print("(type 'exit' to quit)")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Closing. Goodbye!")
        break

    if len(chat_history) > 0:
        rewritten = rewrite_query(chat_history, question)
        print(f"[Rewritten query: {rewritten}]")
    else:
        rewritten = question
        print("[No history yet, using original question]")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Keep your answers short, 2-3 sentences max."},
            {"role": "user", "content": rewritten}
        ]
    )
    answer = response.choices[0].message.content
    print(f"\nAssistant: {answer}")

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})