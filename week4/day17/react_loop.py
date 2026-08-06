"""
Day 17: The ReAct Loop
----------------------------
This is the SAME idea as Day 16, but made explicit and looped:

  REASON  -> the model thinks about what it needs next
  ACT      -> it calls a tool (or decides it's ready to answer)
  OBSERVE  -> we run the tool and feed the result back

Unlike Day 16 (which only allowed ONE round of tool calls before forcing
a final answer), this loop keeps going — reason, act, observe, reason,
act, observe... — until the model says it doesn't need any more tools,
or we hit a safety cap (MAX_STEPS). This is what lets it chain several
tool calls in a row to answer something that needs multiple steps.
"""

import os
import sys
import json

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.append("../../week2/day7")
from chunking import extract_text_from_pdf, chunk_text

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

TOOL_MODEL = "llama-3.1-8b-instant"        # deciding whether/which tool to call
FINAL_MODEL = "llama-3.3-70b-versatile"     # generating the final answer
PDF_PATH = "the_house_on_briar_lane.pdf"   # copy this into day17/ before running
MAX_STEPS = 5                                # safety cap so the loop can't run forever


# ---------- Set up the document lookup tool's vector store (same as day16) ----------

print(f"Loading and chunking: {PDF_PATH}")
text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_document")

embeddings = embed_model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(embeddings=embeddings, documents=chunks, ids=ids)
print("Chunks stored in vector store!\n")


# ---------- Same 3 tools as Day 16 ----------

def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression like '47 * 89'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains characters that aren't allowed."
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def document_lookup(question: str) -> str:
    """Search the story for information relevant to the question."""
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=len(chunks))
    docs = results['documents'][0]
    return "\n---\n".join(docs)


def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    count = len(text.split())
    return f"{count} words"


AVAILABLE_TOOLS = {
    "calculator": calculator,
    "document_lookup": document_lookup,
    "word_count": word_count,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression, e.g. '47 * 89' or '(12 + 8) / 4'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate."}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "document_lookup",
            "description": "Search the story document for information relevant to a question about its content, characters, or events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to search the document for."}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Count how many words are in a given piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to count words in."}
                },
                "required": ["text"]
            }
        }
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant for a short story called 'The House on Briar Lane'. "
    "You have access to three tools: document_lookup (search the story for characters, "
    "events, objects, or details), calculator (evaluate math expressions), and word_count "
    "(count words in a piece of text).\n\n"
    "Rules:\n"
    "- If the question is about the story in any way, always use document_lookup, even if "
    "you think you might already know the answer.\n"
    "- If the question involves a calculation, use calculator.\n"
    "- If the question asks to count words, use word_count.\n"
    "- If answering fully needs more than one tool (e.g. look something up, then calculate "
    "something with it), call them one at a time, using each result before deciding the next step.\n"
    "- For general knowledge questions unrelated to the story, answer directly without using a tool."
)


def call_tool_with_fallback(tool_name, tool_args, user_message):
    """Runs the requested tool. If the model's tool-call generation itself
    failed to parse (a known Groq free-tier glitch), fall back to calling
    document_lookup directly using the raw user message."""
    if tool_name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[tool_name](**tool_args)
    return document_lookup(user_message)


def react_loop(user_message: str, max_retries: int = 2):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")
        print("[REASON] deciding whether a tool is needed...")

        response = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=TOOL_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0,
                    parallel_tool_calls=False,
                )
                break
            except Exception:
                if attempt == max_retries - 1:
                    # Groq tool-call format glitch — fall back to document_lookup directly
                    print("[REASON] tool-call generation had trouble, falling back to document_lookup")
                    observation = document_lookup(user_message)
                    print(f"[OBSERVE] {observation[:100]}...")
                    messages.append({"role": "assistant", "content": f"(used document_lookup as a fallback)"})
                    messages.append({"role": "user", "content": f"Tool result: {observation}"})
                    response = None

        if response is None:
            continue  # fallback already added to messages, go reason again next step

        response_message = response.choices[0].message

        # REASON concluded: no tool needed, this is the final answer
        if not response_message.tool_calls:
            print("[REASON] no tool needed, ready to answer.")
            break

        messages.append(response_message)

        # ACT + OBSERVE for every tool call the model asked for
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"[ACT] calling {tool_name}({tool_args})")

            tool_result = call_tool_with_fallback(tool_name, tool_args, user_message)

            print(f"[OBSERVE] {tool_result[:120]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })
    else:
        print("[REASON] hit max steps, wrapping up with what we have.")

    # final answer, concise, no more tool calls
    messages.append({
        "role": "system",
        "content": (
            "Using the tool results above, answer the ORIGINAL question completely. "
            "If the question had multiple parts, give ONE short sentence per part — "
            "just the direct fact, no extra description or elaboration. "
            "Do not leave any part unanswered."
        )
    })

    for attempt in range(max_retries):
        try:
            final_response = client.chat.completions.create(
                model=FINAL_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="none",
                temperature=0,
            )
            return final_response.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                return "(final answer failed, please try again)"


def run_cli():
    print("Day 17 ReAct Loop Chatbot. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = react_loop(user_input)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    run_cli()