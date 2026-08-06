"""
Day 19: Multi-Agent Router
--------------------------------
One small "router" call decides which of THREE paths handles a question:

  1. "document"    -> send it to the RAG pipeline (Week 2/3 retrieval,
                        same idea as day15/day17's document_lookup)
  2. "tool"        -> send it to the tool-calling agent (Day 16/17's
                        calculator / word_count)
  3. "conversation" -> just answer directly, no retrieval or tool needed
                        (e.g. "hey, how are you")

This is the fix for "one agent trying to do everything gets messy" —
instead of one giant prompt juggling retrieval + tools + small talk,
a router looks at the question ONCE and hands it to the right specialist.
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

ROUTER_MODEL = "llama-3.1-8b-instant"
ANSWER_MODEL = "llama-3.3-70b-versatile"
PDF_PATH = "the_house_on_briar_lane.pdf"  # copy this into day19/ before running


# ---------- Set up the document retrieval path (same as day15/day17) ----------

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


def document_lookup(question: str) -> str:
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=len(chunks))
    return "\n---\n".join(results['documents'][0])


# ---------- Set up the tool-calling path (same as day16/day17) ----------

def calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains characters that aren't allowed."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def word_count(text: str) -> str:
    return f"{len(text.split())} words"


AVAILABLE_TOOLS = {"calculator": calculator, "word_count": word_count}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression, e.g. '47 * 89'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
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
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        }
    },
]


# ---------- STEP 1: the router itself ----------

def classify_question(question: str) -> str:
    """Decide which of the 3 paths this question needs. Returns one of:
    'document', 'tool', or 'conversation'."""
    system_prompt = (
        "Classify the user's question into EXACTLY ONE category. Reply with "
        "only the category word, nothing else.\n\n"
        "- document: the question could be about the short story 'The House on Briar Lane' — "
        "its characters (Naila, Danish, Amal), events, objects, places, or any detail from it — "
        "even if it doesn't explicitly say 'the story' or name a character. Questions about "
        "'where someone lives', 'what happened', or 'tell me about X' from a story-like scenario "
        "should default to document unless clearly about something else (math, greetings).\n"
        "- tool: the question needs a calculation (math) or counting words in text.\n"
        "- conversation: greetings, small talk, or general knowledge clearly unrelated to any "
        "story or narrative (e.g. capitals, weather, how are you).\n\n"
        "Examples:\n"
        "'whats inside the wooden box' -> document\n"
        "'tell me about the box' -> document\n"
        "'where does naila live' -> document\n"
        "'where does she live' -> document\n"
        "'how are you' -> conversation\n"
        "'what's 5 times 5' -> tool\n"
        "'who lives in the house' -> document\n"
        "'what's the weather like' -> conversation"
    )

    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    label = response.choices[0].message.content.strip().lower()

    # guard against the model returning something unexpected
    if "document" in label:
        return "document"
    if "tool" in label:
        return "tool"
    return "conversation"


# ---------- STEP 2: the three specialist handlers ----------

def handle_document(question: str) -> str:
    context = document_lookup(question)
    prompt = (
        "Answer using ONLY the context below, in 1-2 short sentences. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def handle_tool(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with a calculator and word counter tool."},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        temperature=0,
    )
    response_message = response.choices[0].message

    if not response_message.tool_calls:
        return response_message.content

    messages.append(response_message)
    for tool_call in response_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

    messages.append({"role": "system", "content": "Answer in 1 short sentence using the tool result above."})
    final = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="none",
        temperature=0,
    )
    return final.choices[0].message.content


def handle_conversation(question: str) -> str:
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": "You are a friendly assistant. Keep replies short and natural."},
            {"role": "user", "content": question}
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ---------- STEP 3: the router loop ----------

def run_cli():
    print("Day 19 Multi-Agent Router. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        route = classify_question(user_input)
        print(f"[router -> {route}]")

        if route == "document":
            answer = handle_document(user_input)
        elif route == "tool":
            answer = handle_tool(user_input)
        else:
            answer = handle_conversation(user_input)

        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    run_cli()