"""
Day 16: Function Calling Basics
------------------------------------
Give the model a set of tools it can CHOOSE to call. The model never runs
code itself — it replies with a structured "call this function with these
arguments" response, our code actually runs the function, then we send the
result back so the model can use it in its final answer.

Loop:
  1. Send user message + tool definitions to the model.
  2. Check if the model wants to call a tool (response.tool_calls).
  3. If yes: run the real Python function ourselves, send the result back
     as a "tool" message, then ask the model again for the final answer.
  4. If no: the model just answered directly, we're done.
"""

import os
import sys
import json

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# reuse the exact chunking code from Week 2, day7 (same pattern as day15)
sys.path.append("../../week2/day7")
from chunking import extract_text_from_pdf, chunk_text

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

TOOL_MODEL = "llama-3.1-8b-instant"       # deciding whether/which tool to call (avoids malformed tool-call bug)
FINAL_MODEL = "llama-3.3-70b-versatile"    # generating the final answer (better quality, no tool-call risk here)
PDF_PATH = "the_house_on_briar_lane.pdf"  # copy this into day16/ before running


# ---------- Set up the document lookup tool's vector store (same as day15) ----------

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


# ---------- TOOL 1: calculator ----------

def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression like '47 * 89'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains characters that aren't allowed."
    try:
        result = eval(expression)  # safe here because we already whitelisted characters
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---------- TOOL 2: document lookup ----------

def document_lookup(question: str) -> str:
    """Search the vector store for chunks relevant to the question.
    The story is short, so we just retrieve ALL chunks — this guarantees
    nothing gets missed, regardless of how well the embedding search ranks
    any single chunk."""
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=len(chunks))
    docs = results['documents'][0]
    return "\n---\n".join(docs)


# ---------- TOOL 3: word counter ----------

def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    count = len(text.split())
    return f"{count} words"


# map tool names -> actual Python functions, so we can call them dynamically
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "document_lookup": document_lookup,
    "word_count": word_count,
}

# tool DEFINITIONS the model sees (JSON schema describing name/args/purpose)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression, e.g. '47 * 89' or '(12 + 8) / 4'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate."
                    }
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
                    "question": {
                        "type": "string",
                        "description": "The question to search the document for."
                    }
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
                    "text": {
                        "type": "string",
                        "description": "The text to count words in."
                    }
                },
                "required": ["text"]
            }
        }
    },
]


# ---------- STEP: the raw function calling loop ----------

def ask_with_tools(user_message: str, max_retries: int = 2):
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant for a short story called 'The House on Briar Lane'. "
            "You have access to three tools: document_lookup (search the story for characters, "
            "events, objects, or details), calculator (evaluate math expressions), and word_count "
            "(count words in a piece of text).\n\n"
            "Rules:\n"
            "- If the question is about the story in any way, always use document_lookup, even if "
            "you think you might already know the answer.\n"
            "- If the question involves a calculation, use calculator.\n"
            "- If the question asks to count words, use word_count.\n"
            "- For general knowledge questions unrelated to the story (capitals, historical facts, "
            "definitions), answer directly without using a tool."
        )},
        {"role": "user", "content": user_message}
    ]

    response = None
    for attempt in range(max_retries):
        try:
            # 1. send the message + tool definitions to the model
            response = client.chat.completions.create(
                model=TOOL_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",  # let the model decide whether to use a tool
                temperature=0,       # more deterministic, reduces malformed tool-call glitches
                parallel_tool_calls=False,
            )
            break  # success, stop retrying
        except Exception as e:
            if attempt == max_retries - 1:
                # Groq's tool-call generation is still failing after retries.
                # Fallback: since our system prompt already says story-related
                # questions need document_lookup, just call it directly
                # ourselves instead of depending on the broken tool-call format.
                print("[tool call had trouble, using document lookup as a fallback]")
                tool_result = document_lookup(user_message)
                final_answer = client.chat.completions.create(
                    model=FINAL_MODEL,
                    messages=[
                        {"role": "system", "content": "Answer in 1-2 short sentences using ONLY the context. If the answer isn't in the context, say you don't know."},
                        {"role": "user", "content": f"Context:\n{tool_result}\n\nQuestion: {user_message}"}
                    ],
                    temperature=0,
                )
                return final_answer.choices[0].message.content

    response_message = response.choices[0].message

    # 2. did the model ask to call a tool?
    if not response_message.tool_calls:
        # no tool needed, this is already the final answer
        print("[no tool used]")
        return response_message.content

    # 3. the model wants to call one (or more) tools — run them ourselves
    messages.append(response_message)  # keep the assistant's tool-call request in history

    for tool_call in response_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"[tool call -> {tool_name}({tool_args})]")

        tool_function = AVAILABLE_TOOLS[tool_name]
        tool_result = tool_function(**tool_args)

        print(f"[tool result -> {tool_result[:100]}]")

        # send the tool's result back as a "tool" role message
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

    # ask for a concise final answer, matching the style used elsewhere in the project
    messages.append({
        "role": "system",
        "content": "Answer in 1-2 short sentences. Give only the direct answer, no extra detail."
    })

    for attempt in range(max_retries):
        try:
            final_response = client.chat.completions.create(
                model=FINAL_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="none",  # force a text answer now, don't call another tool
                temperature=0,
            )
            return final_response.choices[0].message.content
        except Exception as e:
            print(f"[final answer attempt {attempt + 1} had trouble, retrying]")
            if attempt == max_retries - 1:
                return f"(final answer failed after {max_retries} attempts, please try again)"


def run_cli():
    print("Day 16 Tool-Calling Chatbot. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = ask_with_tools(user_input)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    run_cli()