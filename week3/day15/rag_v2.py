"""
Day 15: RAG v2 — Conversational RAG
---------------------------------------
Combines everything from this week into ONE chatbot, reusing the same
building blocks already built in Week 2 and earlier this week:
  - chunking.py       (Week 2, day7)  -> PDF text extraction + chunking
  - vectorstore idea   (Week 2, day8)  -> ChromaDB + sentence-transformers embeddings
  - query rewriting     (Week 3, day12) -> turns follow-ups into standalone questions
  - buffer memory        (Week 3, day13) -> keeps last N messages

Flow for every new message:
  new question -> rewrite_query(history, new question) -> standalone question
  standalone question -> retrieve() -> relevant chunks from the vector store
  chunks + ORIGINAL conversational question -> final answer from the LLM
  (question, answer) -> added back into buffer memory
"""

import os
import sys
from collections import deque

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# reuse the exact chunking code from Week 2, day7
sys.path.append("../../week2/day7")
from chunking import extract_text_from_pdf, chunk_text

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"
BUFFER_SIZE = 8          # same as day13
DISTANCE_THRESHOLD = 2.0  # same "not found in document" cutoff as day8
PDF_PATH = "the_house_on_briar_lane.pdf"  # copy this into day15/ before running


# ---------- STEP 1: load, chunk, embed, store (Week 2 day7 + day8 logic) ----------

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


def retrieve(question, k=2):
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results


# ---------- STEP 2: query rewriting (same logic as day12) ----------

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
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# ---------- STEP 3: buffer memory (same as day13) ----------

class BufferMemory:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.history = deque(maxlen=buffer_size)

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get(self):
        return list(self.history)


# ---------- STEP 4: final answer generation using retrieved chunks ----------

def build_final_prompt(chunks_text, conversational_question):
    context = "\n\n---\n\n".join(chunks_text)
    return (
        "Answer the question using ONLY the context below. If the answer "
        "isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {conversational_question}\n\n"
        "Answer:"
    )


# ---------- STEP 5: the conversational loop, wiring everything together ----------

def run_chat():
    memory = BufferMemory()

    print("Day 15 Conversational RAG. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot is closing! Goodbye")
            break
        if not user_input:
            continue

        history = memory.get()

        # 1. rewrite the question to be standalone, using chat history
        if len(history) > 0:
            standalone_question = rewrite_query(history, user_input)
            print(f"[Rewritten query: {standalone_question}]")
        else:
            standalone_question = user_input
            print("[No history yet, using original question]")

        # 2. retrieve relevant chunks using the REWRITTEN question
        results = retrieve(standalone_question, k=3)
        best_distance = results['distances'][0][0]

        if best_distance > DISTANCE_THRESHOLD:
            answer = "Sorry, I could not find this in the document."
        else:
            retrieved_chunks = results['documents'][0]

            # 3. build final prompt using chunks + the ORIGINAL conversational question
            final_prompt = build_final_prompt(retrieved_chunks, user_input)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Keep your answers short, 2-3 sentences max."},
                    {"role": "user", "content": final_prompt}
                ]
            )
            answer = response.choices[0].message.content

        print(f"\nAssistant: {answer}\n")

        # 4. add both turns back into memory
        memory.add("user", user_input)
        memory.add("assistant", answer)


if __name__ == "__main__":
    run_chat()