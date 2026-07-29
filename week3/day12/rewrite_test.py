import sys
sys.path.append("../../week2/day7")
sys.path.append("../../week2/day8")

from chunking import extract_text_from_pdf, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- Setup: story load karo, chunk karo, vector store banao ----
text = extract_text_from_pdf("the_house_on_briar_lane.pdf")
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_document")

embeddings = model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(embeddings=embeddings, documents=chunks, ids=ids)
print("Chunks stored in vector store!\n")


def retrieve(question, k=5):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results['documents'][0], results['distances'][0]


def rewrite_query(chat_history, new_question):
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    history_text = ""
    for msg in recent_history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""Given the recent conversation history and a new question, decide if the new question is already a standalone question that makes sense on its own, or if it's a follow-up that depends on the conversation history to be understood.

If the new question is already standalone and clear on its own, return it EXACTLY as it is, unchanged.

If the new question is a follow-up that depends on context (uses words like "it", "that", "why though", "what about X instead"), rewrite it into a standalone version using the conversation history.

Recent conversation:
{history_text}

New question: {new_question}

Return ONLY the final question (either unchanged or rewritten), nothing else:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ---- Main loop ----
chat_history = []

print("Rewrite + Retrieval Test - ask your question about the story!")
print("(type 'exit' to quit)\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Closing. Goodbye!")
        break

    if len(chat_history) > 0:
        rewritten = rewrite_query(chat_history, question)
        print(f"[Rewritten query: {rewritten}]")
    else:
        rewritten = question
        print("[No history yet, using original question]")

   # Rewritten query se retrieval karo (raw follow-up nahi)
    retrieved_chunks, distances = retrieve(rewritten, k=3)

    best_distance = distances[0]
    threshold = 1.5

    if best_distance > threshold:
        print("\nThis isn't in the document's context.")
    else:
        print("\n--- Retrieved chunks (using rewritten query) ---")
        for i, (chunk, dist) in enumerate(zip(retrieved_chunks, distances)):
            print(f"[{i+1}] (distance: {dist:.4f}) {chunk[:100]}...")