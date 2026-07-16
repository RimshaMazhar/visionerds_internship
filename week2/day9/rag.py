import sys
sys.path.append("../day7")
sys.path.append("../day8")

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


text = extract_text_from_pdf("the_house_on_briar_lane.pdf")
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_document")

embeddings = model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=ids
)
print("Chunks stored in vector store!\n")


def retrieve(question, k=3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    return results['documents'][0], results['distances'][0]
def ask_rag(question):
    retrieved_chunks, distances = retrieve(question, k=3)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are answering questions about a document using only the context provided below.

Instructions:
- If the question asks for a specific fact that is directly stated in the context, answer it directly and concisely.
- If the question requires connecting information that is explicitly present across multiple parts of the context, reason through it briefly and give a clear answer.
- If the exact answer is not explicitly stated in the context, simply say "This isn't explicitly stated in the document." Do not speculate, infer, or guess at length.
- Do not use outside knowledge beyond what's in the context.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions strictly based on the provided context."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    return answer, retrieved_chunks, distances


# ---- Main ----
while True:
    question = input("\nAsk your question (or write 'exit'): ")

    if question.lower() == "exit":
        print("Chatbot is closing, goodbye!")
        break

    answer, used_chunks,distances = ask_rag(question)

    print(f"\nAnswer: {answer}")

    print("\n--- Sources used(with relevance) ---")
    for i, chunk in enumerate(used_chunks):
        print(f"[{i+1}] {chunk[:100]}...")