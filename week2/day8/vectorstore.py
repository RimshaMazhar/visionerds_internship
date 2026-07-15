import sys
sys.path.append("../day7")

from chunking import extract_text_from_pdf, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('all-MiniLM-L6-v2')

text = extract_text_from_pdf("the_house_on_briar_lane.pdf")
chunks = chunk_text(text)
print(f"Total chunks: {len(chunks)}")

client = chromadb.Client()
collection = client.create_collection(name="my_document")

embeddings = model.encode(chunks).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=ids
)
print("Chunks stored in vector store!")


def retrieve(question, k=2):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    return results


while True:
    question = input("\nAsk your question (or type 'exit' to close): ")

    if question.lower() == "exit":
        print("Chatbot is closing!, goodbye")
        break

    results = retrieve(question, k=2)

    best_distance = results['distances'][0][0]
    threshold = 1.5

    if best_distance > threshold:
        print("\nSorry i could not find this in document.")
    else:
        print("\n--- Top matching chunks ---")
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"\nChunk {i+1} (distance: {distance:.4f}):")
            print(doc[:200] + "...")