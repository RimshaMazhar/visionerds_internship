from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')


sentences = [
    "The dog sitting on the table",
    "A feline rested on the rug",
    "I hate pancakes",
    "i dont like pancakes",
    "my phone is not working",
    "my mobile is dead"
]


embeddings = model.encode(sentences)


similarity_matrix = util.cos_sim(embeddings, embeddings)
print("Similarity Matrix:")
print(similarity_matrix)

def most_similar(query, sentences, threshold=0.3):
    query_emb = model.encode(query)
    sentence_embs = model.encode(sentences)
    scores = util.cos_sim(query_emb, sentence_embs)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx].item()
    
    if best_score < threshold:
        return None, best_score
    return sentences[best_idx], best_score

query = input("\nwrite your sentence: ")
result, score = most_similar(query, sentences)

if result is None:
    print(f"no match found. (Best score: {score:.4f})")
else:
    print(f"Most similar sentence: {result}")
    print(f"Score: {score:.4f}")