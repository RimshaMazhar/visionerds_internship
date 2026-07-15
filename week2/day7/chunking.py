from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=80, overlap=30):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":
    text = extract_text_from_pdf("the_house_on_briar_lane.pdf")
    print(f"Total characters: {len(text)}\n")


    chunks = chunk_text(text, chunk_size=50, overlap=30)
    print(f"Total chunks: {len(chunks)}\n")

    print("--- Chunk 1 ---")
    print(chunks[0])
    print()

    print("--- Chunk 2 ---")
    print(chunks[1])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)
    print("--- Chunk 3 ---")
    print(chunks[2])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)
    
    print("--- Chunk 4 ---")
    print(chunks[3])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)
    print("--- Chunk 5 ---")
    print(chunks[4])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)
    print("--- Chunk 6 ---")
    print(chunks[5])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)
    print("--- Chunk 7 ---")
    print(chunks[6])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)
    print("--- Chunk 8 ---")
    print(chunks[7])
    chunks_small = chunk_text(text, chunk_size=60, overlap=15)
    chunks_big = chunk_text(text, chunk_size=300, overlap=50)

    print(f"\nSmall chunks total: {len(chunks_small)}")
    print(f"Big chunks total: {len(chunks_big)}")

    print("\n--- Small chunk example ---")
    print(chunks_small[0])

    print("\n--- Big chunk example ---")
    print(chunks_big[0])

