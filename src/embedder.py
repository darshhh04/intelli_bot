from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")  


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return model.encode(chunks).tolist()





# ___________testing_____________


if __name__ == "__main__":
    from src.loader import load_documents
    from src.chunker import chunk_text

    docs = load_documents("data")
    for name, text in docs.items():
        chunks = chunk_text(text)
        vectors = embed_chunks(chunks)
        print(f"{name}: {len(chunks)} chunks → {len(vectors)} vectors, dim={len(vectors[0])}")