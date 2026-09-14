import chromadb
from src.loader import load_documents
from src.chunker import chunk_text
from src.embedder import embed_chunks


client = chromadb.PersistentClient(path="chroma_db")


collection = client.get_or_create_collection(
    name="intellibot_docs",
    metadata={"hnsw:space": "cosine"},
)


def build_vector_store(folder_path: str = "data"):

    docs = load_documents(folder_path)

    for filename, text in docs.items():
        chunks = chunk_text(text)
        vectors = embed_chunks(chunks)

   
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]  

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=metadatas,
        )
        print(f"Stored {len(chunks)} chunks from {filename}")


def search(query: str, top_k: int = 3):
    """Embed the query and return the top_k most similar chunks."""
    query_vector = embed_chunks([query])[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )
    return results





# ___________testing_____________


if __name__ == "__main__":
    build_vector_store()


    test_query = "What is this document about?"
    results = search(test_query)

    print(f"\nTop results for: '{test_query}'")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"- [{meta['source']}] (distance={dist:.3f}) {doc[:80]}...")