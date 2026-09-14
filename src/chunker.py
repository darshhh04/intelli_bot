from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split one document's text into overlapping chunks.
    chunk_size=500 chars is a reasonable default — small enough for
    precise retrieval, big enough to keep a full idea together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)






# ___________testing_____________

if __name__ == "__main__":
    from src.loader import load_documents

    docs = load_documents("data")
    for name, text in docs.items():
        chunks = chunk_text(text)
        print(f"{name}: {len(chunks)} chunks")