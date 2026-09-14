from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
   
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