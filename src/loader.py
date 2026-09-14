import pymupdf as fitz
from pathlib import Path


def load_pdf(file_path: str) -> str:
   
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_txt(file_path: str) -> str:
   
    return Path(file_path).read_text(encoding="utf-8")


def load_documents(folder_path: str) -> dict[str, str]:
   
    documents = {}
    folder = Path(folder_path)

    for file in folder.iterdir():
        try:
            if file.suffix.lower() == ".pdf":
                documents[file.name] = load_pdf(str(file))
            elif file.suffix.lower() == ".txt":
                documents[file.name] = load_txt(str(file))
        except Exception as e:
            
            print(f"⚠️ Skipped {file.name}: {e}")

    return documents




# ___________testing_____________


if __name__ == "__main__":
    docs = load_documents("data")
    for name, text in docs.items():
        print(f"{name}: {len(text)} characters loaded")