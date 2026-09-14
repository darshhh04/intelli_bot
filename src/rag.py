from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY
from src.vector_store import search

client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are IntelliBot, a helpful assistant that answers questions
using ONLY the provided context from the user's documents.

Rules:
- If the answer is not in the context, say "I don't have that information in the provided documents."
- Do not use any outside knowledge, even if you know the answer.
- Keep answers concise and cite which document the info came from when possible.
"""


def build_context(chunks: list[str], sources: list[str]) -> str:

    parts = []
    for chunk, source in zip(chunks, sources):
        parts.append(f"[Source: {source}]\n{chunk}")
    return "\n\n".join(parts)


def ask(query: str, top_k: int = 3) -> str:

    results = search(query, top_k=top_k)
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]

    context = build_context(chunks, sources)

    user_prompt = f"""Context from documents:
{context}

Question: {query}

Answer using only the context above."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,  
        ),
    )

    return response.text




# ___________testing_____________

if __name__ == "__main__":
    query = "What projects are listed in the resume?"
    answer = ask(query)
    print(f"Q: {query}\nA: {answer}")