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
- Use the conversation history to understand follow-up questions (e.g. "what about X" refers back to the prior topic).
- Keep answers concise and cite which document the info came from when possible.
"""


chat_history = []


def build_context(chunks: list[str], sources: list[str]) -> str:
    parts = []
    for chunk, source in zip(chunks, sources):
        parts.append(f"[Source: {source}]\n{chunk}")
    return "\n\n".join(parts)


def build_history_text(history: list[dict]) -> str:

    if not history:
        return "No previous conversation."
    lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in history]
    return "\n".join(lines)


def ask(query: str, top_k: int = 3) -> str:
    results = search(query, top_k=top_k)
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    context = build_context(chunks, sources)

    history_text = build_history_text(chat_history)

    user_prompt = f"""Conversation so far:
{history_text}

Context from documents:
{context}

Question: {query}

Answer using only the context above, and use the conversation history to
understand what the question is really asking if it's a follow-up."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )

    answer = response.text


    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer


def clear_history():

    chat_history.clear()




# ___________testing_____________



if __name__ == "__main__":
    print(ask("What is IntelliBot's tech stack?"))
    print(ask("What about the evaluation framework specifically?"))