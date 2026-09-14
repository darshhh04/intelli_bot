from src.rag import ask
from src.vector_store import search
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

TEST_QUESTIONS = [
    "What is IntelliBot's tech stack?",
    "What UI framework does the project use?",
    "What does the evaluation framework use?",
    "What LLM provider does the project use?",
    "What are the deliverables in this project?",
    "What should I eat today?",  # not in docs
]


JUDGE_PROMPT = """You are evaluating a RAG chatbot's answer.

Context provided to the bot:
{context}

Question: {question}
Bot's answer: {answer}

Score two things from 0.0 to 1.0:
1. Faithfulness: Is every claim in the answer actually supported by the context?
   (If the bot correctly said "I don't have that information" for an unanswerable
   question, that counts as faithfulness = 1.0)
2. Relevancy: Does the answer actually address what was asked?

Respond in EXACTLY this format, nothing else:
faithfulness: <score>
relevancy: <score>
"""


def judge_answer(question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),  
    )
    scores = {}
    for line in response.text.strip().split("\n"):
        key, val = line.split(":")
        scores[key.strip()] = float(val.strip())
    return scores


def run_evaluation():
    results = []
    for q in TEST_QUESTIONS:
        retrieved = search(q, top_k=3)
        context = "\n\n".join(retrieved["documents"][0])
        answer = ask(q, history=[])
        scores = judge_answer(q, context, answer)
        results.append({"question": q, "answer": answer, **scores})

    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["relevancy"] for r in results) / len(results)

    print(f"\n{'='*50}")
    for r in results:
        print(f"Q: {r['question']}")
        print(f"A: {r['answer'][:100]}...")
        print(f"Faithfulness: {r['faithfulness']} | Relevancy: {r['relevancy']}\n")
    print(f"AVERAGE — Faithfulness: {avg_faithfulness:.2f} | Relevancy: {avg_relevancy:.2f}")

    return results


if __name__ == "__main__":
    run_evaluation()