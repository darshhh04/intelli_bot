
from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)
response = client.chat.completions.create(
    model="llama-3.3-70b-specdec",
    messages=[{"role": "user", "content": "Say hello in 5 words."}]
)
print(response.choices[0].message.content)