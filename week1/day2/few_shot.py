import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

prompt = """Turn messy sentences into clean, grammatically correct ones.

Messy: "i dont no where he go"
Clean: "I don't know where he went."

Messy: "she dont like it no more"
Clean: "She doesn't like it anymore."

Messy: "we was going to the store yesterday"
Clean: "We were going to the store yesterday."

Messy: "he dont never call me back"
Clean:"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)