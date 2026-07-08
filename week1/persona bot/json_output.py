import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

system_prompt = """You are a data extraction tool. 
Extract the name, city, and intent from the user's message.
Respond ONLY with valid JSON in this exact format, no explanation, no markdown, no extra text:
{"name": "", "city": "", "intent": ""}"""

user_message = "Hi, I'm Ahmed from Lahore, I want to book a table for dinner tonight."

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
)

print(response.choices[0].message.content)