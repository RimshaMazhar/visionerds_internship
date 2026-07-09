import os
from dotenv import load_dotenv
from openai import OpenAI
from config import SYSTEM_PROMPT

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)




print("Chat started! Type 'exit' to end the conversation.\n")


while True:
    user_input = input("You: ")
    
    
    if user_input.lower() == "exit":
        print("Chat ended.")
        break
    
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )
    
    
    print("AI:", response.choices[0].message.content)
    print()