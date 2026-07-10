# Day 3 – Chat Loop, Config File & Git Branching

## What this covers
Turning the basic API script into an interactive chat loop, moving the system prompt into a separate config file, and practicing Git branching.

## Files
- **chat_loop.py** — Lets you have a back-and-forth conversation with the model in the terminal. Type `exit` to end the chat. Each message is treated fresh (no memory yet).
- **config.py** — Stores the system prompt separately, so the bot's persona can be changed without touching the main script.

## Setup
1. Install dependencies:
   pip install openai python-dotenv

2. Create a .env file with your API keys:
   GROQ_API_KEY=your_key_here

3. Run the chat loop:
   python chat_loop.py

## What I learned
- How to build a continuous chat loop using a while loop and user input
- Separating configuration (system prompt) from logic makes the bot easier to customize
- Git branching workflow: creating a branch, committing changes, and pushing it separately from main
