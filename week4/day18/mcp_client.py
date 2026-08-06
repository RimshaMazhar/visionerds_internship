"""
Day 18: MCP (Model Context Protocol)
------------------------------------------
In Day 16/17 we hand-wrote every tool (calculator, document_lookup,
word_count) and manually described each one to the model. MCP flips
this: an MCP SERVER already exposes a standard set of tools, and our
code just CONNECTS to it as a client. We don't write the tool
implementations, we don't write the JSON schemas by hand — the server
tells us what's available and how to call it.

Here we connect to the "filesystem" MCP server (an official, free,
public MCP server) which gives tools to list and read files. We plug
those MCP tools into the same kind of function-calling loop as Day 16,
but this time the tool definitions come from the MCP server, not from
us.

Compare this file to day16/tool_calling.py:
  - Day 16: we wrote TOOL_DEFINITIONS by hand, and AVAILABLE_TOOLS mapped
    names to our own Python functions.
  - Day 18: TOOL_DEFINITIONS come from session.list_tools() (the server
    describes itself), and running a tool means session.call_tool(...)
    instead of calling our own function.
"""

import os
import json
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

TOOL_MODEL = "llama-3.1-8b-instant"        # deciding which MCP tool to call
FINAL_MODEL = "llama-3.3-70b-versatile"     # generating the final answer (more reliable)

# the filesystem MCP server: an official example server that lets a
# model list/read files within a folder we allow it to see.
# "." means "the folder this script is run from" (day18/) — it will
# only be able to see files inside here, nothing outside.
SERVER_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "."],
)


def mcp_tool_to_openai_format(mcp_tool):
    """Convert an MCP tool description into the JSON schema format the
    OpenAI-compatible function-calling API expects. This is the part
    that replaces hand-writing TOOL_DEFINITIONS like we did in Day 16."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.input_schema,
        }
    }


async def run_chat():
    print("Connecting to the filesystem MCP server...")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ask the MCP server what tools it has — we didn't write these ourselves
            mcp_tools = await session.list_tools()
            tool_definitions = [mcp_tool_to_openai_format(t) for t in mcp_tools.tools]

            print(f"Connected! The server exposes {len(tool_definitions)} tools:")
            for t in tool_definitions:
                print(f"  - {t['function']['name']}: {t['function']['description'][:70]}")
            print()

            print("Day 18 MCP Chatbot. Type 'exit' to quit.")
            print("Try asking things like: 'what files are in this folder?' or")
            print("'read the file the_house_on_briar_lane.pdf' (if it's a text file)\n")

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                messages = [
                    {"role": "system", "content": (
                        "You are a helpful assistant with access to filesystem tools via MCP. "
                        "The current folder you have access to is represented by the path '.'  "
                        "Always use '.' as the path when the user refers to 'this folder' or "
                        "'the current folder'. Use these tools whenever the question is about "
                        "files or folders."
                    )},
                    {"role": "user", "content": user_input}
                ]

                response = client.chat.completions.create(
                    model=TOOL_MODEL,
                    messages=messages,
                    tools=tool_definitions,
                    tool_choice="auto",
                    temperature=0,
                )

                response_message = response.choices[0].message

                if not response_message.tool_calls:
                    print(f"Bot: {response_message.content}\n")
                    continue

                messages.append(response_message)

                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"[MCP tool call -> {tool_name}({tool_args})]")

                    # THIS is the MCP part: instead of calling our own Python
                    # function like in Day 16, we ask the MCP server to run it.
                    mcp_result = await session.call_tool(tool_name, tool_args)
                    result_text = "\n".join(
                        block.text for block in mcp_result.content if hasattr(block, "text")
                    )

                    print(f"[MCP result -> {result_text[:150]}]")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                messages.append({
                    "role": "system",
                    "content": "Answer in 1-2 short sentences using the tool result above. No extra detail."
                })

                final_response = client.chat.completions.create(
                    model=FINAL_MODEL,
                    messages=messages,
                    tools=tool_definitions,
                    tool_choice="none",
                    temperature=0,
                )
                print(f"Bot: {final_response.choices[0].message.content}\n")


if __name__ == "__main__":
    asyncio.run(run_chat())