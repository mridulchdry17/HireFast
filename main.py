# def main():
#     print("Hello from agentic-hr!")


# if __name__ == "__main__":
#     main()
import asyncio
import os
from dotenv import load_dotenv
from observee_agents import chat_with_tools_stream

load_dotenv()

async def stream_example():
    async for chunk in chat_with_tools_stream(
        message="create a new event in my calendar for tomorrow saying 'hello world' at 10:00am",
        provider="groq",
        model="openai/gpt-oss-120b",
        enable_filtering=True,
        expand_by_server=True,
        observee_api_key=os.getenv("OBSERVEE_API_KEY")
    ):
        if chunk["type"] == "content":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "final_content":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "tool_result":
            print(f"\n🔧 [Tool: {chunk['tool_name']}]")

if __name__ == "__main__":
    # Check required environment variables
    if not os.getenv("OBSERVEE_API_KEY"):
        print("❌ Please set OBSERVEE_API_KEY environment variable")
        exit(1)
    
    asyncio.run(stream_example())
