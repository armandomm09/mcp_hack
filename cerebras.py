from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
import json
import os

load_dotenv()

server_params = StdioServerParameters(
    command="uv",
    args=["run", "./server.py"],
)


CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

messages = {
    "messages": [
        {"role": "user", "content": "Can you show me points of interest near 48.8575° N, 2.3514° E?"},
    ]
}


async def main():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await load_mcp_tools(session)

            llm = ChatCerebras(
                model="llama-3.3-70b",
                temperature=0,
                max_tokens=10000,
                timeout=None,
                max_retries=1,
                api_key=CEREBRAS_API_KEY
            )
            agent = create_react_agent(
                llm,
                tools=tools,
                debug=True,
                prompt="""You are a helpful travel planner. When the user asks for travel options,
                            call exactly one tool (and no more). After calling that tool and getting its output,
                            stop reasoning and deliver the final answer. Do NOT think about calling any other tool. """,
            )
            agent_response = await agent.ainvoke(messages, stream_mode="debug")
            print(str(agent_response))


asyncio.run(main())
