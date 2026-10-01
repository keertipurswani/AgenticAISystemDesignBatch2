from dotenv import load_dotenv
load_dotenv()
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import os

async def run_agent():
 client = MultiServerMCPClient(
     {
       "EducosysFileSystem": {
               "command": "python",
               "args": [
                   "./filesystem_mcp.py"
               ],
               "transport":"stdio"
        }
     }
 )

 tools = await client.get_tools()
 agent = create_agent("openai:gpt-4o", tools)
 response = await agent.ainvoke({"messages": "create file sandeep.txt in present directory"})
 print(response["messages"][-1].content)


if __name__ == "__main__":
 asyncio.run(run_agent())
