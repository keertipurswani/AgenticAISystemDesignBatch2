from educosys_claude.agent.factory import build_agent
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


def handle_query(question: str) -> str:
   """Entry point for all user queries - builds the agent and runs it."""
   logger.info(f"Handling query: {question}")
   agent = build_agent()
   response = agent.invoke({"messages": [{"role": "user", "content": question}]})
   return response["messages"][-1].content
   return response["output"]
