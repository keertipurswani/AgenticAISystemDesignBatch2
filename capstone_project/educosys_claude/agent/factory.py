from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from educosys_claude.llm.factory import get_llm
from educosys_claude.agent.tools import search_codebase
from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a senior software engineer with deep knowledge of the codebase.
Always use the search_codebase tool before answering any question.
Reference specific file names, function names and line numbers in your answers.
If you cannot find the answer in the codebase, say so explicitly."""


def build_agent():
  """Create and return a LangChain agent."""
  llm = get_llm()
  tools = [search_codebase]
  logger.info("Creating agent")
  return create_agent(
      llm,
      tools=tools,
      system_prompt=SYSTEM_PROMPT,
  )
