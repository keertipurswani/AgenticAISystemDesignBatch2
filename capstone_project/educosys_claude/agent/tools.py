from langchain.tools import tool

from educosys_claude.context.retrievers.semantic_chroma import retrieve
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)

@tool
def search_codebase(query: str) -> str:
   """
   Search the codebase for relevant classes, functions or logic.
   Use this tool whenever you need to find code related to a question.
   """
   logger.info(f"Tool called: search_codebase with query: {query}")
   chunks = retrieve(query, k=5)


   if not chunks:
       return "No relevant code found."


   # Format chunks into a readable string for the LLM
   results = []
   for chunk in chunks:
       results.append(
           f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
           f"Type: {chunk['type']} — {chunk['name']}\n"
           f"Code:\n{chunk['content']}\n"
       )
   return "\n---\n".join(results)
