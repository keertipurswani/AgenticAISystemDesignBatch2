import argparse
import ast
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEBASE = PROJECT_ROOT / "sample_project"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 32
TOP_K = 4

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}


# ---------------------------------------------------------------------------
# 1. LOAD & CHUNK
# ---------------------------------------------------------------------------

def load_codebase(repo_path: Path) -> list[Document]:
   docs = []
   for path in repo_path.rglob("*.py"):
       if any(part in SKIP_DIRS for part in path.parts):
           continue
       text = path.read_text(encoding="utf-8", errors="ignore")
       docs.append(Document(page_content=text, metadata={"source": str(path.relative_to(repo_path))}))
   return docs


def chunk_code(docs: list[Document]) -> list[Document]:
   splitter = RecursiveCharacterTextSplitter.from_language(
       language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
   )
   return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 2. BUILD INDEXES — vector + keyword + AST-based symbol/call lookups
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list[Document]) -> Chroma:
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
   return Chroma.from_documents(chunks, embedding=embeddings)


def build_bm25_index(chunks: list[Document]) -> tuple[BM25Okapi, list[Document]]:
   tokenized = [doc.page_content.lower().split() for doc in chunks]
   return BM25Okapi(tokenized), chunks


def build_symbol_index(repo_path: Path) -> dict[str, list[dict]]:
   """Map lowercase symbol name -> definitions (function/class name, file, line range)."""
   symbol_index = defaultdict(list)
   for path in repo_path.rglob("*.py"):
       if any(part in SKIP_DIRS for part in path.parts):
           continue
       try:
           tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
       except SyntaxError:
           continue
       for node in ast.walk(tree):
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
               symbol_index[node.name.lower()].append({
                   "name": node.name,
                   "file": str(path.relative_to(repo_path)),
                   "start_line": node.lineno,
                   "end_line": getattr(node, "end_lineno", node.lineno),
               })
   return symbol_index


def build_call_graph(repo_path: Path) -> dict[str, set[str]]:
   """Lightweight static call graph: function name -> names it calls (AST-based, no type resolution)."""
   call_graph = defaultdict(set)
   for path in repo_path.rglob("*.py"):
       if any(part in SKIP_DIRS for part in path.parts):
           continue
       try:
           tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
       except SyntaxError:
           continue
       for node in ast.walk(tree):
           if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
               continue
           for child in ast.walk(node):
               if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                   call_graph[node.name].add(child.func.id)
               elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                   call_graph[node.name].add(child.func.attr)
   return call_graph


# ---------------------------------------------------------------------------
# 3. TOOLS — the agent picks between these itself, and may call several
# ---------------------------------------------------------------------------

def build_tools(vector_store, bm25, bm25_docs, symbol_index, call_graph):

   @tool
   def semantic_search(query: str) -> str:
       """Search code by meaning (dense vector search). Use for conceptual questions like 'how does X work?'."""
       docs = vector_store.similarity_search(query, k=TOP_K)
       print(f"  [semantic_search] '{query}' -> {len(docs)} chunks")
       if not docs:
           return "No semantic results found."
       return "\n\n".join(f"# {d.metadata['source']}\n{d.page_content}" for d in docs)

   @tool
   def lexical_search(query: str) -> str:
       """Search code by exact keyword match (BM25). Use for exact names, error strings, constants."""
       tokens = query.lower().split()
       scores = bm25.get_scores(tokens)
       top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
       docs = [bm25_docs[i] for i in top]
       print(f"  [lexical_search] '{query}' -> {len(docs)} chunks")
       return "\n\n".join(f"# {d.metadata['source']}\n{d.page_content}" for d in docs)

   @tool
   def symbol_search(symbol: str) -> str:
       """Find the exact definition of a function or class by name."""
       key = symbol.strip().lower()
       matches = symbol_index.get(key) or [
           m for name, ms in symbol_index.items() if key in name for m in ms
       ]
       print(f"  [symbol_search] '{symbol}' -> {len(matches)} matches")
       if not matches:
           return f"No symbol found for: {symbol}"
       return "\n".join(f"{m['name']} — {m['file']}:{m['start_line']}-{m['end_line']}" for m in matches[:TOP_K])

   @tool
   def dependency_search(function_name: str) -> str:
       """List what a function calls, using a static AST call graph. Use for 'what does X call?'."""
       calls = call_graph.get(function_name) or next(
           (c for name, c in call_graph.items() if name.lower() == function_name.lower()), set()
       )
       print(f"  [dependency_search] '{function_name}' -> {len(calls)} calls")
       if not calls:
           return f"No dependencies found for {function_name}"
       return f"{function_name} calls:\n" + "\n".join(f"  -> {c}" for c in sorted(calls))

   return [semantic_search, lexical_search, symbol_search, dependency_search]


# ---------------------------------------------------------------------------
# 4. AGENT
# ---------------------------------------------------------------------------

def build_agent(tools):
   llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
   system_prompt = (
       "You are a senior engineer investigating a Python codebase. Investigate before answering — don't guess.\n\n"
       "Tools:\n"
       "- semantic_search: conceptual questions ('how does X work?')\n"
       "- lexical_search: exact names, error strings, constants\n"
       "- symbol_search: locate a function/class definition\n"
       "- dependency_search: what a function calls\n\n"
       "Use multiple tools and follow dependencies when one search isn't enough. "
       "Reference exact file, function, and class names in your answer. "
       "If the codebase doesn't contain the answer, say so — never fabricate code or relationships."
   )
   return create_agent(
       model=llm,
       tools=tools,
       system_prompt=system_prompt,
       middleware=[
           ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="semantic_search", run_limit=4, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="lexical_search", run_limit=4, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="symbol_search", run_limit=5, exit_behavior="end"),
           ToolCallLimitMiddleware(tool_name="dependency_search", run_limit=5, exit_behavior="end"),
       ],
   )


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Agentic RAG demo — an agent picks its own retrieval tools over a codebase.")
   parser.add_argument("--repo", type=Path, default=DEFAULT_CODEBASE)
   args = parser.parse_args()
   repo_path = args.repo.resolve()

   docs = load_codebase(repo_path)
   chunks = chunk_code(docs)
   print(f"Loaded {len(docs)} files → {len(chunks)} chunks")

   vector_store = build_vector_store(chunks)
   bm25, bm25_docs = build_bm25_index(chunks)
   symbol_index = build_symbol_index(repo_path)
   call_graph = build_call_graph(repo_path)
   print(f"Indexed {len(symbol_index)} symbols, {len(call_graph)} functions with dependencies")

   tools = build_tools(vector_store, bm25, bm25_docs, symbol_index, call_graph)
   agent = build_agent(tools)

   print("\nReady. Ask your question. Type 'exit' to quit.")
   while True:
       question = input("\nYou: ").strip()
       if not question or question.lower() in ("exit", "quit"):
           break

       for step in agent.stream(
           {"messages": [{"role": "user", "content": question}]},
           stream_mode="values",
       ):
           last_msg = step["messages"][-1]
           if getattr(last_msg, "tool_calls", None):
               for call in last_msg.tool_calls:
                   print(f"  → {call['name']}({call['args']})")
           elif last_msg.content:
               print(f"Agent: {last_msg.content}")
