import argparse
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEBASE = PROJECT_ROOT / "sample_project"

CODE_EXTENSIONS = {".py", ".ts", ".js", ".java", ".go", ".rs", ".md"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache"}

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687") 
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j") 
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") 

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# relationship between 2 nodes - edges
class CodeRelationship(BaseModel):
   subject: str = Field(description="Class, function, module, or file path")
   predicate: str = Field(
       description="Relationship type (e.g., DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, INHERITS_FROM)"
   )
   obj: str = Field(description="Target class, function, module, or file path")

# graph - list of edges/relationships
class GraphDocument(BaseModel):
   relationships: list[CodeRelationship] = Field(description="All code relationships extracted from the source file")

# Nodes
class Entities(BaseModel):
   names: list[str] = Field(
       description="Code entities in the query: class names, function names, modules, file paths"
   )

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_codebase(root: Path) -> list[tuple[str, str]]:
   root = root.resolve()
   if not root.is_dir():
       raise FileNotFoundError(f"Codebase path does not exist: {root}")
   files = []
   for path in sorted(root.rglob("*")):
       if not path.is_file():
           continue
       if path.suffix.lower() not in CODE_EXTENSIONS:
           continue
       if any(part in SKIP_DIRS for part in path.parts):
           continue
       files.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
   return files


# ---------------------------------------------------------------------------
# 2. EXTRACT RELATIONSHIPS & BUILD GRAPH
# ---------------------------------------------------------------------------

def extract_relationships(relationship_extractor, files: list[tuple[str, str]]) -> list[CodeRelationship]:
   relationships = []
   for rel_path, content in files:
       result = relationship_extractor.invoke(
           {"messages": [{"role": "user", "content": f"File: {rel_path}\n\n{content}"}]}
       )
       relationships.extend(result["structured_response"].relationships)
   return relationships


def sanitize_rel_type(predicate: str) -> str:
   """Neo4j relationship types must be valid identifiers — the LLM's predicate isn't guaranteed to be."""
   cleaned = re.sub(r"[^A-Za-z0-9_]", "_", predicate.strip().upper())
   cleaned = re.sub(r"_+", "_", cleaned).strip("_")
   return cleaned or "RELATED_TO"


def build_graph(driver, relationships: list[CodeRelationship]) -> None:
   with driver.session() as session:
       # make node lookups by name fast and prevent duplicate nodes with the same name
       session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
       session.run("MATCH (e:Entity) DETACH DELETE e")  # reset demo data from any previous run
       for r in relationships:
           rel_type = sanitize_rel_type(r.predicate)
           # MERGE = "get this node/relationship if it exists, else create it" (no duplicates)
           session.run(
               f"""
               MERGE (a:Entity {{name: $subject}})
               MERGE (b:Entity {{name: $obj}})
               MERGE (a)-[:{rel_type}]->(b)
               """,
               subject=r.subject.strip(),
               obj=r.obj.strip(),
           )


def graph_stats(driver) -> tuple[int, int]:
   # simple counts, used to print a summary after building the graph
   with driver.session() as session:
       nodes = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
       edges = session.run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c").single()["c"]
   return nodes, edges


def print_graph(driver) -> None:
   # fetch every edge in the graph, as (source, relation, target), for the printed summary
   with driver.session() as session:
       records = session.run(
           """
           MATCH (a:Entity)-[r]->(b:Entity)
           RETURN a.name AS source, type(r) AS relation, b.name AS target
           ORDER BY source, target
           """
       )
       for record in records:
           print(f"  {record['source']:35s} -[{record['relation']:20s}]-> {record['target']}")


def clear_graph(driver) -> None:
   # wipes every Entity node and relationship — used by the "delete" command in the Q&A loop
   with driver.session() as session:
       session.run("MATCH (e:Entity) DETACH DELETE e")


# ---------------------------------------------------------------------------
# 3. RETRIEVE
# ---------------------------------------------------------------------------
def match_nodes(session, entity: str) -> list[str]:
   # fuzzy-match an LLM-guessed entity name (e.g. "PaymentGateway") to real node names in the graph
   needle = entity.strip().lower()
   result = session.run(
       """
       MATCH (e:Entity)
       WHERE toLower(e.name) CONTAINS $needle OR $needle CONTAINS toLower(e.name)
       RETURN e.name AS name
       """,
       needle=needle,
   )
   return [record["name"] for record in result]


def graph_retrieve(driver, entity_extractor, query: str, depth: int = 2) -> str:
   # 1. ask the LLM which entities the user's question is about
   result = entity_extractor.invoke({"messages": [{"role": "user", "content": query}]})
   entities = result["structured_response"].names

   relationships = set()
   with driver.session() as session:
       # 2. for each entity, find its matching node(s) in Neo4j
       for entity in entities:
           for node_name in match_nodes(session, entity):
               # 3. walk out from that node up to `depth` hops in either direction,
               # collecting every relationship (edge) crossed along the way.
               # variable-length path length can't be parameterized in Cypher, so `depth`
               # (an int from argparse) is interpolated directly into the query string.
               records = session.run(
                   f"""
                   MATCH (start:Entity {{name: $name}})
                   MATCH path = (start)-[*1..{depth}]-(neighbour)
                   UNWIND relationships(path) AS rel
                   RETURN startNode(rel).name AS source, type(rel) AS relation, endNode(rel).name AS target
                   """,
                   name=node_name,
               )
               for record in records:
                   relationships.add(f"{record['source']} -[{record['relation']}]-> {record['target']}")

   # 4. format the collected edges into plain text context for the QA agent
   if not relationships:
       return "No relevant graph data found."
   return "Knowledge Graph context:\n" + "\n".join(sorted(relationships))



# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="GraphRAG demo - answer questions about a codebase using a knowledge graph stored in Neo4j.")
   parser.add_argument("--repo", type=Path, default=DEFAULT_CODEBASE)
   parser.add_argument("--depth", type=int, default=2, help="Graph neighbourhood radius (default: 2)")
   args = parser.parse_args()

   # step 1: load every source file from the target codebase
   files = load_codebase(args.repo.resolve())
   if not files:
       raise SystemExit(f"No source files found under {args.repo}")


   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   # 3 agents: one to extract (subject, predicate, object) facts from code,
   # one to pull entity names out of a user question, one to answer questions from context

   relationship_extractor = create_agent(
       model=llm,
       tools=[],
       response_format=GraphDocument,
       system_prompt=(
           "Extract code relationships from the source file as (subject, predicate, object) facts.\n"
           "Use ALL_CAPS predicates such as: DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, "
           "INHERITS_FROM, IMPLEMENTS, VALIDATES, SENDS_TO, CONFIGURES.\n"
           "Subjects and objects should be class names, function names, module paths, or file paths.\n"
           "Capture imports, constructor dependencies, method calls, and cross-module relationships.\n"
           "Be consistent: use the same name for the same class or module across relationships."
       ),
   )

   entity_extractor = create_agent(
       model=llm,
       tools=[],
       response_format=Entities,
       system_prompt="Extract code-related entities from the user message: class names, function names, module names, and file paths.",
   )

   qa_agent = create_agent(
       model=llm,
       tools=[],
       system_prompt=(
           "You are a codebase assistant. Answer ONLY from the context provided in the user message. "
           "Reference specific classes, files, and relationships when possible. "
           "If you cannot answer from context, say so."
       ),
   )


   # connect to Neo4j and check the connection works before doing any real work
   driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
   driver.verify_connectivity()

   try:
       # step 2: use the LLM to turn each file into (subject, predicate, object) facts
       print(f"Loaded {len(files)} files — extracting knowledge graph...")
       relationships = extract_relationships(relationship_extractor, files)


       # step 3: write those facts into Neo4j as nodes + relationships
       build_graph(driver, relationships)
       node_count, edge_count = graph_stats(driver)
       print(f"Graph: {node_count} nodes, {edge_count} edges (written to Neo4j at {NEO4J_URI})")


       print("\nKNOWLEDGE GRAPH — All extracted relationships:")
       print_graph(driver)


       # step 4: interactive Q&A loop — ask a question, retrieve graph context, get an answer
       print("\nReady. Ask your question. Type 'delete' to clear the graph, or 'exit' to quit.")
       while True:
           question = input("\nYou: ").strip()
           if not question or question.lower() in ("exit", "quit"):
               break

           if question.lower() in ("delete", "clear", "reset"):
               confirm = input("This will delete ALL nodes and relationships in the graph. Type 'yes' to confirm: ").strip()
               if confirm.lower() == "yes":
                   clear_graph(driver)
                   print("Graph cleared.")
               else:
                   print("Cancelled.")
               continue

           context = graph_retrieve(driver, entity_extractor, question, depth=args.depth)
           for step in qa_agent.stream(
               {"messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]},
               stream_mode="values",
           ):
               last_msg = step["messages"][-1]
               if not getattr(last_msg, "tool_calls", None):
                   print(f"Agent: {last_msg.content}")
   finally:
       driver.close()
