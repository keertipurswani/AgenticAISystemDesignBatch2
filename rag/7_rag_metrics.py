from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import math

load_dotenv()

# ── Knowledge base ────────────────────────────────────────────────────────────
corpus = [
   Document(page_content="Full refund available within 30 days of purchase date.",              metadata={"id": "doc_1"}),
   Document(page_content="Digital products are refundable only if defective or non-functional.",     metadata={"id": "doc_2"}),
   Document(page_content="Refund processing takes 5 to 7 business days after approval.",             metadata={"id": "doc_3"}),
   Document(page_content="Submit refund requests through the support portal at support.acme.com.",    metadata={"id": "doc_4"}),
   Document(page_content="Express shipping takes 2 business days. Standard shipping takes 5-7 days.",   metadata={"id": "doc_5"}),
   Document(page_content="Free standard shipping on all orders over $50.",                            metadata={"id": "doc_6"}),
   Document(page_content="Account passwords must be 12 characters minimum with uppercase, number, special char.", metadata={"id": "doc_7"}),
   Document(page_content="Account passwords expire every 90 days. Cannot reuse last 5 passwords.",            metadata={"id": "doc_8"}),
   Document(page_content="The company was founded in 2010 and serves over 1 million customers.",       metadata={"id": "doc_9"}),  # distractor — never relevant
   Document(page_content="Loyalty program points expire 12 months after the date they were earned.",   metadata={"id": "doc_10"}),
]

# ── Golden dataset ────────────────────────────────────────────────────────────
# Each entry: query + the doc IDs that are TRULY relevant to that query.
# "Truly relevant" means the doc's content actually answers the query — not
# just same-topic. (e.g. doc_2 is about refund eligibility, not timing,
# so it belongs under "what's the refund policy?", not "how long do I have?")
golden_set = [
   {"query": "What is Acme's refund policy?",
    "relevant_ids": ["doc_1", "doc_2"]},   # both docs needed for a complete policy answer
   {"query": "How do I submit a refund request?",
    "relevant_ids": ["doc_4"]},
   {"query": "How long does refund processing take?",
    "relevant_ids": ["doc_3"]},
   {"query": "What are the shipping options and costs?",
    "relevant_ids": ["doc_5", "doc_6"]},
   {"query": "What are the password requirements?",
    "relevant_ids": ["doc_7", "doc_8"]},
   {"query": "How long before my loyalty points expire?",
    "relevant_ids": ["doc_10"]},
]

# ── Build vector store ────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs = InMemoryVectorStore.from_documents(corpus, embeddings)

def num_hits_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> int:
   relevant = set(relevant_ids)
   return sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)

def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
   return num_hits_at_k(retrieved_ids, relevant_ids, k) / len(relevant_ids) if relevant_ids else 0.0

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
   return num_hits_at_k(retrieved_ids, relevant_ids, k) / k if k > 0 else 0.0

def mrr(retrieved_ids: list, relevant_ids: list) -> float:
   relevant = set(relevant_ids)
   for rank, doc_id in enumerate(retrieved_ids, start=1):
       if doc_id in relevant:
           return 1.0 / rank
   return 0.0

def hit_rate_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
   top_k = set(retrieved_ids[:k])
   return 1.0 if set(relevant_ids) & top_k else 0.0

def ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
   relevant = set(relevant_ids)
   dcg = sum(
       1.0 / math.log2(rank + 1)
       for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
       if doc_id in relevant
   )
   ideal_hits = min(len(relevant), k)
   idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
   return dcg / idcg if idcg > 0 else 0.0

results_by_k = {}

for K in [1, 3, 5]:
   results = []
   print(f"\n{'─'*80}")
   print(f"  K = {K}")
   print(f"{'─'*80}")
   print(f"  {'Query':48s} {'R@K':6s} {'P@K':6s} {'MRR':6s} {'Hit':6s} {'nDCG':6s}")
   print(f"  {'-'*80}")

   for item in golden_set:
       retrieved_docs = vs.similarity_search(item["query"], k=K)
       retrieved_ids  = [doc.metadata["id"] for doc in retrieved_docs]


       r  = recall_at_k(retrieved_ids, item["relevant_ids"], K)
       p  = precision_at_k(retrieved_ids, item["relevant_ids"], K)
       m  = mrr(retrieved_ids, item["relevant_ids"])
       hr = hit_rate_at_k(retrieved_ids, item["relevant_ids"], K)
       nd = ndcg_at_k(retrieved_ids, item["relevant_ids"], K)

       results.append({"query": item["query"], "relevant_ids": item["relevant_ids"],
                        "retrieved_ids": retrieved_ids,
                        "recall": r, "precision": p, "mrr": m, "hit_rate": hr, "ndcg": nd})
      
       print(f"  {item['query'][:48]:48s} {r:.2f}   {p:.2f}   {m:.2f}   {hr:.2f}   {nd:.2f}")


   print(f"  {'─'*80}")
   for metric in ["recall", "precision", "mrr", "hit_rate", "ndcg"]:
       avg = sum(r[metric] for r in results) / len(results)
       label = "✅" if avg >= 0.7 else ("⚠️ " if avg >= 0.5 else "❌")
       print(f"    {label} Mean {metric:12s}: {avg:.4f}")


   results_by_k[K] = results

# ── Failure analysis (K=3) ────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print("  Failure analysis  (K=3, recall < 1.0)")
print(f"{'─'*80}")
failures = [r for r in results_by_k[3] if r["recall"] < 1.0]
if failures:
   for r in failures:
       missed = set(r["relevant_ids"]) - set(r["retrieved_ids"])
       print(f"  Query   : {r['query']}")
       print(f"  Retrieved: {r['retrieved_ids']}")
       print(f"  Missed  : {sorted(missed)}")
       print()
else:
   print("  No failures — all queries achieved recall = 1.0 at K=3")









