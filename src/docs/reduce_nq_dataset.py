import json
import random
from pathlib import Path
from collections import defaultdict
import csv

# =========================
# CONFIG
# =========================
CORPUS_PATH = "benign/corpus.jsonl"
QUERIES_PATH = "benign/queries.jsonl"
QRELS_PATH = "./qrels/test.tsv"      # hoặc .csv nếu bạn dùng csv

OUT_CORPUS = "corpus.jsonl"
OUT_QUERIES = "new_queries.jsonl"

MAX_CORPUS = 500_000
SEED = 42

random.seed(SEED)

# =========================
# STEP 1: Load corpus
# =========================
print("[1] Loading corpus...")
corpus = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        corpus.append(json.loads(line))

print(f"    Total corpus docs: {len(corpus)}")

# =========================
# STEP 2: Sample corpus
# =========================
print(f"[2] Sampling {MAX_CORPUS} corpus documents...")

if len(corpus) > MAX_CORPUS:
    corpus_sample = random.sample(corpus, MAX_CORPUS)
else:
    corpus_sample = corpus

selected_doc_ids = {doc["_id"] for doc in corpus_sample}

print(f"    Selected corpus docs: {len(corpus_sample)}")

# =========================
# STEP 3: Read qrels with header
# =========================
print("[3] Loading qrels...")
selected_query_ids = set()

with open(QRELS_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")  # đổi delimiter nếu là comma
    for row in reader:
        qid = row["query-id"]
        did = row["corpus-id"]
        score = int(row["score"])

        if score > 0 and did in selected_doc_ids:
            selected_query_ids.add(qid)

print(f"    Queries linked to sampled corpus: {len(selected_query_ids)}")

# =========================
# STEP 4: Filter queries
# =========================
print("[4] Filtering queries...")
filtered_queries = []

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj["_id"] in selected_query_ids:
            filtered_queries.append(obj)

print(f"    Final queries: {len(filtered_queries)}")

# =========================
# STEP 5: Write outputs
# =========================
print("[5] Writing corpus.jsonl")
with open(OUT_CORPUS, "w", encoding="utf-8") as f:
    for doc in corpus_sample:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print("[6] Writing new_queries.jsonl")
with open(OUT_QUERIES, "w", encoding="utf-8") as f:
    for q in filtered_queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print("\n✅ DONE")
print(f"   Corpus:  {OUT_CORPUS} ({len(corpus_sample)})")
print(f"   Queries: {OUT_QUERIES} ({len(filtered_queries)})")
