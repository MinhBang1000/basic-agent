# rag.py
import os
import json
import yaml
from glob import glob
from docx import Document  # pip install python-docx
from pdfminer.high_level import extract_text  # pip install pdfminer.six

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_file(filepath: str) -> str:
    """
    Extract readable text from many document formats.
    Add more formats anytime.
    """
    ext = filepath.lower().split(".")[-1]

    try:
        # ---------------- TEXT-LIKE FILES ----------------
        if ext in ["txt", "md", "log", "csv", "tsv"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        # ---------------- JSON ----------------
        if ext == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return json.dumps(obj, indent=2)

        # ---------------- YAML ----------------
        if ext in ["yaml", "yml"]:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return json.dumps(data, indent=2)

        # ---------------- PDF ----------------
        if ext == "pdf":
            return extract_text(filepath)

        # ---------------- DOCX ----------------
        if ext == "docx":
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])

        # ---------------- CODE FILES ----------------
        if ext in ["py", "js", "html", "css", "java", "cpp", "c", "ts", "go", "rs"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        # ---------------- UNKNOWN TYPES ----------------
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    except Exception as e:
        print(f"[RAG] Could not read {filepath}: {e}")
        return ""


def load_docs_from_folder(folder_path="docs"):
    patterns = ["**/*"]  # recursively include everything
    file_paths = []

    for pattern in patterns:
        file_paths.extend(glob(os.path.join(folder_path, pattern), recursive=True))

    docs = []
    for fp in file_paths:
        if os.path.isdir(fp):
            continue  # skip folders

        text = extract_text_from_file(fp)
        if not text.strip():
            continue

        docs.append(
            (
                text,
                {
                    "source": fp,
                    "filename": os.path.basename(fp),
                    "type": fp.split(".")[-1].lower(),
                },
            )
        )

    return docs


def setup_rag(docs_folder="docs", persist_dir=None):
    raw_docs = load_docs_from_folder(docs_folder)

    if not raw_docs:
        raise ValueError(f"No readable documents inside: {docs_folder}")

    texts = [t for (t, _) in raw_docs]
    metadatas = [m for (_, m) in raw_docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    all_chunks = []
    all_metadatas = []

    for text, meta in zip(texts, metadatas):
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadatas.extend([meta] * len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_db = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=persist_dir,
    )

    return chroma_db