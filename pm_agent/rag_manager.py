"""
RAG Manager — Retrieval-Augmented Generation for PM Platform documentation.
Uses FAISS for fast vector search and multilingual-e5-large for embeddings.
All operations are offline (local_files_only=True).
"""

import os
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGManager:
    def __init__(self, docs_dir: str, model_path: str):
        """
        Initializes the RAG Manager with FAISS index.
        :param docs_dir: Path to the directory containing .md documentation files.
        :param model_path: Path to the local folder with the embedding model.
        """
        self.docs_dir = docs_dir
        self.model_path = model_path
        self.documents: List[Dict[str, str]] = []  # [{title, content, path}]
        self.index: faiss.IndexFlatIP = None  # FAISS inner-product index (cosine sim on normalized vectors)

        print(f"Loading RAG Embedding Model from {model_path} (offline)...")
        try:
            self.model = SentenceTransformer(model_path, local_files_only=True)
            self._build_index()
        except Exception as e:
            print(f"Error loading RAG model: {e}")
            self.model = None

    # ------------------------------------------------------------------
    #  Indexing
    # ------------------------------------------------------------------

    def _build_index(self):
        """Reads all .md files, encodes them and builds a FAISS index."""
        if not os.path.exists(self.docs_dir):
            print(f"Docs directory not found: {self.docs_dir}")
            return

        self.documents = []
        texts_for_embedding: List[str] = []

        for root, _, files in os.walk(self.docs_dir):
            for file in sorted(files):
                if not file.endswith(".md"):
                    continue
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    title = os.path.splitext(file)[0]  # strip .md extension
                    self.documents.append({
                        "title": title,
                        "content": content,
                        "path": file_path,
                    })
                    # For embedding we prepend the title so the vector captures the topic
                    texts_for_embedding.append(f"{title}\n{content}")
                except Exception as e:
                    print(f"Error reading {file}: {e}")

        if not texts_for_embedding or not self.model:
            print("No documents indexed.")
            return

        print(f"Indexing {len(texts_for_embedding)} documents with FAISS...")
        embeddings = self.model.encode(texts_for_embedding, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product on L2-normalised = cosine sim
        self.index.add(embeddings)
        print(f"RAG index built: {self.index.ntotal} vectors, dim={dim}.")

    # ------------------------------------------------------------------
    #  Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Find most relevant documents for *query*.
        Returns list of dicts: {title, content, path, score, formatted}.
        """
        if not self.is_ready():
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        scores, idxs = self.index.search(query_vec, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue  # FAISS pad with -1 when fewer results than top_k
            doc = self.documents[idx]
            results.append({
                "title": doc["title"],
                "content": doc["content"],
                "path": doc["path"],
                "score": float(score),
                # Pre-formatted context block for LLM
                "formatted": (
                    f"--- ДОКУМЕНТ: {doc['title']} ---\n"
                    f"Путь: {doc['path']}\n\n"
                    f"{doc['content']}\n"
                    f"--- КОНЕЦ ДОКУМЕНТА ---"
                ),
            })

        return results

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self.model is not None and self.index is not None

    def reload(self):
        """Re-scan docs directory and rebuild index (useful if docs changed)."""
        if self.model:
            self._build_index()
