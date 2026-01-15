import os
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class RAGManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", local_model_path: str = None):
        """
        Initialize RAG Manager.
        :param model_name: Name of the SentenceTransformer model.
        :param local_model_path: Path to the locally saved model (for offline use).
        """
        # Default local path
        default_local = os.path.join("models", model_name)
        if not local_model_path and os.path.exists(default_local):
            local_model_path = default_local

        self.model_path = local_model_path or model_name
        print(f"Loading embedding model from: {self.model_path}...")

        # Auto-detect device (GPU/CPU)
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.index = None
        self.documents = []
        self.metadata = []

    def load_excel(self, file_path: str):
        """
        Load process data from Excel file.
        Expected columns: 'Process ID', 'Process Name', 'Description' (optional)
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return False

        df = pd.read_excel(file_path)
        self.documents = []
        self.metadata = []

        for _, row in df.iterrows():
            proc_id = str(row.get("Process ID", ""))
            proc_name = str(row.get("Process Name", ""))
            description = str(row.get("Description", ""))

            # Combine for indexing
            text_to_index = f"ID: {proc_id} | Name: {proc_name} | Description: {description}"
            self.documents.append(text_to_index)

            # Store metadata for retrieval
            self.metadata.append({
                "id": proc_id,
                "name": proc_name,
                "description": description
            })

        print(f"Loaded {len(self.documents)} documents from {file_path}")
        self._initialize_index()
        return True

    def _initialize_index(self):
        """Create FAISS index from loaded documents."""
        if not self.documents:
            return

        embeddings = self.model.encode(self.documents)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("FAISS index initialized.")

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the RAG system for relevant documents.
        """
        if self.index is None or not text:
            return []

        query_embedding = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

    def get_context_string(self, query_text: str, top_k: int = 3) -> str:
        """
        Returns a formatted string of retrieved documents for LLM context.
        """
        results = self.query(query_text, top_k)
        if not results:
            return "No relevant process information found in RAG."

        context_lines = ["=== RELEVANT PROCESS INFORMATION (RAG) ==="]
        for res in results:
            context_lines.append(f"- ID: {res['id']}, Name: {res['name']}, Description: {res['description']}")
        context_lines.append("=== END OF RAG CONTEXT ===")

        return "\n".join(context_lines)
