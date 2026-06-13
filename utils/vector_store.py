"""
FAISS Vector Store for Policy Document Retrieval
Uses free SentenceTransformers embeddings (no API key needed)
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PolicyVectorStore:
    """
    Manages FAISS index of payer policy documents.
    Uses all-MiniLM-L6-v2 for free local embeddings.
    """

    def __init__(
        self,
        embedding_model: str = None,
        index_path: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.index_path = Path(index_path or os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []   # Stores text + metadata for each chunk

        # Try loading existing index
        if self._index_exists():
            self.load_index()
        else:
            logger.info("No existing FAISS index found. Will build on first use.")
            self._init_empty_index()

    def _init_empty_index(self):
        """Create a fresh inner-product (cosine sim) index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []

    def _index_exists(self) -> bool:
        return (
            (self.index_path / "index.faiss").exists()
            and (self.index_path / "chunks.pkl").exists()
        )

    def _chunk_text(self, text: str, metadata: dict) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"text": chunk_text, "metadata": metadata})
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def add_policy(self, policy_id: str, payer: str, title: str, content: str, file_path: str = ""):
        """Add a policy document to the vector store."""
        metadata = {
            "policy_id": policy_id,
            "payer": payer,
            "title": title,
            "file_path": file_path,
        }
        new_chunks = self._chunk_text(content, metadata)

        # Encode chunks
        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

        logger.info(f"Added policy {policy_id}: {len(new_chunks)} chunks indexed.")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant policy chunks for a query.
        Returns list of {text, metadata, score} dicts.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty. No results returned.")
            return []

        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"],
                    "score": float(score),
                })

        return results

    def save_index(self):
        """Persist FAISS index and chunk metadata to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Index saved: {self.index.ntotal} vectors at {self.index_path}")

    def load_index(self):
        """Load FAISS index and chunk metadata from disk."""
        self.index = faiss.read_index(str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Index loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks.")

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0

    def get_policy_ids(self) -> List[str]:
        """Return unique policy IDs in the index."""
        return list({c["metadata"]["policy_id"] for c in self.chunks})
