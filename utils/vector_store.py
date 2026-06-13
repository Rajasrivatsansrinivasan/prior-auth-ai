"""
FAISS Vector Store for Policy Document Retrieval
Uses free SentenceTransformers embeddings (no API key needed)
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PolicyVectorStore:
    def __init__(self, embedding_model: str = None, index_path: str = None,
                 chunk_size: int = 150, chunk_overlap: int = 30):
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.index_path = Path(index_path or os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []

        if self._index_exists():
            self.load_index()
        else:
            self._init_empty_index()

    def _init_empty_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []

    def _index_exists(self) -> bool:
        return ((self.index_path / "index.faiss").exists() and
                (self.index_path / "chunks.pkl").exists())

    def _chunk_text(self, text: str, metadata: dict) -> List[Dict]:
        """Split on sentences first, then enforce word limit."""
        import re
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        chunks = []
        current_words = []
        current_sentences = []

        for sentence in sentences:
            words = sentence.split()
            if len(current_words) + len(words) > self.chunk_size and current_words:
                chunk_text = " ".join(current_words)
                chunks.append({"text": chunk_text, "metadata": metadata})
                # Overlap: keep last N words
                overlap_words = current_words[-self.chunk_overlap:]
                current_words = overlap_words + words
            else:
                current_words.extend(words)

        if current_words:
            chunks.append({"text": " ".join(current_words), "metadata": metadata})

        return chunks

    def add_policy(self, policy_id: str, payer: str, title: str, content: str, file_path: str = ""):
        metadata = {"policy_id": policy_id, "payer": payer, "title": title, "file_path": file_path}
        new_chunks = self._chunk_text(content, metadata)

        if not new_chunks:
            return

        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        logger.info(f"Added {policy_id}: {len(new_chunks)} chunks")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"],
                    "score": float(score),
                })
        return results

    def save_index(self):
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        self.index = faiss.read_index(str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0

    def get_policy_ids(self) -> List[str]:
        return list({c["metadata"]["policy_id"] for c in self.chunks})